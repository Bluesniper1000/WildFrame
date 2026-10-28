"""Static INT8 quantisation of a YOLO-pose ONNX export, ported from ASTMA-wildlife pose/quantize.py.

  quant_format   QOperator: Conv -> QLinearConv, MatMul -> QLinearMatMul (no QDQ pairs)
  types          uint8 asymmetric activations, int8 per-channel symmetric weights (ORT's CPU recommendation)
  calibration    MinMax over N images drawn with a fixed seed from the AP-10K TRAIN split (never val/test)
  exclusions     --exclude-regex cv4 keeps the YOLO-pose keypoint branch (9 convs) in fp32 ("kpfp32");
                 quantising it is what collapses n-scale YOLO-pose accuracy

Preprocessing is WildFrame's own letterbox (wildframe.pose.backend), so calibration sees exactly the
tensors the deployed pipeline feeds the model. A manifest JSON with the settings, the calibration
image-list hash and the op census is written next to the output model.

  python -m bench.quantize --model ap10k_ft/weights_quant/best_fp32.onnx \
      --out ap10k_ft/weights_quant/best_int8_kpfp32.onnx --exclude-regex cv4 \
      --data ../ASTMA-wildlife/pose/data/ap10k
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

# onnxruntime.tools imports torch whenever it's installed, and torch's DLLs are blocked by Windows
# Application Control on the dev machine. Quantisation doesn't need torch: a None entry makes
# importlib.util.find_spec("torch") report it absent, so ORT skips its torch helpers.
sys.modules.setdefault("torch", None)

import cv2  # noqa: E402
import onnx  # noqa: E402
import onnxruntime  # noqa: E402
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static  # noqa: E402
from onnxruntime.quantization.shape_inference import quant_pre_process  # noqa: E402

from wildframe.pose.backend import REPO_ROOT, _letterbox  # noqa: E402


def calib_images(calib_json: Path, images_dir: Path, n: int, seed: int) -> list[str]:
    """Deterministic sample of n image paths from a COCO-style split."""
    d = json.loads(calib_json.read_text())
    imgs = {i["id"]: i for i in d["images"]}
    ids = sorted(imgs)
    random.Random(seed).shuffle(ids)
    return [str(images_dir / imgs[i]["file_name"]) for i in ids[:n]]


class _Reader(CalibrationDataReader):
    def __init__(self, paths: list[str], input_name: str, imgsz: int):
        self.paths, self.input_name, self.imgsz, self.i = paths, input_name, imgsz, 0

    def get_next(self):
        if self.i >= len(self.paths):
            return None
        img = cv2.imread(self.paths[self.i])
        self.i += 1
        if img is None:
            raise FileNotFoundError(self.paths[self.i - 1])
        return {self.input_name: _letterbox(img, self.imgsz)[0]}

    def rewind(self):
        self.i = 0


def op_census(path: str | Path) -> dict[str, int]:
    return dict(collections.Counter(n.op_type for n in onnx.load(str(path)).graph.node))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="fp32 ONNX")
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=str(REPO_ROOT / "data/ap10k"), help="AP-10K root (annotations/, data/)")
    p.add_argument("--calib-split", default="ap10k-train-split1.json")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--ops", nargs="+", default=["Conv", "MatMul"])
    p.add_argument("--exclude-regex", default=None, help="keep matching Conv/MatMul nodes fp32, e.g. 'cv4'")
    p.add_argument("--calib-method", choices=["MinMax", "Percentile", "Entropy"], default="MinMax")
    a = p.parse_args()

    src, out = Path(a.model), Path(a.out)
    prep = out.with_name(out.stem + "_prep.onnx")
    try:
        quant_pre_process(str(src), str(prep), skip_symbolic_shape=False)
        symbolic = True
    except Exception:  # noqa: BLE001 - ORT symbolic shape inference fails on TopK in YOLO26 NMS-free heads
        # (TypeError) and on the dynamic-shape YOLO26n export ("Incomplete symbolic shape inference")
        quant_pre_process(str(src), str(prep), skip_symbolic_shape=True)
        symbolic = False

    exclude = sorted(
        n.name for n in onnx.load(str(prep)).graph.node
        if a.exclude_regex and n.op_type in a.ops and re.search(a.exclude_regex, n.name)
    )
    input_name = onnxruntime.InferenceSession(str(prep), providers=["CPUExecutionProvider"]).get_inputs()[0].name
    data = Path(a.data)
    paths = calib_images(data / "annotations" / a.calib_split, data / "data", a.n, a.seed)

    t0 = time.time()
    quantize_static(
        str(prep), str(out), _Reader(paths, input_name, a.imgsz),
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=getattr(CalibrationMethod, a.calib_method),
        op_types_to_quantize=a.ops,
        nodes_to_exclude=exclude,
    )
    prep.unlink(missing_ok=True)

    census_fp32, census_int8 = op_census(src), op_census(out)
    onnxruntime.InferenceSession(str(out), providers=["CPUExecutionProvider"])  # must load on the CPU EP
    manifest = {
        "fp32_model": src.name, "int8_model": out.name,
        "quant_format": "QOperator", "activation_type": "QUInt8", "weight_type": "QInt8", "per_channel": True,
        "calibrate_method": a.calib_method, "op_types_to_quantize": a.ops,
        "exclude_regex": a.exclude_regex, "nodes_to_exclude": exclude, "symbolic_shape_inference": symbolic,
        "calib_split": a.calib_split, "calib_n": len(paths), "calib_seed": a.seed,
        "calib_items_sha256": hashlib.sha256(json.dumps([Path(x).name for x in paths]).encode()).hexdigest(),
        "onnxruntime": onnxruntime.__version__, "onnx": onnx.__version__,
        "conv_fp32": census_fp32.get("Conv", 0), "qlinearconv_int8": census_int8.get("QLinearConv", 0),
        "conv_left_fp32": census_int8.get("Conv", 0), "op_census_int8": census_int8,
        "size_mb": {"fp32": round(src.stat().st_size / 1e6, 2), "int8": round(out.stat().st_size / 1e6, 2)},
        "seconds": round(time.time() - t0, 1),
    }
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("int8_model", "conv_fp32", "qlinearconv_int8", "conv_left_fp32", "size_mb", "seconds")}))
    if census_int8.get("QLinearConv", 0) == 0 or census_int8.get("Conv", 0) != len(exclude):
        raise SystemExit(f"expected {len(exclude)} Conv kept fp32, found {census_int8.get('Conv', 0)}")


if __name__ == "__main__":
    main()
