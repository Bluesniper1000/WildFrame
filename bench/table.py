"""Build markdown tables from bench/run.py summary JSONs.

  python -m bench.table results/bench/*.json > results/bench/tables.md

INT8 rows get deltas against the fp32 row of the same --family.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bench.ap10k import KEYPOINT_NAMES, RULE_KEYPOINTS


def _f(x, nd=3, signed=False):
    if x is None:
        return "–"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def _delta(row: dict, ref: dict | None, get) -> float | None:
    if ref is None:
        return None
    a, b = get(row), get(ref)
    return None if a is None or b is None else a - b


def _prec(r: dict) -> str:
    return r["precision"] + (f" ({r['variant']})" if r.get("variant") else "")


def _lat(entry: dict | None) -> str:
    return f"{entry['model_only_ms']['median']:.1f} / {entry['e2e_ms']['median']:.1f}" if entry else "–"


def main_table(rows: list[dict], latency_1t: dict[str, dict]) -> str:
    by_key = {(r["family"], r["precision"]): r for r in rows}
    lines = [
        "| Model | Prec. | Input | AP | AP50 | AP75 | Eye PCK@0.05 | Eye NME | Orient. acc (L/R) | Vis. AUROC | Params (M) | GFLOPs | CPU ms model / e2e (N thr) | CPU ms model / e2e (1 thr) | ΔAP | ΔEye NME | ΔOrient. |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        ref = by_key.get((r["family"], "fp32")) if r["precision"] == "int8" else None
        lat_s = _lat(r.get("latency")) + " | " + _lat(latency_1t.get(r["name"]))
        params = r.get("params_m") or (ref or {}).get("params_m")
        gflops = r.get("gflops") or (ref or {}).get("gflops")
        lines.append(" | ".join([
            f"| {r['family']}", _prec(r), r.get("input_res", "–"),
            _f(r["ap"]["AP"]), _f(r["ap"]["AP50"]), _f(r["ap"]["AP75"]),
            _f(r["eye_pck"]), _f(r["eye_nme"], 4), _f(r["orientation"]["lr_accuracy"]),
            _f(r["visibility_auroc"]["pooled"]),
            _f(params, 2), _f(gflops, 1), lat_s,
            _f(_delta(r, ref, lambda x: x["ap"]["AP"]), signed=True),
            _f(_delta(r, ref, lambda x: x["eye_nme"]), 4, signed=True),
            _f(_delta(r, ref, lambda x: x["orientation"]["lr_accuracy"]), signed=True) + " |",
        ]))
    return "\n".join(lines)


def keypoint_table(rows: list[dict], field: str, nd: int) -> str:
    names = [KEYPOINT_NAMES[k] for k in RULE_KEYPOINTS]
    lines = ["| Model | Prec. | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for r in rows:
        src = r["visibility_auroc"]["per_keypoint"] if field == "auroc" else r[field]
        lines.append(f"| {r['family']} | {_prec(r)} | " +" | ".join(_f(src[n], nd) for n in names) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summaries", nargs="+")
    p.add_argument("--latency-1t", help="JSON of {summary name: {model_only_ms, e2e_ms}} measured at 1 thread")
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252, which has no "Δ"
    latency_1t = json.loads(Path(a.latency_1t).read_text()) if a.latency_1t else {}
    loaded = [json.loads(Path(s).read_text()) for s in a.summaries if not s.endswith(".preds.json")]
    rows = [r for r in loaded if "ap" in r]  # skip non-summary JSONs a glob may pick up
    rows.sort(key=lambda r: (r["family"], r["precision"] != "fp32", r.get("variant", "")))

    hw = next((r["hardware"] for r in rows if "hardware" in r), None)
    print("## Pose benchmark (AP-10K " + rows[0]["split"] + ", AP-10K OKS sigmas)\n")
    print("Keypoint metrics are over predictions matched to GT by box IoU >= 0.5 at score >= "
          f"{rows[0]['matching']['min_score']} (the deployment threshold); orientation and AUROC likewise.")
    if hw:
        print(f"Latency: {hw['cpu']}, N = {hw['threads']} threads (and 1 thread where given), {hw['runtime']}, "
              f"batch {hw['batch']}, median ms. e2e = letterbox + run + decode/NMS at the deployment settings.")
    print()
    print(main_table(rows, latency_1t))
    print("\n### Per-keypoint PCK@0.05 (rule keypoints)\n")
    print(keypoint_table(rows, "pck", 3))
    print("\n### Per-keypoint NME (rule keypoints)\n")
    print(keypoint_table(rows, "nme", 4))
    print("\n### Per-keypoint visibility AUROC (rule keypoints)\n")
    print(keypoint_table(rows, "auroc", 3))


if __name__ == "__main__":
    main()
