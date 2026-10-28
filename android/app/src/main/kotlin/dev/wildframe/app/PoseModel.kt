package dev.wildframe.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import dev.wildframe.engine.Config
import dev.wildframe.engine.Detection
import dev.wildframe.engine.Letterbox
import dev.wildframe.engine.PoseDecoder
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** ONNX Runtime wrapper for the YOLO26n AP-10K pose model (assets/wildframe_pose.onnx, the
 * INT8 export with the keypoint branch kept fp32). Mirrors wildframe/pose/backend.py: letterbox
 * to a square canvas padded with 114, RGB / 255, NCHW; decode + NMS in [PoseDecoder]. */
class PoseModel(context: Context, cfg: Config) : AutoCloseable {
    private val size = cfg.int("input_size", 640)
    private val numKeypoints = cfg.int("num_keypoints", 17)
    private val confThreshold = cfg.double("conf_threshold", 0.25).toFloat()
    private val iouThreshold = cfg.double("iou_threshold_nms", 0.5).toFloat()
    private val maxDet = cfg.int("max_det", 10)

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String

    private val canvasBitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
    private val canvas = Canvas(canvasBitmap)
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val pixels = IntArray(size * size)
    private val input = ByteBuffer.allocateDirect(4 * 3 * size * size).order(ByteOrder.nativeOrder()).asFloatBuffer()

    init {
        val bytes = context.assets.open("wildframe_pose.onnx").use { it.readBytes() }
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(bytes, options)
        inputName = session.inputNames.first()
    }

    fun detect(frame: Bitmap): List<Detection> {
        val lb = Letterbox(frame.width, frame.height, size)
        canvas.drawColor(Color.rgb(114, 114, 114))
        val dst = RectF(lb.padX.toFloat(), lb.padY.toFloat(), (lb.padX + lb.newW).toFloat(), (lb.padY + lb.newH).toFloat())
        canvas.drawBitmap(frame, null, dst, paint)
        canvasBitmap.getPixels(pixels, 0, size, 0, 0, size, size)

        val plane = size * size
        for (i in 0 until plane) {
            val p = pixels[i]
            input.put(i, ((p shr 16) and 0xFF) / 255f)
            input.put(plane + i, ((p shr 8) and 0xFF) / 255f)
            input.put(2 * plane + i, (p and 0xFF) / 255f)
        }
        input.rewind()

        val (values, anchors) = OnnxTensor.createTensor(env, input, longArrayOf(1, 3, size.toLong(), size.toLong())).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { result ->
                val out = result.get(0) as OnnxTensor
                val buf = out.floatBuffer
                val values = FloatArray(buf.remaining())
                buf.get(values)
                values to out.info.shape[2].toInt()
            }
        }
        return PoseDecoder.decode(values, anchors, numKeypoints, lb, confThreshold, iouThreshold, maxDet)
    }

    override fun close() {
        session.close()
    }
}
