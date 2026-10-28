package dev.wildframe.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import dev.wildframe.engine.ByteTracker
import dev.wildframe.engine.Config
import dev.wildframe.engine.Engine

/** Per camera frame (on the analysis thread): rotate upright -> pose model -> tracker -> engine,
 * then hand an immutable [FrameState] to the overlay. Configs are the app's copies of
 * configs/rules.yaml and configs/model.yaml (assets, kept in sync by golden/generate.py). */
class WildFrameAnalyzer(private val context: Context, private val overlay: OverlayView) : ImageAnalysis.Analyzer, AutoCloseable {
    private var model: PoseModel? = null
    private lateinit var engine: Engine
    private lateinit var tracker: ByteTracker

    private fun asset(name: String) = context.assets.open(name).use { it.readBytes().decodeToString() }

    private fun ensureLoaded(): PoseModel = model ?: run {
        val modelCfg = Config.parse(asset("model.json"))
        engine = Engine(Config.parse(asset("rules.json")))
        tracker = ByteTracker(modelCfg)
        PoseModel(context, modelCfg).also { model = it }
    }

    override fun analyze(image: ImageProxy) {
        try {
            val pose = ensureLoaded()
            val t0 = SystemClock.elapsedRealtime()

            // Rotate to display orientation: left/right facing (lead_space) must be judged in
            // the frame the photographer sees, not the sensor's native orientation.
            val raw = image.toBitmap()
            val rotation = image.imageInfo.rotationDegrees
            val frame = if (rotation == 0) raw else
                Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, Matrix().apply { postRotate(rotation.toFloat()) }, true)

            val detections = pose.detect(frame)
            val ids = tracker.update(detections.map { it.bbox })
            val tracked = detections.mapIndexed { i, d -> d.withTrackId(ids[i]) }
            val decision = engine.decide(tracked, frame.width, frame.height)
            val ms = SystemClock.elapsedRealtime() - t0

            val state = FrameState(frame.width, frame.height, decision, ms)
            overlay.post { overlay.update(state) }
        } catch (t: Throwable) {
            Log.e("WildFrame", "frame analysis failed", t)
            overlay.post { overlay.showError("${t.javaClass.simpleName}: ${t.message}") }
        } finally {
            image.close()
        }
    }

    override fun close() {
        model?.close()
        model = null
    }
}
