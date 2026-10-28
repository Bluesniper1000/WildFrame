package dev.wildframe.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.SystemClock
import android.text.Layout
import android.text.StaticLayout
import android.text.TextPaint
import android.view.View
import dev.wildframe.engine.Decision
import dev.wildframe.engine.RuleName
import dev.wildframe.engine.RuleState
import dev.wildframe.engine.Scene
import dev.wildframe.engine.Skeleton
import kotlin.math.max

/** One analysed frame, as handed from the analysis thread to the UI thread. */
class FrameState(val frameW: Int, val frameH: Int, val decision: Decision, val elapsedMs: Long)

/** Draws the thirds grid, each member's skeleton, the subject box, and a banner with the single
 * instruction the engine chose (plus the transient "Now tracking: ..." notice). Frame-to-view
 * mapping matches PreviewView's FILL_CENTER. */
class OverlayView(context: Context) : View(context) {
    private var state: FrameState? = null
    private var error: String? = null
    private var notice: String? = null
    private var noticeUntil = 0L
    private var fps = 0.0
    private var lastFrameAt = 0L

    private val density = resources.displayMetrics.density
    private val gridPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0x55FFFFFF; strokeWidth = 1f * density }
    private val bonePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE; strokeWidth = 2f * density }
    private val jointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.YELLOW }
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 3f * density }
    private val bannerPaint = Paint()
    private val textPaint = TextPaint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE; textSize = 18f * density }
    private val smallPaint = TextPaint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE; textSize = 13f * density }

    fun update(s: FrameState) {
        val now = SystemClock.elapsedRealtime()
        if (lastFrameAt > 0) fps = 0.8 * fps + 0.2 * (1000.0 / max(1L, now - lastFrameAt))
        lastFrameAt = now
        s.decision.modeNotice?.let { notice = it; noticeUntil = now + 2500 }
        state = s
        error = null
        invalidate()
    }

    fun showError(message: String) {
        error = message
        invalidate()
    }

    private fun stateColor(chosen: dev.wildframe.engine.RuleOutput?): Int = when {
        chosen == null -> 0xCC333333.toInt()
        chosen.rule == RuleName.READY && chosen.state == RuleState.SATISFIED -> 0xE01B8A3A.toInt()
        chosen.state == RuleState.VIOLATED -> 0xE0C62828.toInt()
        chosen.state == RuleState.SATISFIED -> 0xE02E7D32.toInt()
        else -> 0xE0B26A00.toInt()
    }

    private fun drawBanner(canvas: Canvas, text: String, color: Int, top: Float, paint: TextPaint): Float {
        val pad = 12f * density
        val width = (getWidth() - 2 * pad).toInt().coerceAtLeast(1)
        val layout = StaticLayout.Builder.obtain(text, 0, text.length, paint, width)
            .setAlignment(Layout.Alignment.ALIGN_NORMAL).build()
        bannerPaint.color = color
        val bottom = top + layout.height + 2 * pad
        canvas.drawRect(0f, top, getWidth().toFloat(), bottom, bannerPaint)
        canvas.save()
        canvas.translate(pad, top + pad)
        layout.draw(canvas)
        canvas.restore()
        return bottom
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        error?.let { drawBanner(canvas, it, 0xE0C62828.toInt(), 0f, textPaint); return }
        val s = state ?: run { drawBanner(canvas, "Starting camera…", 0xCC333333.toInt(), 0f, textPaint); return }

        val scale = max(width.toFloat() / s.frameW, height.toFloat() / s.frameH)
        val dx = (width - s.frameW * scale) / 2f
        val dy = (height - s.frameH * scale) / 2f
        fun x(v: Float) = dx + v * scale
        fun y(v: Float) = dy + v * scale

        // Rule-of-thirds grid of the analysed frame (what eye_thirds measures against).
        for (i in 1..2) {
            canvas.drawLine(x(s.frameW * i / 3f), y(0f), x(s.frameW * i / 3f), y(s.frameH.toFloat()), gridPaint)
            canvas.drawLine(x(0f), y(s.frameH * i / 3f), x(s.frameW.toFloat()), y(s.frameH * i / 3f), gridPaint)
        }

        val d = s.decision
        for (m in d.subjectSet.members) {
            val k = m.keypoints
            for ((a, b) in Skeleton.EDGES) {
                if (k[a][2] >= 0.25f && k[b][2] >= 0.25f) canvas.drawLine(x(k[a][0]), y(k[a][1]), x(k[b][0]), y(k[b][1]), bonePaint)
            }
            for (p in k) if (p[2] >= 0.25f) canvas.drawCircle(x(p[0]), y(p[1]), 3f * density, jointPaint)
        }
        Scene.compositeSubject(d.subjectSet)?.let { subj ->
            boxPaint.color = stateColor(d.chosen) or 0xFF000000.toInt()
            canvas.drawRect(x(subj.bbox[0]), y(subj.bbox[1]), x(subj.bbox[2]), y(subj.bbox[3]), boxPaint)
        }

        var top = 0f
        val message = d.chosen?.message ?: "Point the camera at an animal."
        top = drawBanner(canvas, message, stateColor(d.chosen), top, textPaint)
        if (notice != null && SystemClock.elapsedRealtime() < noticeUntil) {
            top = drawBanner(canvas, notice!!, 0xAA000000.toInt(), top, smallPaint)
        }

        val footer = String.format("%s · %d ms · %.1f fps", d.subjectSet.mode.value, s.elapsedMs, fps)
        canvas.drawText(footer, 12f * density, height - 16f * density, smallPaint)
    }
}
