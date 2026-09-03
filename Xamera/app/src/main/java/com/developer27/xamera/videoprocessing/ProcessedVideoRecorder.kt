package com.developer27.xamera.videoprocessing

import android.graphics.Bitmap
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import android.os.SystemClock
import java.io.File

/** Encodes frames through YUV input images; codec surfaces do not support lockCanvas. */
class ProcessedVideoRecorder(
    private val width: Int,
    private val height: Int,
    private val outputPath: String,
    private val frameRate: Int = 30,
    private val bitRateMultiplier: Int = 5
) {
    private var codec: MediaCodec? = null
    private var muxer: MediaMuxer? = null
    private var trackIndex = -1
    private var muxerStarted = false
    private var framesWritten = 0
    private var startedAtNs = 0L
    private var lastPresentationTimeUs = -1L

    fun start() {
        check(codec == null) { "Recorder already started" }
        require(width > 0 && height > 0 && width % 2 == 0 && height % 2 == 0)
        try {
            val format = MediaFormat.createVideoFormat("video/avc", width, height).apply {
                setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
                setInteger(MediaFormat.KEY_BIT_RATE, width * height * bitRateMultiplier)
                setInteger(MediaFormat.KEY_FRAME_RATE, frameRate)
                setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
            }
            val encoder = MediaCodec.createEncoderByType("video/avc")
            codec = encoder // Retain ownership even if configure/start fails.
            encoder.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            encoder.start()
            muxer = MediaMuxer(outputPath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            startedAtNs = System.nanoTime()
            lastPresentationTimeUs = -1
            framesWritten = 0
        } catch (error: Exception) {
            release()
            File(outputPath).delete()
            throw error
        }
    }

    fun recordFrame(bitmap: Bitmap) {
        val encoder = checkNotNull(codec) { "Recorder is not started" }
        drainEncoder(false)
        val index = encoder.dequeueInputBuffer(10_000)
        if (index < 0) return // Drop a frame when the encoder is busy, without blocking the camera.
        val scaled = Bitmap.createScaledBitmap(bitmap, width, height, true)
        try {
            val pixels = IntArray(width * height)
            scaled.getPixels(pixels, 0, width, 0, 0, width, height)
            val input = checkNotNull(encoder.getInputImage(index)) { "Encoder has no YUV input image" }
            val planes = input.planes
            val starts = planes.map { it.buffer.position() }
            fun write(plane: Int, x: Int, y: Int, value: Int) {
                val target = planes[plane]
                target.buffer.put(starts[plane] + y * target.rowStride + x * target.pixelStride, value.coerceIn(0, 255).toByte())
            }
            for (y in 0 until height) {
                for (x in 0 until width) {
                    val pixel = pixels[y * width + x]
                    val r = pixel shr 16 and 255
                    val g = pixel shr 8 and 255
                    val b = pixel and 255
                    write(0, x, y, ((66 * r + 129 * g + 25 * b + 128) shr 8) + 16)
                    if (x % 2 == 0 && y % 2 == 0) {
                        write(1, x / 2, y / 2, ((-38 * r - 74 * g + 112 * b + 128) shr 8) + 128)
                        write(2, x / 2, y / 2, ((112 * r - 94 * g - 18 * b + 128) shr 8) + 128)
                    }
                }
            }
            val timeUs = ((System.nanoTime() - startedAtNs) / 1_000).coerceAtLeast(lastPresentationTimeUs + 1)
            encoder.queueInputBuffer(index, 0, width * height * 3 / 2, timeUs, 0)
            lastPresentationTimeUs = timeUs
            framesWritten++
        } finally {
            if (scaled !== bitmap) scaled.recycle()
        }
        drainEncoder(false)
    }

    /** Returns false for an empty recording, which is deleted instead of exported. */
    fun stop(): Boolean {
        val encoder = codec ?: return false
        var complete = false
        try {
            if (framesWritten == 0) return false
            val deadline = SystemClock.elapsedRealtime() + 2_000
            var index = encoder.dequeueInputBuffer(10_000)
            while (index < 0 && SystemClock.elapsedRealtime() < deadline) {
                drainEncoder(false)
                index = encoder.dequeueInputBuffer(10_000)
            }
            check(index >= 0) { "Encoder timed out finishing video" }
            encoder.queueInputBuffer(index, 0, 0, lastPresentationTimeUs + 1, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
            drainEncoder(true)
            check(muxerStarted) { "Encoder produced no video" }
            muxer?.stop()
            muxerStarted = false
            complete = true
            return true
        } finally {
            release()
            if (!complete) File(outputPath).delete()
        }
    }

    /** Releases failed recordings without waiting for end of stream. */
    fun cancel() {
        release()
        File(outputPath).delete()
    }

    private fun drainEncoder(endOfStream: Boolean) {
        val encoder = codec ?: return
        val info = MediaCodec.BufferInfo()
        val deadline = SystemClock.elapsedRealtime() + 2_000
        while (true) {
            when (val index = encoder.dequeueOutputBuffer(info, if (endOfStream) 10_000 else 0)) {
                MediaCodec.INFO_TRY_AGAIN_LATER -> {
                    if (!endOfStream) return
                    check(SystemClock.elapsedRealtime() < deadline) { "Encoder timed out draining video" }
                }
                MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    check(!muxerStarted) { "Encoder format changed twice" }
                    trackIndex = checkNotNull(muxer).addTrack(encoder.outputFormat)
                    muxer?.start()
                    muxerStarted = true
                }
                else -> if (index >= 0) {
                    try {
                        if (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) info.size = 0
                        if (info.size > 0) {
                            check(muxerStarted)
                            val buffer = checkNotNull(encoder.getOutputBuffer(index))
                            buffer.position(info.offset)
                            buffer.limit(info.offset + info.size)
                            muxer?.writeSampleData(trackIndex, buffer, info)
                        }
                    } finally {
                        encoder.releaseOutputBuffer(index, false)
                    }
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) return
                }
            }
        }
    }

    private fun release() {
        codec?.let { encoder ->
            try { encoder.stop() } catch (_: Exception) { }
            try { encoder.release() } catch (_: Exception) { }
        }
        codec = null
        muxer?.let { writer ->
            if (muxerStarted) try { writer.stop() } catch (_: Exception) { }
            try { writer.release() } catch (_: Exception) { }
        }
        muxer = null
        muxerStarted = false
        trackIndex = -1
    }
}
