package com.developer27.xamera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.media.MediaMetadataRetriever
import android.os.Build
import android.provider.MediaStore
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.developer27.xamera.videoprocessing.KalmanHelper
import com.developer27.xamera.videoprocessing.MediaExporter
import com.developer27.xamera.videoprocessing.ProcessedFrameRecorder
import com.developer27.xamera.videoprocessing.ProcessedVideoRecorder
import com.developer27.xamera.videoprocessing.Settings
import com.developer27.xamera.videoprocessing.VideoProcessor
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.Point
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

@RunWith(AndroidJUnit4::class)
class ProcessingRegressionTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun settingsRestoreSavedValuesAndMatchXmlDefaults() {
        val prefs = context.getSharedPreferences("regression-settings", Context.MODE_PRIVATE)
        try {
            prefs.edit().clear().commit()
            Settings.load(prefs)
            assertEquals(Settings.DetectionMode.Mode.CONTOUR, Settings.DetectionMode.current)
            assertFalse(Settings.ExportData.videoDATA)
            assertFalse(Settings.ExportData.frameIMG)
            prefs.edit().putString("detection_mode", "YOLO")
                .putBoolean("enable_bounding_box", false)
                .putBoolean("enable_raw_trace", true)
                .putBoolean("enable_spline_trace", false)
                .putBoolean("video_data", true).putBoolean("frame_img", true).commit()
            Settings.load(prefs)
            assertEquals(Settings.DetectionMode.Mode.YOLO, Settings.DetectionMode.current)
            assertTrue(Settings.DetectionMode.enableYOLOinference)
            assertFalse(Settings.BoundingBox.enableBoundingBox)
            assertTrue(Settings.Trace.enableRAWtrace)
            assertFalse(Settings.Trace.enableSPLINEtrace)
            assertTrue(Settings.ExportData.videoDATA)
            assertTrue(Settings.ExportData.frameIMG)
        } finally {
            prefs.edit().clear().commit()
            Settings.load(prefs)
        }
    }

    @Test
    fun pngExportHasPngSignatureAndPublishesCompletedMedia() {
        val file = File.createTempFile("XameraRegression_", ".png", context.cacheDir)
        val bitmap = Bitmap.createBitmap(28, 28, Bitmap.Config.ARGB_8888).apply { eraseColor(Color.WHITE) }
        try {
            assertTrue(ProcessedFrameRecorder(file.path).save(bitmap))
            assertArrayEquals(byteArrayOf(-119, 80, 78, 71, 13, 10, 26, 10), file.readBytes().take(8).toByteArray())
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val uri = MediaExporter.publish(context, file, false)
                try {
                    context.contentResolver.query(uri, arrayOf(MediaStore.MediaColumns.IS_PENDING, MediaStore.MediaColumns.MIME_TYPE), null, null, null)!!.use {
                        assertTrue(it.moveToFirst())
                        assertEquals(0, it.getInt(0))
                        assertEquals("image/png", it.getString(1))
                    }
                    assertArrayEquals(file.readBytes(), context.contentResolver.openInputStream(uri)!!.use { it.readBytes() })
                } finally { context.contentResolver.delete(uri, null, null) }
            }
        } finally {
            bitmap.recycle()
            file.delete()
        }
    }

    @Test(timeout = 15_000)
    fun recorderProducesPlayableVideoAndHandlesEmptyStop() {
        val file = File.createTempFile("XameraRegression_", ".mp4", context.cacheDir)
        val recorder = ProcessedVideoRecorder(128, 128, file.path)
        val bitmap = Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888)
        try {
            recorder.start()
            assertFalse(recorder.stop())
            assertFalse(file.exists())
            assertFalse(recorder.stop())
            recorder.start()
            repeat(12) {
                bitmap.eraseColor(Color.rgb(200, 30 + it, 20))
                recorder.recordFrame(bitmap)
            }
            assertTrue(recorder.stop())
            val retriever = MediaMetadataRetriever()
            try {
                retriever.setDataSource(file.path)
                assertEquals("128", retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH))
                assertEquals("128", retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT))
                val frame = retriever.getFrameAtTime(0)!!
                val pixel = frame.getPixel(64, 64)
                assertTrue("Encoded color should remain red", Color.red(pixel) > Color.green(pixel) + 80)
                frame.recycle()
            } finally { retriever.release() }
        } finally {
            recorder.cancel()
            bitmap.recycle()
            file.delete()
        }
    }

    @Test
    fun kalmanResetStartsAtNewStrokeInsteadOfPreviousPosition() {
        System.loadLibrary("opencv_java4")
        val filter = KalmanHelper()
        filter.initKalmanFilter()
        assertEquals(100.0 to 200.0, filter.applyKalmanFilter(Point(100.0, 200.0)))
        repeat(10) { filter.applyKalmanFilter(Point(110.0 + it, 200.0)) }
        filter.reset()
        assertEquals(600.0 to 700.0, filter.applyKalmanFilter(Point(600.0, 700.0)))
    }

    @Test
    fun contourTraceResetAndIndependentProcessors() {
        Settings.DetectionMode.current = Settings.DetectionMode.Mode.CONTOUR
        val first = VideoProcessor()
        val second = VideoProcessor()
        val bitmap = Bitmap.createBitmap(320, 320, Bitmap.Config.ARGB_8888)
        try {
            repeat(5) { index ->
                bitmap.eraseColor(Color.BLACK)
                Canvas(bitmap).drawRect(50f + index * 15, 70f, 140f + index * 15, 170f, Paint().apply { color = Color.WHITE })
                val frames = first.processFrame(bitmap)
                assertNotNull(frames)
                frames!!.first.recycle()
            }
            assertTrue(first.hasTrace())
            assertFalse(second.hasTrace())
            val trace = first.exportTraceForInference()
            assertEquals(28, trace.width)
            assertEquals(28, trace.height)
            val pixels = IntArray(28 * 28)
            trace.getPixels(pixels, 0, 28, 0, 0, 28, 28)
            assertTrue("A tracked stroke should produce ink", pixels.any { Color.red(it) < 128 })
            assertTrue("Exported ink must be opaque", pixels.all { Color.alpha(it) == 255 })
            trace.recycle()
            first.reset()
            assertFalse(first.hasTrace())
            assertEquals("", first.getTrackingCoordinatesString())
        } finally {
            bitmap.recycle()
            first.close()
            second.close()
        }
    }

    @Test
    fun bundledModelsRunOnCpuWithExpectedInputAndOutputShapes() {
        fun load(name: String): Interpreter {
            val bytes = context.assets.open(name).use { it.readBytes() }
            val buffer = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
            buffer.put(bytes).rewind()
            return Interpreter(buffer, Interpreter.Options().setNumThreads(2))
        }
        for ((name, classes) in listOf("DigitRecog_float32.tflite" to 10, "LetterRecog_float32.tflite" to 26)) {
            load(name).use { model ->
                assertEquals(28 * 28 * 4, model.getInputTensor(0).numBytes())
                val input = ByteBuffer.allocateDirect(28 * 28 * 4).order(ByteOrder.nativeOrder())
                repeat(28 * 28) { input.putFloat(1f) }
                input.rewind()
                val output = Array(1) { FloatArray(classes) }
                model.run(input, output)
                assertTrue(output[0].all { it.isFinite() })
            }
        }
        val processor = VideoProcessor()
        val bitmap = Bitmap.createBitmap(320, 240, Bitmap.Config.ARGB_8888).apply { eraseColor(Color.BLACK) }
        try {
            processor.setInterpreter(load("YOLOv3_float32.tflite"))
            Settings.DetectionMode.current = Settings.DetectionMode.Mode.YOLO
            Settings.DetectionMode.enableYOLOinference = true
            val frames = processor.processFrame(bitmap)
            assertNotNull(frames)
            assertEquals(processor.getModelDimensions().first, frames!!.second.width)
            frames.first.recycle()
            frames.second.recycle()
        } finally {
            Settings.DetectionMode.current = Settings.DetectionMode.Mode.CONTOUR
            Settings.DetectionMode.enableYOLOinference = false
            bitmap.recycle()
            processor.close()
        }
    }
}
