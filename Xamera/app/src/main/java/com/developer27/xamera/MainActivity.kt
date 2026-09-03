package com.developer27.xamera

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.ActivityNotFoundException
import android.content.SharedPreferences
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.graphics.SurfaceTexture
import android.net.Uri
import android.os.Bundle
import android.os.Build
import androidx.preference.PreferenceManager
import android.util.Log
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import com.developer27.xamera.camera.CameraHelper
import com.developer27.xamera.databinding.ActivityMainBinding
import com.developer27.xamera.videoprocessing.MediaExporter
import com.developer27.xamera.videoprocessing.ProcessedFrameRecorder
import com.developer27.xamera.videoprocessing.ProcessedVideoRecorder
import com.developer27.xamera.videoprocessing.Settings
import com.developer27.xamera.videoprocessing.VideoProcessor
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var sharedPreferences: SharedPreferences
    private lateinit var cameraHelper: CameraHelper
    private var tfliteInterpreter: Interpreter? = null
    // Interpreter for letter recognition.
    private var letterInterpreter: Interpreter? = null

    private var processedVideoRecorder: ProcessedVideoRecorder? = null
    private var recordingFile: File? = null
    private val processingExecutor = Executors.newSingleThreadExecutor()
    private var sessionGeneration = 0
    private var isFinalizing = false
    private var isResumed = false
    private var modelsReady = false
    private var videoProcessor: VideoProcessor? = null

    // Flag for tracking (start/stop tracking mode)
    private var isRecording = false
    // Flag for frame processing
    private var isProcessing = false
    private var isProcessingFrame = false

    // Stores the current session tracking coordinates.
    private var trackingCoordinates: String = ""

    // For toggling digit/letter recognition.
    var isLetterSelected = true
    var isDigitSelected = !isLetterSelected

    // Flag for writing mode.
    private var isWriting = false

    // Flag to clear prediction when returning from an external intent.
    private var shouldClearPrediction = false

    // NEW: Accumulated handwriting coordinates (each element corresponds to one letter)
    private val accumulatedCoordinates = mutableListOf<String>()

    private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    private lateinit var requestPermissionLauncher: ActivityResultLauncher<Array<String>>

    private val textureListener = object : TextureView.SurfaceTextureListener {
        @SuppressLint("MissingPermission")
        override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
            if (allPermissionsGranted()) {
                cameraHelper.openCamera()
            }
        }
        override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {}
        override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
            cameraHelper.closeCamera()
            return true
        }
        override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
            if (isProcessing) {
                processFrameWithVideoProcessor()
            }
        }
    }

    @SuppressLint("MissingPermission")
    override fun onCreate(savedInstanceState: Bundle?) {
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        installSplashScreen()
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this)
        PreferenceManager.setDefaultValues(this, R.xml.root_preferences, false)
        Settings.load(sharedPreferences)

        cameraHelper = CameraHelper(this, viewBinding, sharedPreferences)
        videoProcessor = VideoProcessor()
        viewBinding.viewFinder.surfaceTextureListener = textureListener

        viewBinding.processedFrameView.visibility = View.GONE
        viewBinding.predictedLetterTextView.text = "No Prediction Yet"

        viewBinding.titleContainer.setOnClickListener {
            val url = "https://www.zhangxiao.me/"
            val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
            launchExternalActivity(intent)
        }

        requestPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
            if (allPermissionsGranted()) {
                if (isResumed && viewBinding.viewFinder.isAvailable) cameraHelper.openCamera()
            } else {
                Toast.makeText(this, "Camera permission is required for tracking.", Toast.LENGTH_SHORT).show()
            }
        }
        if (!allPermissionsGranted()) requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)

        viewBinding.startProcessingButton.setOnClickListener {
            if (isRecording) {
                stopProcessingAndRecording()
            } else {
                startProcessingAndRecording()
            }
        }

        viewBinding.switchCameraButton.setOnClickListener { switchCamera() }
        viewBinding.aboutButton.setOnClickListener {
            startActivity(Intent(this, AboutXameraActivity::class.java))
        }
        viewBinding.settingsButton.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }

        // Set up the letter/digit switch.
        val letterDigitSwitch = viewBinding.letterDigitSwitch
        if (isLetterSelected) {
            letterDigitSwitch.setTextColor(android.graphics.Color.parseColor("#FFCB05"))
            letterDigitSwitch.thumbTintList = android.content.res.ColorStateList.valueOf(
                android.graphics.Color.parseColor("#FFCB05")
            )
            letterDigitSwitch.trackTintList = android.content.res.ColorStateList.valueOf(
                android.graphics.Color.parseColor("#FFCB05")
            )
            letterDigitSwitch.text = "Letter"
        } else {
            letterDigitSwitch.setTextColor(android.graphics.Color.parseColor("#FFFFFF"))
            letterDigitSwitch.thumbTintList = android.content.res.ColorStateList.valueOf(
                android.graphics.Color.parseColor("#FFFFFF")
            )
            letterDigitSwitch.trackTintList = android.content.res.ColorStateList.valueOf(
                android.graphics.Color.parseColor("#FFFFFF")
            )
            letterDigitSwitch.text = "Digit"
        }
        letterDigitSwitch.isChecked = isLetterSelected

        letterDigitSwitch.setOnCheckedChangeListener { _, isChecked ->
            isLetterSelected = isChecked
            isDigitSelected = !isChecked
            if (isChecked) {
                letterDigitSwitch.setTextColor(android.graphics.Color.parseColor("#FFCB05"))
                letterDigitSwitch.thumbTintList = android.content.res.ColorStateList.valueOf(
                    android.graphics.Color.parseColor("#FFCB05")
                )
                letterDigitSwitch.trackTintList = android.content.res.ColorStateList.valueOf(
                    android.graphics.Color.parseColor("#FFCB05")
                )
                letterDigitSwitch.text = "Letter"
            } else {
                letterDigitSwitch.setTextColor(android.graphics.Color.parseColor("#FFFFFF"))
                letterDigitSwitch.thumbTintList = android.content.res.ColorStateList.valueOf(
                    android.graphics.Color.parseColor("#FFFFFF")
                )
                letterDigitSwitch.trackTintList = android.content.res.ColorStateList.valueOf(
                    android.graphics.Color.parseColor("#FFFFFF")
                )
                letterDigitSwitch.text = "Digit"
            }
        }

        // Set up the Start Writing button.
        viewBinding.startWritingButton.setOnClickListener {
            toggleWritingMode()
        }

        // Set up the Clear Prediction button ("C").
        viewBinding.clearPredictionButton.setOnClickListener {
            if (isWriting) {
                isWriting = false
                viewBinding.startWritingButton.text = "Start Writing"
                viewBinding.startWritingButton.backgroundTintList =
                    ContextCompat.getColorStateList(this, R.color.green)
            }
            if (isRecording) stopProcessingAndRecording()
            sessionGeneration++ // Discard any pending result from before Clear.
            // Pending results are rejected by the session generation above.
            viewBinding.predictedLetterTextView.text = "No Prediction Yet"
            accumulatedCoordinates.clear()
            trackingCoordinates = ""
        }

        loadModels()
        cameraHelper.setupZoomControls()
    }

    // Function for toggling writing mode.
    private fun toggleWritingMode() {
        if (isRecording || isFinalizing) {
            Toast.makeText(this, "Stop tracking before changing writing mode.", Toast.LENGTH_SHORT).show()
            return
        }
        if (!isWriting) {
            // Begin a new writing session.
            viewBinding.predictedLetterTextView.text = ""
            accumulatedCoordinates.clear()
            trackingCoordinates = ""

            isWriting = true
            viewBinding.startWritingButton.text = "Stop Writing"
            viewBinding.startWritingButton.backgroundTintList =
                ContextCompat.getColorStateList(this, R.color.red)
        } else {
            isWriting = false
            viewBinding.startWritingButton.text = "Start Writing"
            viewBinding.startWritingButton.backgroundTintList =
                ContextCompat.getColorStateList(this, R.color.green)
            val prediction = viewBinding.predictedLetterTextView.text.toString()
            if (prediction.isBlank() || prediction == "No Prediction Yet") return
            if (prediction.matches(Regex("^(?=.*[A-Za-z])(?=.*\\d).+$"))) {
                AlertDialog.Builder(this)
                    .setTitle("Send Email")
                    .setMessage("Do you wish to send an email with the text: $prediction?")
                    .setPositiveButton("Yes") { _, _ -> sendEmail(prediction) }
                    .setNegativeButton("No") { _, _ -> launch3DActivity() }
                    .show()
            } else if (isLetterSelected) {
                AlertDialog.Builder(this)
                    .setTitle("Send Email")
                    .setMessage("Do you wish to send an email with the text: $prediction?")
                    .setPositiveButton("Yes") { _, _ -> sendEmail(prediction) }
                    .setNegativeButton("No") { _, _ -> launch3DActivity() }
                    .show()
            } else {
                if (prediction.matches(Regex("\\d+"))) {
                    AlertDialog.Builder(this)
                        .setTitle("Call Number")
                        .setMessage("Do you wish to call the number $prediction?")
                        .setPositiveButton("Yes") { _, _ -> makePhoneCall(prediction) }
                        .setNegativeButton("No") { _, _ -> launch3DActivity() }
                        .show()
                } else {
                    launch3DActivity()
                }
            }
        }
    }

    // New function to send an email with the prediction.
    private fun sendEmail(text: String) {
        val emailIntent = Intent(Intent.ACTION_SENDTO).apply {
            data = Uri.parse("mailto:")
            putExtra(Intent.EXTRA_SUBJECT, "Air-Written Email by Xamera")
            putExtra(Intent.EXTRA_TEXT, text)
        }
        launchExternalActivity(emailIntent)
    }

    private fun startProcessingAndRecording() {
        if (isFinalizing) return
        if (!allPermissionsGranted()) {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
            return
        }
        if (cameraHelper.cameraCaptureSession == null || !modelsReady) {
            Toast.makeText(this, "Camera and recognition models are still loading.", Toast.LENGTH_SHORT).show()
            return
        }
        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P &&
            (Settings.ExportData.videoDATA || Settings.ExportData.frameIMG) &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            requestPermissionLauncher.launch(arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE))
            Toast.makeText(this, "Allow storage access, then start tracking again.", Toast.LENGTH_SHORT).show()
            return
        }
        sessionGeneration++
        isRecording = true
        isProcessing = true
        viewBinding.startProcessingButton.text = "Stop Tracking"
        viewBinding.startProcessingButton.backgroundTintList = ContextCompat.getColorStateList(this, R.color.red)
        viewBinding.processedFrameView.visibility = View.VISIBLE
        val saveVideo = Settings.ExportData.videoDATA
        processingExecutor.execute {
            videoProcessor?.reset()
            if (saveVideo) {
                try {
                    val (width, height) = videoProcessor!!.getModelDimensions()
                    val file = File.createTempFile("XameraVideo_", ".mp4", cacheDir)
                    recordingFile = file
                    processedVideoRecorder = ProcessedVideoRecorder(width, height, file.path).also { it.start() }
                } catch (error: Exception) {
                    discardRecording()
                    reportError("Could not start video saving", error)
                }
            }
        }
    }

    private fun stopProcessingAndRecording() {
        if (!isRecording) return
        val generation = sessionGeneration
        val letterMode = isLetterSelected
        val writingMode = isWriting
        val saveFrame = Settings.ExportData.frameIMG
        isRecording = false
        isProcessing = false
        isFinalizing = true
        viewBinding.startProcessingButton.isEnabled = false
        viewBinding.startProcessingButton.text = "Finishing…"
        viewBinding.startProcessingButton.backgroundTintList = ContextCompat.getColorStateList(this, R.color.blue)
        viewBinding.processedFrameView.visibility = View.GONE
        viewBinding.processedFrameView.setImageBitmap(null)

        // Queued after the last frame, so export and inference see a complete, stable trace.
        processingExecutor.execute {
            var prediction = ""
            var coordinates = ""
            try {
                finishRecording()
                val processor = videoProcessor!!
                if (processor.hasTrace()) {
                    coordinates = processor.getTrackingCoordinatesString()
                    if (saveFrame) {
                        val file = File.createTempFile("DrawnLine_28x28_", ".png", cacheDir)
                        val bitmap = processor.exportTraceForInference()
                        try {
                            check(ProcessedFrameRecorder(file.path).save(bitmap)) { "Could not encode trace image" }
                            MediaExporter.publish(this, file, false)
                        } catch (error: Exception) {
                            reportError("Could not save trace image", error)
                        } finally {
                            bitmap.recycle()
                            file.delete()
                        }
                    }
                    prediction = if (letterMode) runLetterRecognitionInference() else runDigitRecognitionInference()
                }
            } catch (error: Exception) {
                reportError("Could not recognize this trace", error)
            } finally {
                runOnUiThread {
                    isFinalizing = false
                    if (!isDestroyed) {
                        viewBinding.startProcessingButton.isEnabled = true
                        viewBinding.startProcessingButton.text = "Start Tracking"
                        if (generation == sessionGeneration) {
                            trackingCoordinates = coordinates
                            if (prediction.isNotEmpty()) {
                                if (writingMode) {
                                    accumulateCoordinates(coordinates)
                                    val previous = viewBinding.predictedLetterTextView.text.toString()
                                    viewBinding.predictedLetterTextView.text =
                                        (if (previous == "No Prediction Yet") "" else previous) + prediction
                                } else viewBinding.predictedLetterTextView.text = prediction
                            } else if (!writingMode) {
                                viewBinding.predictedLetterTextView.text = "No Prediction Yet"
                            }
                        }
                    }
                }
            }
        }
    }

    private fun finishRecording() {
        val recorder = processedVideoRecorder
        val file = recordingFile
        processedVideoRecorder = null
        recordingFile = null
        try {
            if (recorder?.stop() == true && file != null) MediaExporter.publish(this, file, true)
        } catch (error: Exception) {
            recorder?.cancel()
            reportError("Could not save video", error)
        } finally {
            file?.delete()
        }
    }

    private fun discardRecording() {
        processedVideoRecorder?.cancel()
        processedVideoRecorder = null
        recordingFile?.delete()
        recordingFile = null
    }

    private fun reportError(message: String, error: Exception) {
        Log.e("MainActivity", message, error)
        runOnUiThread {
            if (!isDestroyed) Toast.makeText(this, message, Toast.LENGTH_LONG).show()
        }
    }

    private fun launchExternalActivity(intent: Intent) {
        try {
            startActivity(intent)
            shouldClearPrediction = true
        } catch (error: ActivityNotFoundException) {
            reportError("No installed app can handle this action", error)
        }
    }

    // NEW: Function to accumulate and horizontally offset new tracking coordinates.
    private fun accumulateCoordinates(newCoords: String) {
        if (newCoords.isEmpty()) return
        if (accumulatedCoordinates.isEmpty()) {
            accumulatedCoordinates.add(newCoords)
        } else {
            var offsetX = 0.0
            for (coordStr in accumulatedCoordinates) {
                val pts = coordStr.split(";").mapNotNull {
                    val parts = it.split(",")
                    parts.getOrNull(0)?.toDoubleOrNull()
                }
                if (pts.isNotEmpty()) {
                    val currentMax = pts.maxOrNull() ?: 0.0
                    offsetX = max(offsetX, currentMax)
                }
            }
            offsetX += 10.0
            val adjustedPoints = newCoords.split(";").mapNotNull { pointStr ->
                val parts = pointStr.split(",")
                if (parts.size >= 2) {
                    val x = parts[0].toDoubleOrNull() ?: 0.0
                    val y = parts[1]
                    val z = if (parts.size >= 3) parts[2] else "0.0"
                    "${(x + offsetX)},$y,$z"
                } else null
            }
            val adjustedCoords = adjustedPoints.joinToString(separator = ";")
            accumulatedCoordinates.add(adjustedCoords)
        }
    }

    private fun runDigitRecognitionInference(): String {
        val digitBitmap = videoProcessor?.exportTraceForInference()
        if (digitBitmap == null) {
            Log.e("MainActivity", "No digit image available for inference")
            return ""
        }
        val grayBitmap = convertToGrayscale(digitBitmap)
        val inputBuffer = convertBitmapToGrayscaleByteBuffer(grayBitmap)
        grayBitmap.recycle()
        digitBitmap.recycle()
        val outputArray = Array(1) { FloatArray(10) }
        if (tfliteInterpreter == null) {
            Log.e("MainActivity", "Digit model interpreter not set")
            return ""
        }
        tfliteInterpreter?.run(inputBuffer, outputArray)
        val predictedDigit = outputArray[0].indices.maxByOrNull { outputArray[0][it] } ?: -1
        Log.d("MainActivity", "Digit model predicted: $predictedDigit")
        return predictedDigit.toString()
    }

    private fun runLetterRecognitionInference(): String {
        val letterBitmap = videoProcessor?.exportTraceForInference()
        if (letterBitmap == null) {
            Log.e("MainActivity", "No letter image available for inference")
            return ""
        }
        val grayBitmap = convertToGrayscale(letterBitmap)
        val inputBuffer = convertBitmapToGrayscaleByteBuffer(grayBitmap)
        grayBitmap.recycle()
        letterBitmap.recycle()
        val outputArray = Array(1) { FloatArray(26) }
        if (letterInterpreter == null) {
            Log.e("MainActivity", "Letter model interpreter not set")
            return ""
        }
        letterInterpreter?.run(inputBuffer, outputArray)
        val maxIndex = outputArray[0].indices.maxByOrNull { outputArray[0][it] } ?: -1
        if (maxIndex == -1) {
            return ""
        }
        val predictedLetter = ('A'.code + maxIndex).toChar()
        Log.d("MainActivity", "Letter model predicted: $predictedLetter")
        return predictedLetter.toString()
    }

    private fun convertToGrayscale(bitmap: Bitmap): Bitmap {
        val grayscaleBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(grayscaleBitmap)
        val paint = Paint()
        val colorMatrix = ColorMatrix().apply { setSaturation(0f) }
        val filter = ColorMatrixColorFilter(colorMatrix)
        paint.colorFilter = filter
        canvas.drawBitmap(bitmap, 0f, 0f, paint)
        return grayscaleBitmap
    }

    private fun convertBitmapToGrayscaleByteBuffer(bitmap: Bitmap): ByteBuffer {
        val inputSize = bitmap.width * bitmap.height
        val byteBuffer = ByteBuffer.allocateDirect(inputSize * 4)
        byteBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        for (pixel in intValues) {
            val r = (pixel shr 16 and 0xFF).toFloat()
            // Both classifiers were trained with Normalize(mean=0.5, std=0.5).
            val normalized = r / 127.5f - 1.0f
            byteBuffer.putFloat(normalized)
        }
        byteBuffer.rewind()
        return byteBuffer
    }

    // Modified launch3DActivity(): Combine accumulated coordinates.
    private fun launch3DActivity() {
        val coords = if (accumulatedCoordinates.isNotEmpty())
            accumulatedCoordinates.joinToString(separator = "|")
        else if (trackingCoordinates.isNotEmpty())
            trackingCoordinates
        else
            "0.0,0.0,0.0;5.0,10.0,-5.0;-5.0,15.0,10.0;20.0,-5.0,5.0;-10.0,0.0,-10.0;10.0,-15.0,15.0;0.0,20.0,-5.0"
        val intent = Intent(this, com.xamera.ar.core.components.java.sharedcamera.SharedCameraActivity::class.java)
        intent.putExtra("LETTER_KEY", viewBinding.predictedLetterTextView.text.toString())
        intent.putExtra("PATH_COORDINATES", coords)
        shouldClearPrediction = true
        startActivity(intent)
    }

    // Simulate making a phone call using ACTION_DIAL.
    private fun makePhoneCall(digits: String) {
        val callIntent = Intent(Intent.ACTION_DIAL)
        callIntent.data = Uri.parse("tel:$digits")
        launchExternalActivity(callIntent)
    }

    private fun processFrameWithVideoProcessor() {
        if (isProcessingFrame || !isProcessing) return
        val bitmap = viewBinding.viewFinder.bitmap ?: return
        val generation = sessionGeneration
        isProcessingFrame = true
        processingExecutor.execute {
            val frames = videoProcessor?.processFrame(bitmap)
            bitmap.recycle()
            try {
                frames?.let { (_, recordingBitmap) -> processedVideoRecorder?.recordFrame(recordingBitmap) }
            } catch (error: Exception) {
                discardRecording()
                reportError("Video saving stopped", error)
            }
            runOnUiThread {
                isProcessingFrame = false
                frames?.let { (output, recordingBitmap) ->
                    if (generation == sessionGeneration && isProcessing && !isDestroyed) {
                        viewBinding.processedFrameView.setImageBitmap(output)
                    } else output.recycle()
                    if (recordingBitmap !== output) recordingBitmap.recycle()
                }
            }
        }
    }

    private fun loadModels() {
        processingExecutor.execute {
            try {
                // CPU interpreters are created, used and closed on this worker. GPU delegates
                // require thread affinity and can fail during interpreter creation on some phones.
                fun load(name: String): Interpreter {
                    val bytes = assets.open(name).use { it.readBytes() }
                    val buffer = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
                    buffer.put(bytes).rewind()
                    return Interpreter(buffer, Interpreter.Options().setNumThreads(
                        Runtime.getRuntime().availableProcessors().coerceIn(1, 4)))
                }
                videoProcessor?.setInterpreter(load("YOLOv3_float32.tflite"))
                tfliteInterpreter = load("DigitRecog_float32.tflite")
                letterInterpreter = load("LetterRecog_float32.tflite")
                runOnUiThread { if (!isDestroyed) modelsReady = true }
            } catch (error: Exception) {
                reportError("Could not load recognition models", error)
            }
        }
    }

    private var isFrontCamera = false
    private fun switchCamera() {
        if (isRecording) {
            stopProcessingAndRecording()
        }
        isFrontCamera = !isFrontCamera
        cameraHelper.isFrontCamera = isFrontCamera
        cameraHelper.closeCamera()
        if (allPermissionsGranted() && isResumed) cameraHelper.openCamera()
    }

    override fun onResume() {
        super.onResume()
        isResumed = true
        Settings.load(sharedPreferences)
        cameraHelper.startBackgroundThread()
        if (viewBinding.viewFinder.isAvailable && allPermissionsGranted()) cameraHelper.openCamera()
        if (shouldClearPrediction) {
            sessionGeneration++
            accumulatedCoordinates.clear()
            trackingCoordinates = ""
            viewBinding.predictedLetterTextView.text = "No Prediction Yet"
            shouldClearPrediction = false
        }
    }

    override fun onPause() {
        isResumed = false
        if (isRecording) stopProcessingAndRecording()
        cameraHelper.closeCamera()
        cameraHelper.stopBackgroundThread()
        super.onPause()
    }

    override fun onDestroy() {
        sessionGeneration++
        processingExecutor.execute {
            discardRecording()
            videoProcessor?.close()
            tfliteInterpreter?.close()
            letterInterpreter?.close()
        }
        processingExecutor.shutdown()
        super.onDestroy()
    }

    private fun allPermissionsGranted(): Boolean {
        return REQUIRED_PERMISSIONS.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }
    }
}