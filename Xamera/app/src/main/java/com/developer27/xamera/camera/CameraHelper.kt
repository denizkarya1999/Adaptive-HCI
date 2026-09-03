package com.developer27.xamera.camera

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.SharedPreferences
import android.graphics.Rect
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.media.MediaRecorder
import android.os.Handler
import android.os.HandlerThread
import android.util.Size
import android.view.MotionEvent
import android.view.Surface
import android.widget.Toast
import com.developer27.xamera.MainActivity
import com.developer27.xamera.databinding.ActivityMainBinding
import android.content.pm.PackageManager
import androidx.core.content.ContextCompat

/**
 * CameraHelper is responsible for:
 *  - Opening & closing the camera
 *  - Switching front/back
 *  - Creating a preview
 *  - Handling zoom & shutter speed
 *  - Starting a background thread for camera operations
 *
 *  This version forces a specific AWB mode & color correction to avoid color tint on Pixel 4a.
 */
class CameraHelper(
    private val activity: MainActivity,
    private val viewBinding: ActivityMainBinding,
    private val sharedPreferences: SharedPreferences
) {
    // The Android Camera2 API
    val cameraManager: CameraManager by lazy {
        activity.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    }

    // Active camera device + capture session
    var cameraDevice: CameraDevice? = null
    var cameraCaptureSession: CameraCaptureSession? = null

    // Capture builder for preview (and record)
    var captureRequestBuilder: CaptureRequest.Builder? = null

    // Preview + video sizes
    var previewSize: Size? = null
    var videoSize: Size? = null

    // Sensor area for zoom
    var sensorArraySize: Rect? = null

    // Whether we are using the front camera
    var isFrontCamera = false

    // Thread for camera operations
    private var backgroundThread: HandlerThread? = null
    var backgroundHandler: Handler? = null
        private set

    // Zoom control
    private var zoomLevel = 1.0f
    private var maxZoom = 1.0f
    private var cameraGeneration = 0
    private var isOpening = false
    private var isActive = false
    private var previewSurface: Surface? = null
    private val mainHandler = Handler(activity.mainLooper)
    private val zoomHandler = Handler(activity.mainLooper)

    // ------------------------------------------------------------------------
    // Background Thread Setup
    // ------------------------------------------------------------------------
    fun startBackgroundThread() {
        isActive = true
        if (backgroundThread != null) return
        backgroundThread = HandlerThread("CameraBackground").also { it.start() }
        backgroundHandler = Handler(backgroundThread!!.looper)
    }

    fun stopBackgroundThread() {
        isActive = false
        backgroundThread?.quitSafely()
        try {
            backgroundThread?.join()
            backgroundThread = null
            backgroundHandler = null
        } catch (e: InterruptedException) {
            e.printStackTrace()
        }
    }

    // ------------------------------------------------------------------------
    // Open/Close Camera
    // ------------------------------------------------------------------------
    @SuppressLint("MissingPermission")
    fun openCamera() {
        if (!isActive || isOpening || cameraDevice != null || !viewBinding.viewFinder.isAvailable) return
        if (ContextCompat.checkSelfPermission(activity, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return
        val generation = ++cameraGeneration
        try {
            // Decide which camera (front/back)
            val cameraId = getCameraId()
            val characteristics = cameraManager.getCameraCharacteristics(cameraId)

            // Grab the full sensor area for zoom
            sensorArraySize = characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
            maxZoom = (characteristics.get(CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM) ?: 1f).coerceAtLeast(1f)
            zoomLevel = zoomLevel.coerceIn(1f, maxZoom)

            // Possible output sizes
            val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                ?: return

            // Choose your preview/video sizes
            previewSize = chooseOptimalSize(map.getOutputSizes(SurfaceTexture::class.java))
            videoSize = chooseOptimalSize(map.getOutputSizes(MediaRecorder::class.java))

            // Now open the selected camera
            isOpening = true
            cameraManager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    if (generation != cameraGeneration || !isActive) {
                        camera.close()
                        return
                    }
                    isOpening = false
                    cameraDevice = camera
                    createCameraPreview()
                }
                override fun onDisconnected(camera: CameraDevice) {
                    camera.close()
                    if (generation == cameraGeneration) closeCamera()
                }
                override fun onError(camera: CameraDevice, error: Int) {
                    onDisconnected(camera)
                    if (isActive) Toast.makeText(activity, "Camera unavailable. Try reopening Xamera.", Toast.LENGTH_LONG).show()
                }
            }, mainHandler)
        } catch (e: CameraAccessException) {
            isOpening = false
            e.printStackTrace()
        } catch (e: SecurityException) {
            isOpening = false
            e.printStackTrace()
            Toast.makeText(activity, "Camera permission needed.", Toast.LENGTH_SHORT).show()
        }
    }

    fun closeCamera() {
        cameraGeneration++
        isOpening = false
        zoomHandler.removeCallbacksAndMessages(null)
        cameraCaptureSession?.close()
        cameraCaptureSession = null
        cameraDevice?.close()
        cameraDevice = null
        captureRequestBuilder = null
        previewSurface?.release()
        previewSurface = null
    }

    // ------------------------------------------------------------------------
    // Create Preview
    // ------------------------------------------------------------------------
    fun createCameraPreview() {
        try {
            val texture = viewBinding.viewFinder.surfaceTexture ?: return
            // Match the texture view size to the chosen preview size
            previewSize?.let { texture.setDefaultBufferSize(it.width, it.height) }

            val device = cameraDevice ?: return
            val generation = cameraGeneration
            cameraCaptureSession?.close()
            cameraCaptureSession = null
            previewSurface?.release()
            val previewSurface = Surface(texture).also { this.previewSurface = it }
            // Build a preview request
            captureRequestBuilder = cameraDevice?.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
            // Add the preview surface as a target
            captureRequestBuilder?.addTarget(previewSurface)

            // Apply any manual or auto exposure logic
            applyRollingShutter()
            // Possibly set flash, lighting, zoom
            applyFlashIfEnabled()
            applyLightingMode()
            applyZoom()

            // ----------------------------------------------------------------
            // Force color correction to avoid greenish tint
            // 1) Auto White Balance (set to e.g. DAYLIGHT for consistent color)
            //    or CONTROL_AWB_MODE_AUTO for auto
            // 2) Color Correction Mode => HIGH_QUALITY for better color
            // ----------------------------------------------------------------
            captureRequestBuilder?.set(
                CaptureRequest.CONTROL_AWB_MODE,
                // For strictly "daylight" color:
                // CaptureRequest.CONTROL_AWB_MODE_DAYLIGHT
                // or if you prefer auto, do:
                CaptureRequest.CONTROL_AWB_MODE_AUTO
            )
            captureRequestBuilder?.set(
                CaptureRequest.COLOR_CORRECTION_MODE,
                CaptureRequest.COLOR_CORRECTION_MODE_HIGH_QUALITY
            )

            // Now create the capture session
            cameraDevice?.createCaptureSession(
                listOf(previewSurface),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        if (cameraDevice !== device || generation != cameraGeneration || !isActive) {
                            session.close()
                            return
                        }
                        // Save the session
                        cameraCaptureSession = session
                        updatePreview() // Start the preview
                    }

                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        Toast.makeText(
                            activity,
                            "Preview config failed.",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                },
                mainHandler
            )
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        } catch (e: IllegalStateException) {
            // A camera/session may close while a request is being submitted.
            e.printStackTrace()
        }
    }

    /**
     * Update the camera preview with latest builder settings
     */
    fun updatePreview() {
        if (cameraDevice == null || captureRequestBuilder == null) return
        try {
            // Keep forcing color correction and AWB
            captureRequestBuilder?.set(
                CaptureRequest.CONTROL_AWB_MODE,
                CaptureRequest.CONTROL_AWB_MODE_AUTO
            )
            captureRequestBuilder?.set(
                CaptureRequest.COLOR_CORRECTION_MODE,
                CaptureRequest.COLOR_CORRECTION_MODE_HIGH_QUALITY
            )

            cameraCaptureSession?.setRepeatingRequest(
                captureRequestBuilder!!.build(),
                null,
                backgroundHandler
            )
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        } catch (e: IllegalStateException) {
            // A camera/session may close while a request is being submitted.
            e.printStackTrace()
        }
    }

    // ------------------------------------------------------------------------
    // Camera Selection (Front/Back)
    // ------------------------------------------------------------------------
    fun getCameraId(): String {
        for (id in cameraManager.cameraIdList) {
            val facing = cameraManager
                .getCameraCharacteristics(id)
                .get(CameraCharacteristics.LENS_FACING)
            if (!isFrontCamera && facing == CameraCharacteristics.LENS_FACING_BACK) {
                return id
            } else if (isFrontCamera && facing == CameraCharacteristics.LENS_FACING_FRONT) {
                return id
            }
        }
        // fallback if none matched
        return cameraManager.cameraIdList.first()
    }

    private fun chooseOptimalSize(choices: Array<Size>): Size {
        val targetWidth = 1280
        val targetHeight = 720

        // Try to find 1280x720 specifically
        val found720p = choices.find { it.width == targetWidth && it.height == targetHeight }
        if (found720p != null) {
            return found720p
        }
        // fallback to the smallest
        return choices.minByOrNull { it.width * it.height } ?: choices[0]
    }

    // ------------------------------------------------------------------------
    // Rolling shutter & exposure
    // ------------------------------------------------------------------------
    fun applyRollingShutter() {
        // Decide if we can do manual or must do auto
        val cameraId = getCameraId()
        val characteristics = cameraManager.getCameraCharacteristics(cameraId)

        val capabilities = characteristics.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
        val canManualExposure = capabilities?.contains(
            CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR
        ) == true

        val shutterFps = sharedPreferences.getString("shutter_speed", "60")?.toIntOrNull() ?: 60
        val shutterValueNs = if (shutterFps > 0) 1_000_000_000L / shutterFps else 0L

        // If no manual or user set 0, just do auto
        if (!canManualExposure || shutterValueNs <= 0) {
            setAutoExposure()
            return
        }

        // If we can do manual, clamp to valid range
        val exposureTimeRange = characteristics.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
        val isoRange = characteristics.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)

        if (exposureTimeRange == null || isoRange == null) {
            // fallback to auto if no valid range
            setAutoExposure()
            return
        }

        val safeExposureNs = shutterValueNs.coerceIn(exposureTimeRange.lower, exposureTimeRange.upper)
        val safeISO = 100.coerceIn(isoRange.lower, isoRange.upper)

        // fully manual
        captureRequestBuilder?.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
        captureRequestBuilder?.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_OFF)
        captureRequestBuilder?.set(CaptureRequest.SENSOR_EXPOSURE_TIME, safeExposureNs)
        captureRequestBuilder?.set(CaptureRequest.SENSOR_SENSITIVITY, safeISO)
    }

    private fun setAutoExposure() {
        captureRequestBuilder?.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
        captureRequestBuilder?.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_ON)
    }

    /**
     * If user changes shutter speed in settings, we re-apply
     */
    fun updateShutterSpeed() {
        if (captureRequestBuilder == null) return
        applyRollingShutter()
        updatePreview()
    }

    // ------------------------------------------------------------------------
    // Flash & Lighting
    // ------------------------------------------------------------------------
    fun applyFlashIfEnabled() {
        val isFlashEnabled = sharedPreferences.getBoolean("enable_flash", false)
        captureRequestBuilder?.set(
            CaptureRequest.FLASH_MODE,
            if (isFlashEnabled) CaptureRequest.FLASH_MODE_TORCH
            else CaptureRequest.FLASH_MODE_OFF
        )
    }

    fun applyLightingMode() {
        // Only apply AE compensation if AE is ON
        val aeMode = captureRequestBuilder?.get(CaptureRequest.CONTROL_AE_MODE)
        if (aeMode == CameraMetadata.CONTROL_AE_MODE_ON) {
            val lightingMode = sharedPreferences.getString("lighting_mode", "normal")
            val cameraId = getCameraId()
            val compensationRange = cameraManager
                .getCameraCharacteristics(cameraId)
                .get(CameraCharacteristics.CONTROL_AE_COMPENSATION_RANGE)

            val exposureComp = when (lightingMode) {
                "low_light" -> compensationRange?.lower ?: 0
                "high_light" -> compensationRange?.upper ?: 0
                else -> 0
            }
            captureRequestBuilder?.set(
                CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION,
                exposureComp
            )
        }
    }

    // ------------------------------------------------------------------------
    // Zoom
    // ------------------------------------------------------------------------
    fun setupZoomControls() {
        var zoomInRunnable: Runnable? = null
        var zoomOutRunnable: Runnable? = null

        // Repetitive zoom in on long-press
        viewBinding.zoomInButton.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    zoomInRunnable = object : Runnable {
                        override fun run() {
                            zoomIn()
                            zoomHandler.postDelayed(this, 50)
                        }
                    }
                    zoomHandler.post(zoomInRunnable!!)
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    zoomInRunnable?.let { zoomHandler.removeCallbacks(it) }
                    true
                }
                else -> false
            }
        }

        // Repetitive zoom out on long-press
        viewBinding.zoomOutButton.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    zoomOutRunnable = object : Runnable {
                        override fun run() {
                            zoomOut()
                            zoomHandler.postDelayed(this, 50)
                        }
                    }
                    zoomHandler.post(zoomOutRunnable!!)
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    zoomOutRunnable?.let { zoomHandler.removeCallbacks(it) }
                    true
                }
                else -> false
            }
        }
    }

    private fun zoomIn() {
        if (zoomLevel < maxZoom) {
            zoomLevel = (zoomLevel + 0.1f).coerceAtMost(maxZoom)
            applyZoom()
        }
    }

    private fun zoomOut() {
        if (zoomLevel > 1.0f) {
            zoomLevel = (zoomLevel - 0.1f).coerceAtLeast(1f)
            applyZoom()
        }
    }

    /**
     * Applies digital zoom by setting the SCALER_CROP_REGION
     */
    fun applyZoom() {
        if (sensorArraySize == null || captureRequestBuilder == null) return
        val ratio = 1 / zoomLevel
        val croppedWidth = sensorArraySize!!.width() * ratio
        val croppedHeight = sensorArraySize!!.height() * ratio

        val left = sensorArraySize!!.left + ((sensorArraySize!!.width() - croppedWidth) / 2).toInt()
        val top = sensorArraySize!!.top + ((sensorArraySize!!.height() - croppedHeight) / 2).toInt()
        val right = (left + croppedWidth).toInt()
        val bottom = (top + croppedHeight).toInt()

        val zoomRect = Rect(left, top, right, bottom)
        captureRequestBuilder?.set(CaptureRequest.SCALER_CROP_REGION, zoomRect)

        try {
            cameraCaptureSession?.setRepeatingRequest(
                captureRequestBuilder!!.build(),
                null,
                backgroundHandler
            )
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        } catch (e: IllegalStateException) {
            // A camera/session may close while a request is being submitted.
            e.printStackTrace()
        }
    }
}