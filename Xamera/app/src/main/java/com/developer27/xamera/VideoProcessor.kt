package com.developer27.xamera

// TODO <Zaynab Mourtada>: Uncomment these once you have PyTorch Mobile in your build.gradle
// import org.pytorch.IValue
// import org.pytorch.Module
// import org.pytorch.Tensor
// import org.pytorch.torchvision.TensorImageUtils

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import android.widget.Toast
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import org.apache.commons.math3.analysis.interpolation.SplineInterpolator
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.video.KalmanFilter
import java.util.LinkedList

data class FrameData(
    val x: Double,
    val y: Double,
    val area: Double,
    val frameCount: Int
)

object Settings {
    object Contour {
        var threshold = 500
    }
    object Trace {
        var lineLimit = 50
        var splineStep = 0.01
        var originalLineColor = Scalar(255.0, 0.0, 0.0) // Red
        var splineLineColor = Scalar(0.0, 0.0, 255.0)  // Blue
        var lineThickness = 4
    }
    object Brightness {
        var factor = 2.0
        var threshold = 150.0
    }
    object Debug {
        var enableToasts = true
        var enableLogging = true
    }
}

class VideoProcessor(private val context: Context) {
    private lateinit var kalmanFilter: KalmanFilter

    // For line-drawing (visualization)
    private val rawDataList = LinkedList<Point>()
    private val smoothDataList = LinkedList<Point>()

    // For storing raw vs. filtered data across frames
    private val preFilter4Ddata = mutableListOf<FrameData>()
    private val postFilter4Ddata = mutableListOf<FrameData>()

    private var frameCount = 0

    init {
        initializeOpenCV()
    }

    private fun initializeOpenCV() {
        if (OpenCVLoader.initDebug()) {
            showToast("OpenCV loaded successfully")
            initializeKalmanFilter()
        }
    }

    private fun initializeKalmanFilter() {
        kalmanFilter = KalmanFilter(4, 2)
        kalmanFilter._transitionMatrix = Mat.eye(4, 4, CvType.CV_32F).apply {
            // For x,y velocity
            put(0, 2, 1.0)
            put(1, 3, 1.0)
        }
        kalmanFilter._measurementMatrix = Mat.eye(2, 4, CvType.CV_32F)
        kalmanFilter._processNoiseCov = Mat.eye(4, 4, CvType.CV_32F).apply {
            setTo(Scalar(1e-4))
        }
        kalmanFilter._measurementNoiseCov = Mat.eye(2, 2, CvType.CV_32F).apply {
            setTo(Scalar(1e-2))
        }
        kalmanFilter._errorCovPost = Mat.eye(4, 4, CvType.CV_32F)
    }

    /**
     * Clears tracking data at the start of a new tracking session.
     */
    fun clearTrackingData() {
        frameCount = 0
        preFilter4Ddata.clear()
        postFilter4Ddata.clear()
        rawDataList.clear()
        smoothDataList.clear()
        showToast("Tracking started: data reset.")
    }

    /**
     * Called *repeatedly* (each frame) in MainActivity if isTracking == true.
     * - Preprocessing
     * - Contour detection
     * - Kalman filtering
     * - Storing (x, y, frameCount)
     * - Drawing lines
     */
    suspend fun processFrame(bitmap: Bitmap): Bitmap? = withContext(Dispatchers.Default) {
        try {
            val mat = ImageUtils.bitmapToMat(bitmap)
            val cleanedMat = preprocessFrame(mat)
            mat.release()

            val (centerInfo, processedMat) = detectContourBlob(cleanedMat)
            cleanedMat.release()

            val (center, area) = centerInfo
            center?.let {
                val frameData = FrameData(it.x, it.y, area = area ?: 0.0, frameCount = frameCount++)

                rawDataList.add(it) //add raw data to rawDataList
                preFilter4Ddata.add(frameData)

                // Apply Kalman filter
                val (fx, fy) = applyKalmanFilter(it, area ?: 0.0)
                val smoothPoint = Point(fx, fy)
                smoothDataList.add(smoothPoint) //add smoothed data to smoothDataList
                postFilter4Ddata.add(FrameData(smoothPoint.x, smoothPoint.y, frameData.area, frameData.frameCount))

                listOf(rawDataList, smoothDataList).forEach { dataList ->
                    if (dataList.size > Settings.Trace.lineLimit) {
                        dataList.pollFirst() // Remove the first (oldest) point
                    }
                }

                // Draw lines
                TraceRenderer.drawRawTrace(rawDataList, processedMat)
                TraceRenderer.drawSplineCurve(smoothDataList, processedMat)
            }

            return@withContext ImageUtils.matToBitmap(processedMat).also {
                processedMat.release()
            }

        } catch (e: Exception) {
            Log.e("VideoProcessor", "Error processing frame: ${e.message}")
            e.printStackTrace()
            return@withContext null
        }
    }

    /**
     * Called once the user stops tracking. Waits a bit to gather all data,
     * then does final "predictions" from the collected data.
     *
     * E.g., We gather the (x, y, frameCount) points from postFilter4Ddata,
     * feed them to the letter & digit models, then return a combined result.
     */
    suspend fun predictFromCollectedData(): String? = withContext(Dispatchers.Default) {
        // 1) Wait a little while to ensure all frames are collected
        //    (Sometimes the last frames might still be processing).
        delay(1000)  // e.g., wait 1 second. Adjust as needed.

        // 2) If no data was collected, return null
        if (preFilter4Ddata.isEmpty()) {
            return@withContext null
        }

        // 3) For demonstration, produce dummy predictions based on the last frame index
        val lastFrameIndex = preFilter4Ddata.last().frameCount
        val letterPredicted = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[lastFrameIndex % 26]
        val digitPredicted = (lastFrameIndex % 10).toString()

        // Return combined string
        val result = "Letter: $letterPredicted\nDigit: $digitPredicted"
        return@withContext result
    }

    /**
     * Expose postFilter4Ddata so MainActivity can retrieve all (x, y, frameCount)
     * if user says "No" and we want to store them for future training.
     */
    fun getPostFilterData(): List<FrameData> {
        return postFilter4Ddata.toList() // return a copy
    }

    fun getPreFilterData(): List<FrameData> {
        return preFilter4Ddata.toList() // return a copy
    }

    private fun applyKalmanFilter(point: Point, area: Double): Pair<Double, Double> {
        val measurement = Mat(2, 1, CvType.CV_32F).apply {
            put(0, 0, point.x)
            put(1, 0, point.y)
        }
        kalmanFilter.predict()
        val corrected = kalmanFilter.correct(measurement)
        val fx = corrected[0,0][0]
        val fy = corrected[1,0][0]
        return fx to fy
    }

    private fun preprocessFrame(src: Mat): Mat {
        val grayMat = Preprocessing.applyGrayscale(src)
        val enhancedMat = Preprocessing.enhanceBrightness(grayMat)
        grayMat.release()

        val thresholdMat = Preprocessing.conditionalThresholding(enhancedMat)
        enhancedMat.release()

        val blurredMat = Preprocessing.applyGaussianBlur(thresholdMat)
        thresholdMat.release()

        val closedMat = Preprocessing.applyMorphologicalClosing(blurredMat)
        blurredMat.release()
        return closedMat
    }

    private fun detectContourBlob(image: Mat): Pair<Pair<Point?, Double?>, Mat> {
        val binaryImage = Mat()
        Imgproc.threshold(image, binaryImage, 200.0, 255.0, Imgproc.THRESH_BINARY)

        val contours = mutableListOf<MatOfPoint>()
        Imgproc.findContours(binaryImage, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        val largestContour = contours.maxByOrNull { Imgproc.contourArea(it) }
        val outputImage = Mat()
        Imgproc.cvtColor(image, outputImage, Imgproc.COLOR_GRAY2BGR)

        val areaThreshold = Settings.Contour.threshold.toDouble()
        val (centerPoint, area) = largestContour
            ?.takeIf { Imgproc.contourArea(it) > areaThreshold }
            ?.let {
                // Draw in pink
                Imgproc.drawContours(outputImage, listOf(it), -1, Scalar(255.0, 105.0, 180.0), Imgproc.FILLED)
                val areaVal = Imgproc.contourArea(it)
                val centerVal = calculateCenter(it, outputImage).first
                Pair(centerVal, areaVal)
            } ?: Pair(null, null)

        binaryImage.release()
        return Pair(Pair(centerPoint, area), outputImage)
    }

    private fun calculateCenter(contour: MatOfPoint, image: Mat): Pair<Point?, Pair<Int, Int>?> {
        val moments = Imgproc.moments(contour)
        if (moments.m00 == 0.0) return Pair(null, null)

        val cx = (moments.m10 / moments.m00).toInt()
        val cy = (moments.m01 / moments.m00).toInt()
        val centerPoint = Point(cx.toDouble(), cy.toDouble())

        // Draw a small red dot at the center
        Imgproc.circle(image, centerPoint, 10, Scalar(0.0, 0.0, 255.0), -1)
        return Pair(centerPoint, Pair(cx, cy))
    }

    private fun showToast(msg: String) {
        if (Settings.Debug.enableToasts) {
            Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
        }
    }

    private fun logDebug(tag: String = "MyAppDebug", msg: String) {
        if (Settings.Debug.enableLogging) {
            Log.d(tag, msg)
        }
    }
}

object TraceRenderer {
    fun drawRawTrace(data: List<Point>, image: Mat) {
        for (i in 1 until data.size) {
            Imgproc.line(
                image,
                data[i - 1],
                data[i],
                Settings.Trace.originalLineColor,
                Settings.Trace.lineThickness
            )
        }
    }

    fun drawSplineCurve(data: List<Point>, image: Mat) {
        val splinePair = SplineHelper.applySplineInterpolation(data) ?: return
        val (splineX, splineY) = splinePair

        var prevPoint: Point? = null
        var t = 0.0
        val maxT = (data.size - 1).toDouble()
        while (t <= maxT) {
            val currentPoint = Point(splineX.value(t), splineY.value(t))
            prevPoint?.let {
                Imgproc.line(
                    image,
                    it,
                    currentPoint,
                    Settings.Trace.splineLineColor,
                    Settings.Trace.lineThickness
                )
            }
            prevPoint = currentPoint
            t += Settings.Trace.splineStep
        }
    }
}

object SplineHelper {
    fun applySplineInterpolation(
        data: List<Point>
    ): Pair<org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction,
            org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction>? {
        if (data.size < 2) return null

        val interpolator = SplineInterpolator()
        val xData = data.map { it.x }.toDoubleArray()
        val yData = data.map { it.y }.toDoubleArray()
        val tData = data.indices.map { it.toDouble() }.toDoubleArray()

        val splineX = interpolator.interpolate(tData, xData)
        val splineY = interpolator.interpolate(tData, yData)
        return Pair(splineX, splineY)
    }
}

class ImageUtils {
    companion object {
        fun bitmapToMat(bitmap: Bitmap): Mat = Mat().also {
            Utils.bitmapToMat(bitmap, it)
        }
        fun matToBitmap(mat: Mat): Bitmap = Bitmap.createBitmap(
            mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888
        ).apply {
            Utils.matToBitmap(mat, this)
        }
    }
}

class Preprocessing {
    companion object {
        fun applyGrayscale(frame: Mat): Mat {
            val grayMat = Mat()
            Imgproc.cvtColor(frame, grayMat, Imgproc.COLOR_BGR2GRAY)
            return grayMat
        }
        fun enhanceBrightness(image: Mat): Mat = Mat().apply {
            Core.multiply(image, Scalar(Settings.Brightness.factor), this)
        }
        fun conditionalThresholding(image: Mat): Mat {
            val thresholdMat = Mat()
            Imgproc.threshold(
                image,
                thresholdMat,
                Settings.Brightness.threshold,
                255.0,
                Imgproc.THRESH_TOZERO
            )
            return thresholdMat
        }
        fun applyGaussianBlur(image: Mat): Mat {
            val blurredMat = Mat()
            Imgproc.GaussianBlur(image, blurredMat, Size(5.0, 5.0), 0.0)
            return blurredMat
        }
        fun applyMorphologicalClosing(image: Mat): Mat {
            val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
            val closedImage = Mat()
            Imgproc.morphologyEx(image, closedImage, Imgproc.MORPH_CLOSE, kernel)
            return closedImage
        }
    }
}
