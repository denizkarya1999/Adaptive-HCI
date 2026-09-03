@file:Suppress("SameParameterValue")

package com.developer27.xamera.videoprocessing

import android.content.SharedPreferences
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import org.apache.commons.math3.analysis.interpolation.SplineInterpolator
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.video.KalmanFilter
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.common.ops.NormalizeOp
import java.util.LinkedList
import kotlin.math.max
import kotlin.math.min

data class DetectionResult(
    val xCenter: Float, val yCenter: Float,
    val width: Float, val height: Float,
    val confidence: Float
)
data class BoundingBox(
    val x1: Float, val y1: Float,
    val x2: Float, val y2: Float,
    val confidence: Float, val classId: Int
)


// Object to hold various configuration settings.
object Settings {
    fun load(preferences: SharedPreferences) {
        DetectionMode.current = if (preferences.getString("detection_mode", "CONTOUR") == "YOLO")
            DetectionMode.Mode.YOLO else DetectionMode.Mode.CONTOUR
        DetectionMode.enableYOLOinference = DetectionMode.current == DetectionMode.Mode.YOLO
        BoundingBox.enableBoundingBox = preferences.getBoolean("enable_bounding_box", true)
        Trace.enableRAWtrace = preferences.getBoolean("enable_raw_trace", false)
        Trace.enableSPLINEtrace = preferences.getBoolean("enable_spline_trace", true)
        ExportData.frameIMG = preferences.getBoolean("frame_img", false)
        ExportData.videoDATA = preferences.getBoolean("video_data", false)
    }

    object DetectionMode {
        enum class Mode { CONTOUR, YOLO }
        var current: Mode = Mode.CONTOUR // YOLO: MAIN MODE for Demo, CONTOUR: For Testing & For 28x28 IMG
        var enableYOLOinference = false  // Only use with YOLO enabled
    }
    object Inference {
        var confidenceThreshold: Float = 0.5f
        var iouThreshold: Float = 0.5f
    }
    object Trace {
        var enableRAWtrace = false     // RAW collected connected line (has harsh angles)
        var enableSPLINEtrace = true   // SMOOTHED collected connected line (spline, transposed from RAW line)
        var lineLimit = 75             // Line Length
        var splineStep = 0.01          // Granularity of the spline line (smoothed line)
        var originalLineColor = Scalar(0.0, 39.0, 76.0)
        var splineLineColor = Scalar(255.0, 203.0, 5.0)
        var lineThickness = 4
    }
    object BoundingBox {
        var enableBoundingBox = true
        var boxColor = Scalar(0.0, 39.0, 76.0)
        var boxThickness = 2
    }
    object Brightness {
        var factor = 2.0
        var threshold = 10.0
    }
    object ExportData {
        var frameIMG = false          // enable or disable 28x28 IMG saving
        var videoDATA = false        // enable or disable video saving (for YOLO training)
    }
}

// Main VideoProcessor class.
class VideoProcessor {
    private var tfliteInterpreter: Interpreter? = null
    private val rawDataList = LinkedList<Point>()
    private val smoothDataList = LinkedList<Point>()
    private val kalmanHelper = KalmanHelper()
    private val traceRenderer = TraceRenderer(rawDataList, smoothDataList, kalmanHelper)
    private val inputProcessor = ImageProcessor.Builder().add(NormalizeOp(0f, 255f)).build()


    init {
        initOpenCV()
        kalmanHelper.initKalmanFilter()
    }
    private fun initOpenCV() {
        try {
            System.loadLibrary("opencv_java4")
        } catch (e: UnsatisfiedLinkError) {
            Log.d("VideoProcessor","OpenCV failed to load: ${e.message}", e)
        }
    }
    fun setInterpreter(model: Interpreter) {
        tfliteInterpreter?.close()
        tfliteInterpreter = model
        Log.d("VideoProcessor","TFLite Model set in VideoProcessor successfully!")
    }
    fun reset() {
        rawDataList.clear()
        smoothDataList.clear()
        kalmanHelper.reset()
    }

    fun close() {
        tfliteInterpreter?.close()
        tfliteInterpreter = null
        reset()
    }

    fun hasTrace(): Boolean = smoothDataList.size >= 3

    // All processor operations run on the activity's single processing executor.
    fun processFrame(bitmap: Bitmap): Pair<Bitmap, Bitmap>? = try {
        when (Settings.DetectionMode.current) {
            Settings.DetectionMode.Mode.CONTOUR -> processFrameInternalCONTOUR(bitmap)
            Settings.DetectionMode.Mode.YOLO -> processFrameInternalYOLO(bitmap)
        }
    } catch (error: Exception) {
        Log.e("VideoProcessor", "Error processing frame", error)
        null
    }

    // Processes a frame using Contour Detection - Returns a Pair containing outputBitmap and videoBitmap.
    private fun processFrameInternalCONTOUR(bitmap: Bitmap): Pair<Bitmap, Bitmap>? {
        return try {
            val sMat = Mat().also { Utils.bitmapToMat(bitmap, it) }
            val pMat = Preprocessing.preprocessFrame(bitmap)
            val rois: MutableList<Rect> = ContourDetection.processContourDetection(pMat)
            if (Settings.BoundingBox.enableBoundingBox) {
                for (roi in rois) { Imgproc.rectangle(sMat, roi.tl(), roi.br(), Scalar(255.0, 0.0, 0.0), 3) }
            }
            // Find the largest ROI by area (width * height).
            val largestROI = rois.maxByOrNull { it.width * it.height }
            // Compute the center of the largest ROI.
            val cent: Point? = largestROI?.let {
                Point(it.x + it.width / 2.0, it.y + it.height / 2.0)
            }
            traceRenderer.drawTrace(cent, sMat)

            val outBmp = Bitmap.createBitmap(sMat.cols(), sMat.rows(), Bitmap.Config.ARGB_8888).also { Utils.matToBitmap(sMat, it) }
            pMat.release(); sMat.release()
            outBmp to outBmp
        } catch (e: Exception) {
            Log.d("VideoProcessor","Error processing frame: ${e.message}", e)
            null
        }
    }
    // Processes a frame using YOLO - Returns a Pair containing outputBitmap and letterboxedBitmap.
    private fun processFrameInternalYOLO(bitmap: Bitmap): Pair<Bitmap, Bitmap> {
        val (inputW, inputH, outputShape) = getModelDimensions()
        val (letterboxed, offsets) = YOLOHelper.createLetterboxedBitmap(bitmap, inputW, inputH)
        val m = Mat().also { Utils.bitmapToMat(bitmap, it) }
        if (Settings.DetectionMode.enableYOLOinference && tfliteInterpreter != null) {
            val out = Array(outputShape[0]) { Array(outputShape[1]) { FloatArray(outputShape[2]) } }
            val input = inputProcessor.process(TensorImage(DataType.FLOAT32).apply { load(letterboxed) })
            tfliteInterpreter?.run(input.buffer, out)
            YOLOHelper.parseTFLite(out)?.let {
                val (box, c) = YOLOHelper.rescaleInferencedCoordinates(it, bitmap.width, bitmap.height, offsets, inputW, inputH)
                if (Settings.BoundingBox.enableBoundingBox) YOLOHelper.drawBoundingBoxes(m, box)
                traceRenderer.drawTrace(c, m)
            }
        }
        val yoloBmp = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888).also {
            Utils.matToBitmap(m, it)
            m.release()
        }
        return yoloBmp to letterboxed
    }
    // Dynamically retrieves the model input size.
    fun getModelDimensions(): Triple<Int, Int, List<Int>> {
        val inTensor = tfliteInterpreter?.getInputTensor(0)
        val inShape = inTensor?.shape()
        val (h, w) = (inShape?.getOrNull(1) ?: 416) to (inShape?.getOrNull(2) ?: 416)
        val outTensor = tfliteInterpreter?.getOutputTensor(0)
        val outShape = outTensor?.shape()?.toList() ?: listOf(1, 5, 3549)
        return Triple(w, h, outShape)
    }
    // Creates a white, square (28x28) Bitmap that encapsulates the drawn spline trace (with padding).
    fun exportTraceForInference(): Bitmap {
        // Ensure there is some trace data.
        if (smoothDataList.isEmpty()) {
            return Bitmap.createBitmap(28, 28, Bitmap.Config.ARGB_8888).apply { eraseColor(Color.WHITE) }
        }
        // 1. Compute the bounding box of the trace points.
        var minX = Double.MAX_VALUE
        var minY = Double.MAX_VALUE
        var maxX = -Double.MAX_VALUE
        var maxY = -Double.MAX_VALUE
        for (pt in smoothDataList) {
            minX = min(minX, pt.x)
            minY = min(minY, pt.y)
            maxX = max(maxX, pt.x)
            maxY = max(maxY, pt.y)
        }
        // 2. Define padding (in pixels) around the output img.
        val padding = 30.0
        // Compute optimal dimensions.
        val (optimalWidth, optimalHeight) = (max((maxX - minX + 2 * padding).toInt(), 1)) to (max((maxY - minY + 2 * padding).toInt(), 1))
        // 3. Determine the square size as the greatest of the optimal dimensions.
        val squareSize = max(optimalWidth, optimalHeight)
        // 4. Create a white square Mat of the computed dimensions.
        val mat = Mat(squareSize, squareSize, CvType.CV_8UC4, Scalar(255.0, 255.0, 255.0, 255.0))
        // 5. Compute offsets to center the drawn trace inside the square.
        val (xOffset, yOffset) = ((squareSize - optimalWidth) / 2.0) to ((squareSize - optimalHeight) / 2.0)
        // 6. Create an adjusted list of points so that the drawing starts at (padding, padding) plus the offsets.
        val adjustedPoints = smoothDataList.map { Point(it.x - minX + padding + xOffset, it.y - minY + padding + yOffset) }
        // Render opaque black ink without changing the live preview's trace style.
        traceRenderer.drawSplineCurve(adjustedPoints, mat, Scalar(0.0, 0.0, 0.0, 255.0), 40)
        // 10. Convert the Mat back to a Bitmap.
        val outputBitmap = Bitmap.createBitmap(squareSize, squareSize, Bitmap.Config.ARGB_8888).apply {
            Utils.matToBitmap(mat, this)
            mat.release()
        }
        val scaledBitmap = Bitmap.createScaledBitmap(outputBitmap, 28, 28, true)
        if (scaledBitmap !== outputBitmap) outputBitmap.recycle()
        return scaledBitmap
    }
    fun exportTracesAsBitmap(): Bitmap {
        // 1) Combine data & handle empty
        val allPoints = rawDataList + smoothDataList
        if (allPoints.isEmpty()) {
            // Return a simple 500x500 white image if no data
            return Bitmap.createBitmap(500, 500, Bitmap.Config.ARGB_8888).apply {
                eraseColor(Color.WHITE)
            }
        }

        // 2) Compute bounding box for all points
        var minX = Double.MAX_VALUE
        var minY = Double.MAX_VALUE
        var maxX = -Double.MAX_VALUE
        var maxY = -Double.MAX_VALUE

        for (pt in allPoints) {
            minX = min(minX, pt.x)
            minY = min(minY, pt.y)
            maxX = max(maxX, pt.x)
            maxY = max(maxY, pt.y)
        }

        // 3) Define padding and scale factor
        val padding = 50.0
        val scaleFactor = 2.0  // Increase if you want an even bigger final image

        // Base width/height in unscaled coordinates
        val baseWidth = (maxX - minX + 2 * padding).coerceAtLeast(1.0)
        val baseHeight = (maxY - minY + 2 * padding).coerceAtLeast(1.0)

        // Scaled width/height
        val width = (baseWidth * scaleFactor).toInt()
        val height = (baseHeight * scaleFactor).toInt()

        // Create the Mat (background = white)
        val mat = Mat(height, width, CvType.CV_8UC3, Scalar(255.0, 255.0, 255.0))

        // Helper to convert a "data point" (in raw coords) to scaled Mat coords
        fun toScaledPoint(pt: Point): Point {
            val shiftedX = (pt.x - minX + padding) * scaleFactor
            val shiftedY = (pt.y - minY + padding) * scaleFactor
            return Point(shiftedX, shiftedY)
        }

        // 4) Draw the Raw (blue) trace
        if (rawDataList.size > 1) {
            for (i in 1 until rawDataList.size) {
                val p1 = toScaledPoint(rawDataList[i - 1])
                val p2 = toScaledPoint(rawDataList[i])
                Imgproc.line(
                    mat,
                    p1,
                    p2,
                    Settings.Trace.originalLineColor,  // e.g. Scalar(255,0,0)
                    Settings.Trace.lineThickness
                )
            }
        }

        // 5) Draw the Spline (maize) trace
        if (smoothDataList.size >= 3) {
            val (splineX, splineY) = traceRenderer.applySplineInterpolation(smoothDataList)
            var prevPoint: Point? = null
            var t = 0.0
            val maxT = (smoothDataList.size - 1).toDouble()

            while (t <= maxT) {
                val unscaledX = splineX.value(t) - minX + padding
                val unscaledY = splineY.value(t) - minY + padding
                val curPoint = Point(unscaledX * scaleFactor, unscaledY * scaleFactor)
                if (prevPoint != null) {
                    Imgproc.line(
                        mat,
                        prevPoint,
                        curPoint,
                        Settings.Trace.splineLineColor, // e.g. Scalar(0,255,255)
                        Settings.Trace.lineThickness
                    )
                }
                prevPoint = curPoint
                t += Settings.Trace.splineStep
            }
        }

        // 6) Measure path length of the Raw data & draw vertical reference line
        //    (only the length is used, no further measuring logic)
        val rawLength = measurePathLength(rawDataList)

        // Midpoint in the x-direction (unscaled)
        val midX = (minX + maxX) / 2.0

        // Shift & scale for the line's X location
        val midLineX = (midX - minX + padding) * scaleFactor

        // We'll place a vertical line with length = rawLength
        // and try to center it in the bounding box
        val startY = max(0.0, (baseHeight - rawLength) / 2.0) * scaleFactor
        val endY = startY + rawLength * scaleFactor

        val lineStart = Point(midLineX, startY)
        val lineEnd = Point(midLineX, endY)

        // Draw the vertical reference line (black)
        Imgproc.line(mat, lineStart, lineEnd, Scalar(0.0, 0.0, 0.0), 2)

        // 7) Convert Mat -> Bitmap and return
        val outBmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).apply {
            Utils.matToBitmap(mat, this)
        }
        mat.release()
        return outBmp
    }

    // Helper to measure total path length
    private fun measurePathLength(points: List<Point>): Double {
        var totalDist = 0.0
        for (i in 1 until points.size) {
            val dx = points[i].x - points[i - 1].x
            val dy = points[i].y - points[i - 1].y
            totalDist += kotlin.math.sqrt(dx * dx + dy * dy)
        }
        return totalDist
    }

    // Returns the tracking coordinates as a semicolon-separated string. Each point is formatted as "x,y,0.0".
    fun getTrackingCoordinatesString(): String {
        return smoothDataList.joinToString(separator = ";") { "${it.x},${it.y},0.0" }
    }
}

// Helper object to draw raw and spline traces.
class TraceRenderer(
    private val rawDataList: LinkedList<Point>,
    private val smoothDataList: LinkedList<Point>,
    private val kalmanHelper: KalmanHelper
) {
    fun drawTrace(center: Point?, contourMat: Mat) {
        center?.let { detectedCenter ->
            rawDataList.add(detectedCenter)
            val (fx, fy) = kalmanHelper.applyKalmanFilter(detectedCenter)
            smoothDataList.add(Point(fx, fy))
            if (rawDataList.size > Settings.Trace.lineLimit) rawDataList.pollFirst()
            if (smoothDataList.size > Settings.Trace.lineLimit) smoothDataList.pollFirst()
        }
        with(Settings.Trace) {
            if (enableRAWtrace) drawRawTrace(rawDataList, contourMat)
            if (enableSPLINEtrace) drawSplineCurve(smoothDataList, contourMat)
        }
    }
    private fun drawRawTrace(data: List<Point>, image: Mat) {
        for (i in 1 until data.size) {
            Imgproc.line(image, data[i - 1], data[i], Settings.Trace.originalLineColor, Settings.Trace.lineThickness)
        }
    }
    fun drawSplineCurve(
        data: List<Point>, image: Mat,
        color: Scalar = Settings.Trace.splineLineColor,
        thickness: Int = Settings.Trace.lineThickness
    ) {
        if (data.size < 3) return
        val splinePair = applySplineInterpolation(data)
        val (splineX, splineY) = splinePair
        var prevPoint: Point? = null
        var t = 0.0
        val maxT = (data.size - 1).toDouble()
        while (t <= maxT) {
            val currentPoint = Point(splineX.value(t), splineY.value(t))
            prevPoint?.let { Imgproc.line(image, it, currentPoint, color, thickness) }
            prevPoint = currentPoint
            t += Settings.Trace.splineStep
        }
    }
    internal fun applySplineInterpolation(data: List<Point>): Pair<PolynomialSplineFunction, PolynomialSplineFunction> {
        val interpolator = SplineInterpolator()
        val xData = data.map { it.x }.toDoubleArray()
        val yData = data.map { it.y }.toDoubleArray()
        val tData = data.indices.map { it.toDouble() }.toDoubleArray()
        val splineX = interpolator.interpolate(tData, xData)
        val splineY = interpolator.interpolate(tData, yData)
        return Pair(splineX, splineY)
    }
}

// Helper object for applying a Kalman filter to smooth tracking points.
class KalmanHelper {
    private var initialized = false
    fun reset() { initialized = false }
    private lateinit var kalmanFilter: KalmanFilter
    fun initKalmanFilter() {
        kalmanFilter = KalmanFilter(4, 2)
        kalmanFilter._transitionMatrix = Mat.eye(4, 4, CvType.CV_32F).apply {
            put(0, 2, 1.0)
            put(1, 3, 1.0)
        }
        kalmanFilter._measurementMatrix = Mat.eye(2, 4, CvType.CV_32F)
        kalmanFilter._processNoiseCov = Mat.eye(4, 4, CvType.CV_32F).apply { Core.multiply(this, Scalar(1e-4), this) }
        kalmanFilter._measurementNoiseCov = Mat.eye(2, 2, CvType.CV_32F).apply { Core.multiply(this, Scalar(1e-2), this) }
        kalmanFilter._errorCovPost = Mat.eye(4, 4, CvType.CV_32F)
    }
    fun applyKalmanFilter(point: Point): Pair<Double, Double> {
        if (!initialized) {
            val state = Mat.zeros(4, 1, CvType.CV_32F).apply {
                put(0, 0, point.x)
                put(1, 0, point.y)
            }
            kalmanFilter._statePost = state
            state.release()
            val covariance = Mat.eye(4, 4, CvType.CV_32F)
            kalmanFilter._errorCovPost = covariance
            covariance.release()
            initialized = true
            return point.x to point.y
        }
        val measurement = Mat(2, 1, CvType.CV_32F).apply {
            put(0, 0, point.x)
            put(1, 0, point.y)
        }
        kalmanFilter.predict().release()
        val corrected = kalmanFilter.correct(measurement)
        val fx = corrected[0, 0][0]
        val fy = corrected[1, 0][0]
        measurement.release()
        corrected.release()
        return fx to fy
    }
}

// Helper object for preprocessing frames with OpenCV.
object Preprocessing {
    fun preprocessFrame(src: Bitmap): Mat {
        val sMat = Mat().also { Utils.bitmapToMat(src, it) }
        val gMat = Mat().also { Imgproc.cvtColor(sMat, it, Imgproc.COLOR_RGBA2GRAY); sMat.release() }
        val eMat = Mat().also { Core.multiply(gMat, Scalar(Settings.Brightness.factor), it); gMat.release() }
        val tMat = Mat().also { Imgproc.threshold(eMat, it, Settings.Brightness.threshold, 255.0, Imgproc.THRESH_TOZERO); eMat.release() }
        val bMat = Mat().also { Imgproc.GaussianBlur(tMat, it, Size(5.0, 5.0), 0.0); tMat.release() }
        val k = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        val cMat = Mat().also { Imgproc.morphologyEx(bMat, it, Imgproc.MORPH_CLOSE, k); bMat.release() }
        k.release()
        return cMat
    }
}

// Helper object for contour detection.
object ContourDetection {
    // Main function: finds contours, converts them to bounding boxes, then merges close ones.
    fun processContourDetection(mat: Mat): MutableList<Rect> {
        // Get all contours (ROIs) with an area > 5000.
        val contours = findContours(mat)
        // Convert each contour into its bounding box.
        val rois = contours.map { contour -> Imgproc.boundingRect(contour).also { contour.release() } }.toMutableList()
        // Merge ROIs whose edges are within 10 pixels.
        return mergeCloseROIs(rois)
    }

    // Finds all contours in the image and returns only those with an area > 5000.
    private fun findContours(mat: Mat): MutableList<MatOfPoint> {
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(mat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        hierarchy.release()
        return contours.filter {
            val keep = Imgproc.contourArea(it) > 5000
            if (!keep) it.release()
            keep
        }.toMutableList()
    }

    // Merges ROIs (Rectangles) that are within 10 pixels of each other.
    private fun mergeCloseROIs(rois: MutableList<Rect>): MutableList<Rect> {
        val mergedROIs = rois.toMutableList()
        var merged = true
        // Keep iterating until no more merges occur.
        while (merged) {
            merged = false
            outer@ for (i in 0 until mergedROIs.size) {
                for (j in i + 1 until mergedROIs.size) {
                    val r1 = mergedROIs[i]
                    val r2 = mergedROIs[j]
                    if (areClose(r1, r2)) {
                        val unionRect = union(r1, r2)
                        // Remove the two ROIs and add the merged one.
                        mergedROIs.removeAt(j)
                        mergedROIs.removeAt(i)
                        mergedROIs.add(unionRect)
                        merged = true
                        break@outer
                    }
                }
            }
        }
        return mergedROIs
    }

    // Checks if two rectangles are close: if their expanded versions (by 10 pixels) intersect.
    private fun areClose(r1: Rect, r2: Rect): Boolean {
        // Expand each rectangle by 10 pixels in all directions.
        val expandedR1 = Rect(r1.x - 10, r1.y - 10, r1.width + 20, r1.height + 20)
        val expandedR2 = Rect(r2.x - 10, r2.y - 10, r2.width + 20, r2.height + 20)
        return expandedR1.x < expandedR2.x + expandedR2.width &&
                expandedR1.x + expandedR1.width > expandedR2.x &&
                expandedR1.y < expandedR2.y + expandedR2.height &&
                expandedR1.y + expandedR1.height > expandedR2.y
    }

    // Returns the union of two rectangles.
    private fun union(r1: Rect, r2: Rect): Rect {
        val x = Math.min(r1.x, r2.x)
        val y = Math.min(r1.y, r2.y)
        val x2 = Math.max(r1.x + r1.width, r2.x + r2.width)
        val y2 = Math.max(r1.y + r1.height, r2.y + r2.height)
        return Rect(x, y, x2 - x, y2 - y)
    }
}



// Helper object for YOLO detection using TensorFlow Lite.
object YOLOHelper {
    fun parseTFLite(rawOutput: Array<Array<FloatArray>>): DetectionResult? {
        val numDetections = rawOutput[0][0].size
        // Step 1: Parse detections and filter by confidence.
        val detections = mutableListOf<DetectionResult>()
        for (i in 0 until numDetections) {
            val xCenter = rawOutput[0][0][i]
            val yCenter = rawOutput[0][1][i]
            val width = rawOutput[0][2][i]
            val height = rawOutput[0][3][i]
            val confidence = rawOutput[0][4][i]
            if (confidence >= Settings.Inference.confidenceThreshold) {
                detections.add(DetectionResult(xCenter, yCenter, width, height, confidence))
            }
        }
        if (detections.isEmpty()) {
            Log.d("YOLOTest", "No detections above confidence threshold: ${Settings.Inference.confidenceThreshold}")
            return null
        }
        // Step 2: Convert detections to bounding boxes.
        val detectionBoxes = detections.map { it to detectionToBox(it) }.toMutableList()
        // Sort by confidence (highest first).
        detectionBoxes.sortByDescending { it.first.confidence }
        // Step 3: Apply NMS.
        val nmsDetections = mutableListOf<DetectionResult>()
        while (detectionBoxes.isNotEmpty()) {
            val current = detectionBoxes.removeAt(0)
            nmsDetections.add(current.first)
            detectionBoxes.removeAll { other ->
                computeIoU(current.second, other.second) > Settings.Inference.iouThreshold
            }
        }
        // Step 4: Choose the detection with the highest confidence from the remaining.
        val bestDetection = nmsDetections.maxByOrNull { it.confidence }
        bestDetection?.let { d ->
            Log.d(
                "YOLOTest",
                "BEST DETECTION: confidence=${"%.8f".format(d.confidence)}, x_center=${d.xCenter}, y_center=${d.yCenter}, width=${d.width}, height=${d.height}"
            )
        }
        return bestDetection
    }
    private fun detectionToBox(d: DetectionResult) = BoundingBox(
        d.xCenter - d.width / 2,
        d.yCenter - d.height / 2,
        d.xCenter + d.width / 2,
        d.yCenter + d.height / 2,
        d.confidence,
        1
    )
    private fun computeIoU(boxA: BoundingBox, boxB: BoundingBox): Float {
        val x1 = max(boxA.x1, boxB.x1)
        val y1 = max(boxA.y1, boxB.y1)
        val x2 = min(boxA.x2, boxB.x2)
        val y2 = min(boxA.y2, boxB.y2)
        val intersectionWidth = max(0f, x2 - x1)
        val intersectionHeight = max(0f, y2 - y1)
        val intersectionArea = intersectionWidth * intersectionHeight
        val areaA = (boxA.x2 - boxA.x1) * (boxA.y2 - boxA.y1)
        val areaB = (boxB.x2 - boxB.x1) * (boxB.y2 - boxB.y1)
        val unionArea = areaA + areaB - intersectionArea
        return if (unionArea > 0f) intersectionArea / unionArea else 0f
    }
    fun rescaleInferencedCoordinates(detection: DetectionResult, originalWidth: Int, originalHeight: Int, padOffsets: Pair<Int, Int>, modelInputWidth: Int, modelInputHeight: Int): Pair<BoundingBox, Point> {
        // Compute the scale factor used in the letterbox transformation.
        val scale = min(modelInputWidth / originalWidth.toDouble(), modelInputHeight / originalHeight.toDouble())
        // Get the padding applied during letterboxing.
        val padLeft = padOffsets.first.toDouble()
        val padTop = padOffsets.second.toDouble()
        // Convert normalized coordinates to letterboxed image coordinates.
        val xCenterLetterboxed = detection.xCenter * modelInputWidth
        val yCenterLetterboxed = detection.yCenter * modelInputHeight
        val boxWidthLetterboxed = detection.width * modelInputWidth
        val boxHeightLetterboxed = detection.height * modelInputHeight
        // Remove padding and rescale back to original image coordinates.
        val xCenterOriginal = (xCenterLetterboxed - padLeft) / scale
        val yCenterOriginal = (yCenterLetterboxed - padTop) / scale
        val boxWidthOriginal = boxWidthLetterboxed / scale
        val boxHeightOriginal = boxHeightLetterboxed / scale
        // Compute bounding box corners in original image coordinates.
        val x1Original = xCenterOriginal - (boxWidthOriginal / 2)
        val y1Original = yCenterOriginal - (boxHeightOriginal / 2)
        val x2Original = xCenterOriginal + (boxWidthOriginal / 2)
        val y2Original = yCenterOriginal + (boxHeightOriginal / 2)
        Log.d("YOLOTest", "Adjusted BOUNDING BOX: x1=${"%.8f".format(x1Original)}, y1=${"%.8f".format(y1Original)}, x2=${"%.8f".format(x2Original)}, y2=${"%.8f".format(y2Original)}")
        // Create the bounding box and center point objects.
        val boundingBox = BoundingBox(
            x1Original.toFloat(),
            y1Original.toFloat(),
            x2Original.toFloat(),
            y2Original.toFloat(),
            detection.confidence,
            1 // Class index (or whatever label you're using)
        )
        val center = Point(xCenterOriginal, yCenterOriginal)
        return Pair(boundingBox, center)
    }
    fun drawBoundingBoxes(mat: Mat, box: BoundingBox) {
        val topLeft = Point(box.x1.toDouble(), box.y1.toDouble())
        val bottomRight = Point(box.x2.toDouble(), box.y2.toDouble())
        Imgproc.rectangle(mat, topLeft, bottomRight, Settings.BoundingBox.boxColor, Settings.BoundingBox.boxThickness)
        val label = "User_1 (${("%.2f".format(box.confidence * 100))}%)"
        val fontScale = 0.6
        val thickness = 1
        val baseline = IntArray(1)
        val textSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, fontScale, thickness, baseline)
        val textX = box.x1.toInt()
        val textY = (box.y1 - 5).toInt().coerceAtLeast(10)
        Imgproc.rectangle(
            mat,
            Point(textX.toDouble(), textY.toDouble() + baseline[0]),
            Point(textX + textSize.width, textY - textSize.height),
            Settings.BoundingBox.boxColor,
            Imgproc.FILLED
        )
        Imgproc.putText(
            mat,
            label,
            Point(textX.toDouble(), textY.toDouble()),
            Imgproc.FONT_HERSHEY_SIMPLEX,
            fontScale,
            Scalar(255.0, 255.0, 255.0),
            thickness
        )
    }
    fun createLetterboxedBitmap(srcBitmap: Bitmap, targetWidth: Int, targetHeight: Int, padColor: Scalar = Scalar(0.0, 0.0, 0.0)): Pair<Bitmap, Pair<Int, Int>> {
        val srcMat = Mat().also { Utils.bitmapToMat(srcBitmap, it) }
        val (srcWidth, srcHeight) = (srcMat.cols().toDouble()) to (srcMat.rows().toDouble())
        // Compute scaling factor: use the smaller ratio
        val scale = min(targetWidth / srcWidth, targetHeight / srcHeight)
        val (newWidth, newHeight) = (srcWidth * scale).toInt() to (srcHeight * scale).toInt()
        // Resize the source image
        val resized = Mat().also { Imgproc.resize(srcMat, it, Size(newWidth.toDouble(), newHeight.toDouble())) }
        srcMat.release()
        // Compute padding needed to reach target dimensions
        val (padWidth, padHeight) = (targetWidth - newWidth) to (targetHeight - newHeight)
        val computePadding = { total: Int -> total / 2 to (total - total / 2) }
        val (top, bottom) = computePadding(padHeight)
        val (left, right) = computePadding(padWidth)
        // Create the final letterboxed image with padding
        val letterboxed = Mat().also {Core.copyMakeBorder(resized, it, top, bottom, left, right, Core.BORDER_CONSTANT, padColor)}
        resized.release()
        // Convert the letterboxed Mat back to a Bitmap.
        val outputBitmap = Bitmap.createBitmap(letterboxed.cols(), letterboxed.rows(), srcBitmap.config).apply {
            Utils.matToBitmap(letterboxed, this)
            letterboxed.release()
        }
        // Return the letterboxed image and the top-left padding offset.
        return Pair(outputBitmap, Pair(left, top))
    }
}