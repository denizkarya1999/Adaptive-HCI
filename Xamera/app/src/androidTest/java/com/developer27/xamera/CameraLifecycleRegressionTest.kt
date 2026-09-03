package com.developer27.xamera

import android.os.SystemClock
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.lifecycle.Lifecycle
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

/** Requires camera permission on the test device; no recordings are exported by default. */
@RunWith(AndroidJUnit4::class)
class CameraLifecycleRegressionTest {
    private fun awaitUi(scenario: ActivityScenario<MainActivity>, condition: (MainActivity) -> Boolean) {
        val deadline = SystemClock.elapsedRealtime() + 15_000
        while (SystemClock.elapsedRealtime() < deadline) {
            var ready = false
            scenario.onActivity { ready = condition(it) }
            if (ready) return
            SystemClock.sleep(100)
        }
        fail("Xamera did not reach the expected UI state within 15 seconds")
    }

    private fun startTracking(scenario: ActivityScenario<MainActivity>) {
        awaitUi(scenario) { activity ->
            val button = activity.findViewById<Button>(R.id.startProcessingButton)
            if (button.text == "Stop Tracking") true else {
                if (button.isEnabled) button.performClick()
                false
            }
        }
        awaitUi(scenario) { it.findViewById<ImageView>(R.id.processedFrameView).drawable != null }
    }

    @Test
    fun clearWhileTrackingRejectsPendingPrediction() {
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            startTracking(scenario)
            scenario.onActivity { it.findViewById<View>(R.id.clearPredictionButton).performClick() }
            awaitUi(scenario) { it.findViewById<Button>(R.id.startProcessingButton).text == "Start Tracking" }
            // Let the queued frame's UI callback run after the clear action.
            SystemClock.sleep(250)
            scenario.onActivity {
                assertEquals("No Prediction Yet", it.findViewById<TextView>(R.id.predictedLetterTextView).text.toString())
                assertEquals(View.GONE, it.findViewById<ImageView>(R.id.processedFrameView).visibility)
            }
        }
    }

    @Test
    fun cameraResumesAfterRapidSwitchAndBackgrounding() {
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            startTracking(scenario)
            scenario.onActivity {
                it.findViewById<Button>(R.id.startProcessingButton).performClick()
                repeat(3) { _ -> it.findViewById<View>(R.id.switchCameraButton).performClick() }
            }
            scenario.moveToState(Lifecycle.State.CREATED)
            scenario.moveToState(Lifecycle.State.RESUMED)
            startTracking(scenario)
            scenario.onActivity { it.findViewById<View>(R.id.clearPredictionButton).performClick() }
            awaitUi(scenario) { it.findViewById<Button>(R.id.startProcessingButton).text == "Start Tracking" }
        }
    }
}
