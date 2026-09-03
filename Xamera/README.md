# Xamera 1.0.1

This maintenance release fixes startup, tracking and export failures:

- OpenCV now comes from its official Maven Android package, including the native libraries. The old local SDK module excluded `native/libs` from Git, producing APKs that crashed at startup.
- Saved detection and export preferences are restored on startup and resume. Fresh installs use the defaults shown in Settings: contour detection, with image and video exports off.
- Frame processing, model loading, inference and recorder shutdown share one worker. Stopping waits for the last frame, and Clear rejects pending results from the old session. Empty traces no longer produce arbitrary predictions.
- Recognition uses the same pixel ranges as the training scripts: 0–1 for YOLO and −1–1 for the letter and digit classifiers. Interpreters are closed when the activity is destroyed; CPU execution avoids device-specific delegate and thread-affinity failures.
- Each processor owns its trace and Kalman filter. New strokes reset the filter position; exported ink is opaque and `.png` files contain PNG data.
- Video encoding uses YUV input images with bounded shutdown, actual frame timestamps and empty-recording cleanup. Completed images and videos are published through MediaStore on Android 10+, with legacy storage permission requested only when exporting on Android 8–9.
- Camera opening and preview callbacks reject stale sessions, camera permission is checked before switching, zoom respects the sensor's supported range, and missing email/dialer apps show an error instead of crashing.

## Build and test

Use JDK 17 or newer with Gradle 8.9, Android SDK Platform 34 and Build Tools 34.0.0. Set `sdk.dir` in the untracked `local.properties` file to your Android SDK directory.

From this directory:

```sh
bash gradlew :app:assembleRelease :app:testDebugUnitTest :app:lintDebug
bash gradlew :app:assembleDebug :app:assembleDebugAndroidTest
```

The installable release APK is `app/build/outputs/apk/release/app-release.apk`. The project retains its existing debug-key signing configuration for sideloading; it does not use a production signing key. Keep the same local signing key for future in-place upgrades. The APK supports ARM64 and ARMv7 devices on Android 8.0 or later.

On a connected test device, install the app and test APK without clearing app data, then run the instrumentation suite:

```sh
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb install -r app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
adb shell pm grant com.developer27.xamera android.permission.CAMERA
adb shell am instrument -w com.developer27.xamera.test/androidx.test.runner.AndroidJUnitRunner
```

`ProcessingRegressionTest` verifies settings restoration, PNG contents and MediaStore publication, playable video and empty-stop behavior, independent/reset traces, opaque inference images, and all three bundled models on the device. Test media is removed after verification. `CameraLifecycleRegressionTest` also checks that real camera frames resume after rapid switching and backgrounding, and that Clear rejects pending predictions.

The AR component is supplied as a prebuilt AAR. Full glove recognition accuracy and AR placement still need a manual check with the physical glove and an AR scene.

## Validation

Verified on September 3, 2026: release build, JVM unit test and Android lint pass; all 9 instrumentation tests pass on a Pixel 8a, including live camera processing, rapid camera switches and pause/resume.

## Platform references

- [OpenCV's official Android distribution](https://opencv.org/opencv4android-usage-models/)
- [Android MediaCodec input-image contract](https://developer.android.com/reference/android/media/MediaCodec)
- [Android MediaStore publication fields](https://developer.android.com/reference/android/provider/MediaStore.MediaColumns)
