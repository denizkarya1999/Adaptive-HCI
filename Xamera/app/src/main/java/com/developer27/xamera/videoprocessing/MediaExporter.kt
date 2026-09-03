package com.developer27.xamera.videoprocessing

import android.content.ContentValues
import android.content.Context
import android.media.MediaScannerConnection
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import java.io.File
import java.io.IOException

/** Publishes completed files only, removing unfinished MediaStore entries on failure. */
object MediaExporter {
    fun publish(context: Context, source: File, video: Boolean): Uri {
        val directory = if (video) Environment.DIRECTORY_MOVIES else Environment.DIRECTORY_PICTURES
        val folder = if (video) "Exported Videos from Xamera" else "Exported Lines from Xamera"
        val mimeType = if (video) "video/mp4" else "image/png"
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            @Suppress("DEPRECATION")
            val destination = File(Environment.getExternalStoragePublicDirectory(directory), "$folder/${source.name}")
            check(destination.parentFile?.isDirectory == true || destination.parentFile?.mkdirs() == true)
            try {
                source.copyTo(destination)
            } catch (error: Exception) {
                destination.delete()
                throw error
            }
            MediaScannerConnection.scanFile(context, arrayOf(destination.path), arrayOf(mimeType), null)
            return Uri.fromFile(destination)
        }

        val resolver = context.contentResolver
        val collection = if (video) MediaStore.Video.Media.EXTERNAL_CONTENT_URI else MediaStore.Images.Media.EXTERNAL_CONTENT_URI
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, source.name)
            put(MediaStore.MediaColumns.MIME_TYPE, mimeType)
            put(MediaStore.MediaColumns.RELATIVE_PATH, "$directory/$folder")
            put(MediaStore.MediaColumns.IS_PENDING, 1)
        }
        val uri = resolver.insert(collection, values) ?: throw IOException("Cannot create media entry")
        try {
            resolver.openOutputStream(uri)?.use { output ->
                source.inputStream().use { it.copyTo(output) }
            } ?: throw IOException("Cannot open media output")
            val published = resolver.update(uri, ContentValues().apply {
                put(MediaStore.MediaColumns.IS_PENDING, 0)
            }, null, null)
            check(published == 1) { "Cannot publish media entry" }
            return uri
        } catch (error: Exception) {
            resolver.delete(uri, null, null)
            throw error
        }
    }
}
