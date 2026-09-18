package com.gscp.desktop

import android.content.Context
import android.util.Log
import io.github.muntashirakon.adb.AdbConnection
import io.github.muntashirakon.adb.AdbStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets
import kotlin.math.absoluteValue
import kotlin.random.Random

/**
 * scrcpy 连接会话（移植自参考实现）：
 * 1. adb TCP 连接眼镜 adbd；
 * 2. 推送 assets/scrcpy-server 并以 reverse 隧道拉起；
 * 3. 服务器按 [video, audio, control, overlay] 顺序回连，各路独立线程分流。
 *
 * 不包含 WiFi 配网流程（由桌面端完成）。
 */
class ScrcpyConnection(private val context: Context, private val audioEnabled: Boolean = true) {
    var connected = false
        private set

    private var adb: Adb = Adb(context)
    private var callback: EventCallback? = null
    private var streamCnt = 0

    interface EventCallback {
        fun onConnect()
        fun onVideoPrepare(codec: String, width: Int, height: Int)
        fun onVideoPackage(buffer: ByteArray, offset: Int, length: Int)
        fun onAudioPrepare(codec: String, frameRate: Int, channel: Int)
        fun onAudioPackage(buffer: ByteArray, offset: Int, length: Int)
        fun onOverlayPrepare(codec: String, width: Int, height: Int)
        fun onOverlayPackage(buffer: ByteArray, offset: Int, length: Int)
        fun onDisconnect()
        fun onError()
    }

    fun interface StreamHandler {
        fun run(adbStream: AdbStream)
    }

    /**
     * scrcpy 帧头：8 字节 pts + 4 字节长度（大端），随后为负载。
     */
    private fun readFrame(stream: InputStream, buffer: ByteArray): Int {
        return try {
            val header = ByteArray(12)
            if (!readFully(stream, header, 12)) return -1
            val packageSize = ByteBuffer.wrap(header.copyOfRange(8, 12))
                .order(ByteOrder.BIG_ENDIAN).int
            var offset = 0
            while (offset < packageSize) {
                val read = stream.read(buffer, offset, packageSize - offset)
                if (read < 0) return read
                offset += read
            }
            packageSize
        } catch (e: Exception) {
            Log.d(TAG, "readFrame: ${e.message}")
            -1
        }
    }

    private fun readFully(stream: InputStream, buffer: ByteArray, count: Int): Boolean {
        var readBytes = 0
        while (readBytes < count) {
            val n = stream.read(buffer, readBytes, count - readBytes)
            if (n < 0) return false
            readBytes += n
        }
        return true
    }

    private val nullHandler = StreamHandler { it.close() }

    private val controlHandler = StreamHandler {
        try {
            val stream = it.openOutputStream()
            Log.d(TAG, "control channel")
            while (connected) {
                Thread.sleep(100)
            }
            stream.close()
        } catch (e: Exception) {
            Log.d(TAG, "control: ${e.message}")
        }
    }

    private val audioHandler = StreamHandler {
        try {
            val stream = it.openInputStream()
            val buffer = ByteArray(4096)
            readFully(stream, buffer, 4)
            val codec = String(buffer.copyOfRange(0, 4), StandardCharsets.US_ASCII)
            Log.d(TAG, "audio channel: $codec")
            callback?.onAudioPrepare(codec, 48000, 2)
            readFrame(stream, buffer)
            while (connected) {
                val size = readFrame(stream, buffer)
                if (size > 0) callback?.onAudioPackage(buffer, 0, size)
            }
            stream.close()
        } catch (e: Exception) {
            Log.d(TAG, "audio: ${e.message}")
        }
    }

    private val videoHandler = StreamHandler {
        try {
            val stream = it.openInputStream()
            val buffer = ByteArray(512 * 1024)
            if (!readFully(stream, buffer, 76)) return@StreamHandler

            val codec = String(buffer.copyOfRange(64, 68), StandardCharsets.US_ASCII)
            val width = ByteBuffer.wrap(buffer.copyOfRange(68, 72)).int
            val height = ByteBuffer.wrap(buffer.copyOfRange(72, 76)).int
            Log.d(TAG, "video channel: $codec ${width}x${height}")
            callback?.onVideoPrepare(codec, width, height)
            while (connected) {
                val size = readFrame(stream, buffer)
                if (size > 0) callback?.onVideoPackage(buffer, 0, size)
            }
            stream.close()
        } catch (e: Exception) {
            Log.d(TAG, "video: ${e.message}")
        }
    }

    private val overlayHandler = StreamHandler {
        try {
            val stream = it.openInputStream()
            val buffer = ByteArray(32 * 1024)
            if (!readFully(stream, buffer, 12)) return@StreamHandler

            val codec = String(buffer.copyOfRange(0, 4), StandardCharsets.US_ASCII)
            val width = ByteBuffer.wrap(buffer.copyOfRange(4, 8)).int
            val height = ByteBuffer.wrap(buffer.copyOfRange(8, 12)).int
            Log.d(TAG, "overlay channel: $codec ${width}x${height}")
            callback?.onOverlayPrepare(codec, width, height)
            while (connected) {
                val size = readFrame(stream, buffer)
                if (size > 0) callback?.onOverlayPackage(buffer, 0, size)
            }
            stream.close()
        } catch (e: Exception) {
            Log.d(TAG, "overlay: ${e.message}")
        }
    }

    /** 服务器回连顺序：camera 视频、音频（可选）、control、overlay。 */
    private val handleList: List<StreamHandler> = if (audioEnabled) {
        listOf(videoHandler, audioHandler, controlHandler, overlayHandler)
    } else {
        listOf(videoHandler, controlHandler, overlayHandler, nullHandler)
    }

    fun connectAsync(ip: String, port: Int, cb: EventCallback? = null) {
        if (cb != null) callback = cb
        Thread {
            streamCnt = 0
            adb = Adb(context, object : AdbConnection.StreamCallback {
                override fun onOpen(stream: AdbStream?) {
                    if (stream != null) {
                        val idx = streamCnt
                        Thread {
                            handleList.getOrNull(idx)?.run(stream)
                        }.start()
                        streamCnt++
                    }
                }

                override fun onClose(p0: AdbStream?) {}
            })
            try {
                connected = adb.connect(ip, port)
                if (connected) {
                    callback?.onConnect()

                    val id = "%08x".format(Random.nextInt().absoluteValue)
                    adb.push(context.assets.open("scrcpy-server"), "/data/local/tmp/scrcpy-server.jar")
                    adb.reverse("forward:localabstract:scrcpy_$id;tcp:27813")
                    val param = "log_level=info video_source=camera audio_source=output " +
                        "max_size=1024 video=true audio=$audioEnabled overlay=true"
                    adb.run(
                        "CLASSPATH=/data/local/tmp/scrcpy-server.jar app_process / " +
                            "com.genymobile.scrcpy.Server 3.3.1 scid=$id $param"
                    )
                    Log.d(TAG, "server exit")
                    adb.disconnect()
                    connected = false
                    callback?.onDisconnect()
                } else {
                    callback?.onError()
                }
            } catch (e: Exception) {
                Log.d(TAG, "server error ${e.message}")
                connected = false
                callback?.onError()
            }
        }.start()
    }

    fun disconnect() {
        try {
            if (connected) {
                adb.run("killall app_process")
            }
            adb.disconnect()
        } catch (_: Exception) {
        } finally {
            connected = false
        }
    }

    companion object {
        private const val TAG = "gscp-scrcpy"
    }
}
