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
class ScrcpyConnection(
    private val context: Context,
    private val audioEnabled: Boolean = true,
    /** 试验模式：仅拉取 overlay 流（video=false audio=false）。 */
    private val overlayOnly: Boolean = false,
) {
    var connected = false
        private set

    private var adb: Adb = Adb(context)
    private var callback: EventCallback? = null
    private var streamCnt = 0

    interface EventCallback {
        fun onConnect()
        fun onVideoPrepare(codec: String, width: Int, height: Int)
        fun onVideoPackage(buffer: ByteArray, offset: Int, length: Int)
        /** 音频流就绪：codec 为 4 字节串（"opus" 等）。编解码参数由随后的首帧给出。 */
        fun onAudioPrepare(codec: String)
        /** 音频首帧 = 配置包（OpusHead / csd-0）。必须原样交给解码器，绝不能丢弃。 */
        fun onAudioConfig(csd0: ByteArray)
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
            Log.e(TAG, "control channel opened")
            while (connected) {
                Thread.sleep(100)
            }
            stream.close()
        } catch (e: Exception) {
            Log.e(TAG, "control: ${e.message}")
        }
    }

    private val audioHandler = StreamHandler {
        try {
            val raw = it.openInputStream()
            // 音频流头两种格式（按 socket 序位而异）：
            //  - 首 socket（overlay-only 模式 audio 在前）：64B 设备名 + 4B codec
            //  - 第二 socket（完整流模式，scrcpy 标准）：直接 4B codec
            // 解析器容错：BufferedInputStream mark/reset，先试带前缀，codec 字节
            // 非法则回退无前缀。此前固定按带前缀读，完整流模式下 64B 帧头被当
            // meta 吞掉 → 音频失步无声。
            val stream = java.io.BufferedInputStream(raw, 65536)
            stream.mark(128)
            val meta = ByteArray(64)
            readFully(stream, meta, meta.size)
            val buffer = ByteArray(16384)
            var got = readFully(stream, buffer, 4)
            var codec = if (got) String(buffer.copyOfRange(0, 4), StandardCharsets.US_ASCII) else ""
            if (!codec.all { it.isLetterOrDigit() }) {
                // codec 非法 → 按无前缀格式从头重读
                Log.w(TAG, "audio meta-prefix parse invalid ('$codec'), retry without prefix")
                stream.reset()
                got = readFully(stream, buffer, 4)
                codec = String(buffer.copyOfRange(0, 4), StandardCharsets.US_ASCII)
            }
            Log.e(TAG, "audio channel got=$got codec='$codec' hdr=${buffer.copyOfRange(0, 8).joinToString("") { "%02x".format(it) }}")
            callback?.onAudioPrepare(codec)
            // 帧头 12B：ptsAndFlags(8) + size(4, BE)；首帧 config 位=ptsAndFlags 最高位。
            val header = ByteArray(12)
            var frames = 0
            while (connected) {
                if (!readFully(stream, header, 12)) break
                val isConfig = (header[0].toInt() and 0x80) != 0
                val size = ByteBuffer.wrap(header, 8, 4).order(ByteOrder.BIG_ENDIAN).int
                if (size <= 0 || size > buffer.size) break
                var off = 0
                while (off < size) {
                    val n = stream.read(buffer, off, size - off)
                    if (n < 0) break
                    off += n
                }
                if (off < size) break
                if (isConfig) {
                    Log.i(TAG, "audio config ${buffer.copyOf(minOf(size, 24)).joinToString("") { "%02x".format(it) }}")
                    callback?.onAudioConfig(buffer.copyOf(size))
                } else {
                    callback?.onAudioPackage(buffer, 0, size)
                    frames++
                }
            }
            Log.i(TAG, "audio channel end frames=$frames")
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
            Log.e(TAG, "overlay channel: $codec ${width}x${height}")
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

    /** 服务器回连顺序：camera 视频、音频（可选）、control、overlay（禁用者跳过）。
 *  本工程 overlay 模式下 video 恒关，实测服务端开启顺序为 [audio, control, overlay]，
 *  audio 承载于“首 socket”，前缀带 1 dummy + 64B 设备名（对 display/output 这类
 *  带设备名的首 socket；PC 端在 scrcpy.rs 的 sniff_first_socket_stream 正是这么读的）。 */
    private val handleList: List<StreamHandler> = when {
        overlayOnly && audioEnabled -> listOf(audioHandler, controlHandler, overlayHandler, nullHandler)
        overlayOnly -> listOf(controlHandler, overlayHandler, nullHandler, nullHandler)
        audioEnabled -> listOf(videoHandler, audioHandler, controlHandler, overlayHandler)
        else -> listOf(videoHandler, controlHandler, overlayHandler, nullHandler)
    }

    fun connectAsync(ip: String, port: Int, cb: EventCallback? = null) {
        if (cb != null) callback = cb
        Thread {
            streamCnt = 0
            adb = Adb(context, object : AdbConnection.StreamCallback {
                override fun onOpen(stream: AdbStream?) {
                    if (stream != null) {
                        val idx = streamCnt
                        Log.e("ar-java", "scrcpy stream #$idx opened")
                        Thread { handleList.getOrNull(idx)?.run(stream) }.start()
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
                    val param = if (overlayOnly) {
                        // max_size=640：竖屏设备 overlay 原生 480×640（与真实分辨率一致）；
                        // overlay 模式可再开 audio（audio_source=output 抓眼镜端输出），video 恒关。
                        "log_level=info video=false audio=$audioEnabled max_size=640 overlay=true " +
                            "audio_source=output"
                    } else {
                        "log_level=info video_source=camera audio_source=output " +
                            "max_size=1024 video=true audio=$audioEnabled overlay=true"
                    }
                    Log.e("ar-java", "server param: $param")
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
