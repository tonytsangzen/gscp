package com.gscp.desktop

/**
 * 普通模式播放管线（与 MainActivity 完全同源，逻辑逐行照搬）：
 * scrcpy 全流（相机视频 + overlay + 音频）→ 双 VideoDecoder 硬解直写 SurfaceMixer
 * 两个输入面 → GL 合成输出。AR 后摄模式直接复用此类，不与前摄 Canvas 管线共享
 * 任何连接/解码/音频状态。
 *
 * 生命周期：connect() 建立连接并拉起三路流；stop() 断开并释放解码器、音频，
 * 同时 reset() 合成器输入面（与普通模式 handleStopped 相同的收尾）。
 * 事件回调在 scrcpy 线程触发，调用方自行切线程。
 */
class GlassesPlayer(
    private val context: android.content.Context,
    private val mixer: SurfaceMixer,
    private val audioEnabled: Boolean,
    private val bottomRotationDeg: Int,
    private val bottomMirror: Boolean,
    /** 连接成功时下发合成参数（普通模式 = applySettingsToMixer）。 */
    private val applySettings: (SurfaceMixer) -> Unit,
    private val events: Events,
) {
    interface Events {
        fun onConnect()
        fun onDisconnect()
        fun onError()
    }

    private var connection: ScrcpyConnection? = null
    private var videoDecoder: VideoDecoder? = null
    private var overlayDecoder: VideoDecoder? = null
    private var audioPlayer: AudioPlayer? = null
    private val platformLock = Object()

    fun connect(ip: String, port: Int = 5555) {
        videoDecoder = VideoDecoder()
        overlayDecoder = VideoDecoder()
        audioPlayer = AudioPlayer()
        connection = ScrcpyConnection(context, audioEnabled)
        connection!!.connectAsync(ip, port, callback)
    }

    /** 断开并释放全部媒体资源（含 mixer.reset()，与普通模式 handleStopped 一致）。 */
    fun stop() {
        connection?.disconnect()
        videoDecoder?.stop()
        videoDecoder = null
        overlayDecoder?.stop()
        overlayDecoder = null
        audioPlayer?.stop()
        audioPlayer = null
        mixer.reset()
    }

    private val callback = object : ScrcpyConnection.EventCallback {
        override fun onConnect() {
            applySettings(mixer)
            events.onConnect()
        }

        override fun onVideoPrepare(codec: String, width: Int, height: Int) {
            synchronized(platformLock) {
                mixer.setBottomAspectRatio(height.toFloat() / width)
                mixer.setBottomRotation(bottomRotationDeg.toFloat(), bottomMirror)
                videoDecoder?.start(width, height, mixer.getBottomSurface())
            }
        }

        override fun onVideoPackage(buffer: ByteArray, offset: Int, length: Int) {
            videoDecoder?.decode(buffer, offset, length)
        }

        override fun onAudioPrepare(codec: String) {
            // 解码参数由首帧 OpusHead（onAudioConfig）给出，此处无需处理
        }

        override fun onAudioConfig(csd0: ByteArray) {
            if (audioEnabled) audioPlayer?.start(csd0)
        }

        override fun onAudioPackage(buffer: ByteArray, offset: Int, length: Int) {
            if (audioEnabled) audioPlayer?.play(buffer, offset, length)
        }

        override fun onOverlayPrepare(codec: String, width: Int, height: Int) {
            synchronized(platformLock) {
                mixer.setTopAspectRatio(width.toFloat() / height)
                overlayDecoder?.start(width, height, mixer.getTopSurface())
            }
        }

        override fun onOverlayPackage(buffer: ByteArray, offset: Int, length: Int) {
            overlayDecoder?.decode(buffer, offset, length)
        }

        override fun onDisconnect() {
            events.onDisconnect()
        }

        override fun onError() {
            events.onError()
        }
    }
}
