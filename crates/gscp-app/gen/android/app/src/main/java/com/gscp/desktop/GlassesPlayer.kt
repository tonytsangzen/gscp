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
    private val mixer: RearComposer,
    private val audioEnabled: Boolean,
    private val bottomRotationDeg: Int,
    private val bottomMirror: Boolean,
    /** 连接成功时下发合成参数（普通模式 = applySettingsToMixer）。 */
    private val applySettings: (RearComposer) -> Unit,
    private val events: Events,
    /** stop() 时是否 reset() 合成器输入面。普通模式 true（handleStopped 语义）；
     *  AR 后摄必须传 false——mixer 跨会话复用，异步 stop 的 reset 若晚于新会话
     *  绑定解码器落地，会把新 SurfaceTexture 拆掉 = overlay/视频永远不更新。
     *  AR 在 startRearPlayerIfReady 里新会话开始时自行 reset。 */
    private val resetMixerOnStop: Boolean = true,
    /** 非空：overlay 解码不落 Surface，改为无 Surface 解码 + I420 Image 回调
     *  （AR 后摄复用前摄 convertOverlayFrame CPU 烘焙，消 GLSL 现场键控的黑残留；
     *  此时 mixer.getTopSurface() 不再被消费）。 */
    private val overlayImageCallback: ((android.media.Image) -> Unit)? = null,
    /** sr=true：GPU 烘焙核心启用 GL 原生 ESPCN ×2 超分（debug.gscp.ovsr）。 */
    private val overlaySr: Boolean = false,
    private val overlaySrWeights: ByteArray? = null,
    /** overlay-only 连接（AR 前摄同款）：不向 server 请求视频流（画面由手机
     *  相机提供），仅 overlay(+audio)；getBottomSurface 不被消费。 */
    private val overlayOnly: Boolean = false,
) {
    interface Events {
        fun onConnect()
        fun onDisconnect()
        fun onError()

        /** overlay 流就绪（scrcpy 线程；解码器接线前后均会触发）。
         *  前摄用于流看门狗布防/状态提示；后摄默认不消费。 */
        fun onOverlayPrepare(width: Int, height: Int) {}

        /** overlay 网络包（scrcpy 线程；seq 为本会话 1 起计数）。
         *  前摄用于诊断计数与 GL 烘焙产物转储节拍；后摄默认不消费。 */
        fun onOverlayPackage(seq: Long) {}
    }

    private var connection: ScrcpyConnection? = null
    private var videoDecoder: VideoDecoder? = null
    private var overlayDecoder: VideoDecoder? = null
    private var audioPlayer: AudioPlayer? = null
    private val platformLock = Object()
    private var overlayPkg = 0L   // 诊断:overlay 网络包计数

    fun connect(ip: String, port: Int = 5555) {
        videoDecoder = VideoDecoder().also { it.diagTag = "rear-video" }
        overlayDecoder = VideoDecoder().also { it.diagTag = "rear-overlay" }
        audioPlayer = AudioPlayer()
        // 单连接完整流（Rokid 眼镜实测无法并存两个 server：视频 server 开相机后
        // overlay server 的采集流会被关闭）。overlay 流为按需帧（UI 变化才发），
        // 显示端保留最后一帧即可（前摄语义）。overlayOnly 时 server 不给视频流。
        connection = ScrcpyConnection(context, audioEnabled, overlayOnly)
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
        if (resetMixerOnStop) mixer.reset()
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

        private var overlayPkg = 0L
        override fun onOverlayPrepare(codec: String, width: Int, height: Int) {
            android.util.Log.i("ar-ui", "rear overlayPrepare $codec ${width}x$height")
            events.onOverlayPrepare(width, height)
            synchronized(platformLock) {
                mixer.setTopAspectRatio(width.toFloat() / height)
                if (overlayImageCallback != null) {
                    // 无 Surface 解码：getOutputImage 给出 CPU 可读 I420（同前摄路径）
                    overlayDecoder?.frameCallback = overlayImageCallback
                    overlayDecoder?.start(width, height, null)
                } else {
                    // 合成器自管 overlay 输入面（ArRearGl = GPU 烘焙核心，
                    // SurfaceMixer = 原 GLSL 现场键控 OES）
                    overlayDecoder?.start(width, height,
                        mixer.openOverlayStream(width, height, overlaySr, overlaySrWeights))
                }
            }
        }

        override fun onOverlayPackage(buffer: ByteArray, offset: Int, length: Int) {
            val seq = ++overlayPkg
            if (seq % 50L == 1L) android.util.Log.i("ar-ui", "rear overlayPkg #$seq ${length}B")
            events.onOverlayPackage(seq)
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
