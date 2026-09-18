package com.gscp.desktop

import android.annotation.SuppressLint
import android.content.SharedPreferences
import android.graphics.Point
import android.media.AudioFormat
import android.os.Bundle
import android.util.Patterns
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.WindowManager
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.ProgressBar
import android.widget.SeekBar
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity

/**
 * Android 端入口：原生 scrcpy 连接 + 播放界面。
 *
 * 仅实现「scrcpy 连接、播放」——输入已在桌面端配好网的眼镜 IP，
 * 经 adb TCP 拉起 scrcpy-server 并播放 camera/overlay/音频三路。
 * 不包含 Wi-Fi 配网流程。
 */
class MainActivity : AppCompatActivity() {
    private lateinit var prefs: SharedPreferences

    private lateinit var settingsPanel: android.view.View
    private lateinit var playerPanel: FrameLayout
    private lateinit var surfaceView: SurfaceView
    private lateinit var progressBar: ProgressBar
    private lateinit var ipEdit: EditText
    private lateinit var overlayScale: SeekBar

    private lateinit var mixer: SurfaceMixer
    private lateinit var audioPlayer: AudioPlayer
    private var videoDecoder: VideoDecoder? = null
    private var overlayDecoder: VideoDecoder? = null
    private var connection: ScrcpyConnection? = null

    private val platformLock = Object()
    private var playing = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_player)

        prefs = getSharedPreferences("gscp", MODE_PRIVATE)

        settingsPanel = findViewById(R.id.settings_panel)
        playerPanel = findViewById(R.id.player_panel)
        surfaceView = findViewById(R.id.video_surface)
        progressBar = findViewById(R.id.progress_bar)
        ipEdit = findViewById(R.id.ip_address)
        overlayScale = findViewById(R.id.overlay_scale)
        val connectButton = findViewById<Button>(R.id.button_connect)

        // 画面区域：竖屏下 3:4，横屏下保持高度铺满
        val size = Point()
        @Suppress("DEPRECATION")
        windowManager.defaultDisplay.getRealSize(size)
        val (w, h) = if (size.x <= size.y) {
            size.x to size.x * 4 / 3
        } else {
            size.y to size.y * 3 / 4
        }
        surfaceView.layoutParams = FrameLayout.LayoutParams(w, h).apply {
            gravity = android.view.Gravity.CENTER
        }

        mixer = SurfaceMixer(this, w, h)
        audioPlayer = AudioPlayer(48000, AudioFormat.CHANNEL_OUT_STEREO, AudioFormat.ENCODING_PCM_16BIT)

        surfaceView.holder.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceCreated(holder: SurfaceHolder) {}
            override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
                mixer.attachOutputSurface(holder.surface)
            }

            override fun surfaceDestroyed(holder: SurfaceHolder) {
                mixer.detachOutputSurface(holder.surface)
            }
        })

        ipEdit.setText(prefs.getString("ip", ""))
        overlayScale.progress = prefs.getInt("overlayScale", 50)
        overlayScale.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(bar: SeekBar, value: Int, fromUser: Boolean) {
                mixer.setTopScale(value)
            }

            override fun onStartTrackingTouch(bar: SeekBar) {}
            override fun onStopTrackingTouch(bar: SeekBar) {
                prefs.edit().putInt("overlayScale", bar.progress).apply()
            }
        })

        connectButton.setOnClickListener { connect() }
    }

    private fun connect() {
        val ip = ipEdit.text.toString().trim()
        if (!Patterns.IP_ADDRESS.matcher(ip).matches()) {
            Toast.makeText(this, "请输入有效的 IP 地址", Toast.LENGTH_SHORT).show()
            return
        }
        prefs.edit().putString("ip", ip).apply()

        videoDecoder = VideoDecoder()
        overlayDecoder = VideoDecoder()
        connection = ScrcpyConnection(this)

        settingsPanel.visibility = android.view.View.GONE
        playerPanel.visibility = android.view.View.VISIBLE
        progressBar.visibility = android.view.View.VISIBLE
        playing = true

        connection!!.connectAsync(ip, 5555, callback)
    }

    private fun disconnect() {
        playing = false
        connection?.disconnect()
        handleStopped(withError = false)
    }

    private fun handleStopped(withError: Boolean) {
        runOnUiThread {
            progressBar.visibility = android.view.View.GONE
            videoDecoder?.stop()
            videoDecoder = null
            overlayDecoder?.stop()
            overlayDecoder = null
            audioPlayer.stop()
            mixer.reset()
            if (playing.not() || withError) {
                playerPanel.visibility = android.view.View.GONE
                settingsPanel.visibility = android.view.View.VISIBLE
            }
            if (withError) {
                Toast.makeText(this, "连接失败，请检查眼镜 IP 与网络", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private val callback = object : ScrcpyConnection.EventCallback {
        override fun onConnect() {
            runOnUiThread {
                progressBar.visibility = android.view.View.GONE
            }
            mixer.setTopScale(overlayScale.progress)
        }

        override fun onVideoPrepare(codec: String, width: Int, height: Int) {
            synchronized(platformLock) {
                mixer.setBottomAspectRatio(height.toFloat() / width)
                mixer.setBottomRotation(90f, false)
                videoDecoder?.start(width, height, mixer.getBottomSurface())
            }
        }

        override fun onVideoPackage(buffer: ByteArray, offset: Int, length: Int) {
            videoDecoder?.decode(buffer, offset, length)
        }

        override fun onAudioPrepare(codec: String, frameRate: Int, channel: Int) {
            audioPlayer.start()
        }

        override fun onAudioPackage(buffer: ByteArray, offset: Int, length: Int) {
            audioPlayer.play(buffer, offset, length)
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
            if (playing) handleStopped(withError = false)
        }

        override fun onError() {
            playing = false
            handleStopped(withError = true)
        }
    }

    @Deprecated("Deprecated in Java")
    @SuppressLint("MissingSuperCall")
    override fun onBackPressed() {
        if (playing) {
            disconnect()
        } else {
            finish()
        }
    }

    override fun onDestroy() {
        playing = false
        connection?.disconnect()
        super.onDestroy()
    }
}
