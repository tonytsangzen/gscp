package com.gscp.desktop

import android.annotation.SuppressLint
import android.content.SharedPreferences
import android.graphics.Point
import android.media.AudioFormat
import android.os.Bundle
import android.util.Patterns
import android.view.Gravity
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.view.WindowManager
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.SeekBar
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat

/**
 * Android 端入口：原生 scrcpy 连接 + 播放界面。
 *
 * 仅实现「scrcpy 连接、播放」——输入已在桌面端配好网的眼镜 IP，
 * 经 adb TCP 拉起 scrcpy-server 并播放 camera/overlay/音频三路。
 * 不包含 Wi-Fi 配网流程。播放时进入沉浸模式。
 */
class MainActivity : AppCompatActivity() {
    private lateinit var prefs: SharedPreferences

    private lateinit var settingsPanel: View
    private lateinit var settingsButton: ImageButton
    private lateinit var playerPanel: FrameLayout
    private lateinit var surfaceView: SurfaceView
    private lateinit var progressBar: ProgressBar
    private lateinit var ipEdit: EditText

    private lateinit var mixer: SurfaceMixer
    private lateinit var audioPlayer: AudioPlayer
    private var videoDecoder: VideoDecoder? = null
    private var overlayDecoder: VideoDecoder? = null
    private var connection: ScrcpyConnection? = null

    private val platformLock = Object()
    private var playing = false

    // 可配置参数（连接页右上角设置）
    private var topScalePercent = 50
    private var bottomRotationDeg = 90
    private var bottomMirror = false
    private var topRotationDeg = 0
    private var topMirror = false
    private var audioEnabled = true

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_player)

        prefs = getSharedPreferences("gscp", MODE_PRIVATE)

        settingsPanel = findViewById(R.id.settings_panel)
        settingsButton = findViewById(R.id.button_settings)
        playerPanel = findViewById(R.id.player_panel)
        surfaceView = findViewById(R.id.video_surface)
        progressBar = findViewById(R.id.progress_bar)
        ipEdit = findViewById(R.id.ip_address)
        val connectButton = findViewById<Button>(R.id.button_connect)

        loadSettings()

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
            gravity = Gravity.CENTER
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
        settingsButton.setOnClickListener { openSettingsDialog() }
        connectButton.setOnClickListener { connect() }
    }

    // ── 参数设置 ──────────────────────────────────────────────

    private fun loadSettings() {
        topScalePercent = prefs.getInt("topScalePercent", 50)
        bottomRotationDeg = prefs.getInt("bottomRotationDeg", 90)
        bottomMirror = prefs.getBoolean("bottomMirror", false)
        topRotationDeg = prefs.getInt("topRotationDeg", 0)
        topMirror = prefs.getBoolean("topMirror", false)
        audioEnabled = prefs.getBoolean("audioEnabled", true)
        applySettingsToMixer()
    }

    private fun applySettingsToMixer() {
        mixer.setTopScale(topScalePercent)
        mixer.setBottomRotation(bottomRotationDeg.toFloat(), bottomMirror)
        mixer.setTopRotation(topRotationDeg.toFloat(), topMirror)
    }

    @SuppressLint("InflateParams")
    private fun openSettingsDialog() {
        val pad = (20 * resources.displayMetrics.density).toInt()
        val halfPad = (6 * resources.displayMetrics.density).toInt()
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, halfPad, pad, halfPad)
        }

        fun label(text: String): TextView = TextView(this).apply {
            this.text = text
            textSize = 15f
            setPadding(0, halfPad, 0, halfPad)
        }

        val rotationOptions = arrayOf("0°", "90°", "180°", "270°")
        fun rotationSpinner(initialDeg: Int): Spinner = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_item,
                rotationOptions
            ).also { it.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item) }
            setSelection(initialDeg / 90)
        }

        // overlay 缩放
        container.addView(label("overlay 缩放"))
        val scaleBar = SeekBar(this).apply { max = 100; progress = topScalePercent }
        container.addView(scaleBar)

        // 底图（camera）旋转/镜像
        container.addView(label("底图旋转"))
        val bottomSpin = rotationSpinner(bottomRotationDeg)
        container.addView(bottomSpin)
        val bottomMirrorBox = CheckBox(this).apply { text = "底图镜像"; isChecked = bottomMirror }
        container.addView(bottomMirrorBox)

        // overlay 旋转/镜像
        container.addView(label("overlay 旋转"))
        val topSpin = rotationSpinner(topRotationDeg)
        container.addView(topSpin)
        val topMirrorBox = CheckBox(this).apply { text = "overlay 镜像"; isChecked = topMirror }
        container.addView(topMirrorBox)

        // 音频（下次连接生效）
        val audioBox = CheckBox(this).apply { text = "播放声音（下次连接生效）"; isChecked = audioEnabled }
        container.addView(audioBox)

        AlertDialog.Builder(this)
            .setTitle("参数设置")
            .setView(android.widget.ScrollView(this).apply { addView(container) })
            .setPositiveButton("保存") { _, _ ->
                topScalePercent = scaleBar.progress
                bottomRotationDeg = bottomSpin.selectedItemPosition * 90
                bottomMirror = bottomMirrorBox.isChecked
                topRotationDeg = topSpin.selectedItemPosition * 90
                topMirror = topMirrorBox.isChecked
                audioEnabled = audioBox.isChecked
                prefs.edit()
                    .putInt("topScalePercent", topScalePercent)
                    .putInt("bottomRotationDeg", bottomRotationDeg)
                    .putBoolean("bottomMirror", bottomMirror)
                    .putInt("topRotationDeg", topRotationDeg)
                    .putBoolean("topMirror", topMirror)
                    .putBoolean("audioEnabled", audioEnabled)
                    .apply()
                applySettingsToMixer()
            }
            .setNegativeButton("取消", null)
            .show()
    }

    // ── 连接 / 播放 ──────────────────────────────────────────

    private fun connect() {
        val ip = ipEdit.text.toString().trim()
        if (!Patterns.IP_ADDRESS.matcher(ip).matches()) {
            Toast.makeText(this, "请输入有效的 IP 地址", Toast.LENGTH_SHORT).show()
            return
        }
        prefs.edit().putString("ip", ip).apply()

        videoDecoder = VideoDecoder()
        overlayDecoder = VideoDecoder()
        connection = ScrcpyConnection(this, audioEnabled)

        settingsPanel.visibility = View.GONE
        settingsButton.visibility = View.GONE
        playerPanel.visibility = View.VISIBLE
        progressBar.visibility = View.VISIBLE
        playing = true
        enterImmersive()

        connection!!.connectAsync(ip, 5555, callback)
    }

    private fun disconnect() {
        playing = false
        connection?.disconnect()
        handleStopped(withError = false)
    }

    private fun handleStopped(withError: Boolean) {
        runOnUiThread {
            progressBar.visibility = View.GONE
            videoDecoder?.stop()
            videoDecoder = null
            overlayDecoder?.stop()
            overlayDecoder = null
            audioPlayer.stop()
            mixer.reset()
            playing = false
            exitImmersive()
            playerPanel.visibility = View.GONE
            settingsPanel.visibility = View.VISIBLE
            settingsButton.visibility = View.VISIBLE
            if (withError) {
                Toast.makeText(this, "连接失败，请检查眼镜 IP 与网络", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private val callback = object : ScrcpyConnection.EventCallback {
        override fun onConnect() {
            runOnUiThread {
                progressBar.visibility = View.GONE
            }
            applySettingsToMixer()
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

        override fun onAudioPrepare(codec: String, frameRate: Int, channel: Int) {
            if (audioEnabled) audioPlayer.start()
        }

        override fun onAudioPackage(buffer: ByteArray, offset: Int, length: Int) {
            if (audioEnabled) audioPlayer.play(buffer, offset, length)
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

    // ── 沉浸模式 ─────────────────────────────────────────────

    private fun enterImmersive() {
        WindowCompat.setDecorFitsSystemWindows(window, false)
        val controller = WindowInsetsControllerCompat(window, window.decorView)
        controller.systemBarsBehavior =
            WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        controller.hide(WindowInsetsCompat.Type.systemBars())
    }

    private fun exitImmersive() {
        val controller = WindowInsetsControllerCompat(window, window.decorView)
        controller.show(WindowInsetsCompat.Type.systemBars())
        WindowCompat.setDecorFitsSystemWindows(window, true)
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        // 沉浸模式下从通知/对话框返回时重新隐藏系统栏
        if (hasFocus && playing) {
            enterImmersive()
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
