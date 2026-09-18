package com.gscp.desktop

import android.content.Context
import android.os.Build
import android.util.Base64
import android.util.Log
import io.github.muntashirakon.adb.AbsAdbConnectionManager
import io.github.muntashirakon.adb.AdbConnection.StreamCallback
import io.github.muntashirakon.adb.LocalServices
import java.io.InputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.security.KeyFactory
import java.security.PrivateKey
import java.security.cert.Certificate
import java.security.cert.CertificateFactory
import java.security.cert.X509Certificate
import java.security.spec.PKCS8EncodedKeySpec
import java.util.concurrent.TimeUnit

/**
 * adb over TCP 客户端：手机直连眼镜 adbd（WiFi adb），用于推送并拉起
 * scrcpy-server。密钥对从 assets/ca.key、assets/ca.crt 读取。
 */
class Adb(context: Context, val callback: StreamCallback? = null) : AbsAdbConnectionManager() {
    private val mPrivateKey: PrivateKey
    private val mCertificate: Certificate

    override fun getPrivateKey(): PrivateKey = mPrivateKey

    override fun getCertificate(): Certificate = mCertificate

    override fun getDeviceName(): String = "gscp"

    init {
        api = Build.VERSION.SDK_INT
        setTimeout(3, TimeUnit.SECONDS)
        val asset = context.assets
        InputStreamReader(asset.open("ca.key")).use {
            val encoded = Base64.decode(it.readText(), 0)
            mPrivateKey = KeyFactory.getInstance("RSA").generatePrivate(
                PKCS8EncodedKeySpec(encoded)
            )
        }
        asset.open("ca.crt").use {
            val certFactory = CertificateFactory.getInstance("X.509")
            mCertificate = certFactory.generateCertificate(it) as X509Certificate
        }
    }

    override fun connect(ip: String, port: Int): Boolean {
        setTimeout(3, TimeUnit.SECONDS)
        val ret = super.connect(ip, port)
        if (ret) {
            adbConnection?.RegisterStreamCallback(callback)
        }
        return ret
    }

    fun run(cmd: String) {
        try {
            Log.d(TAG, "shell:$cmd...")
            openStream("shell:$cmd").use {
                val reader = it.openInputStream().reader().buffered()
                while (true) {
                    val line = reader.readLine() ?: break
                    Log.d(TAG, line)
                }
            }
        } catch (e: Exception) {
            Log.d(TAG, "shell 失败: ${e.message}")
        }
    }

    fun reverse(cmd: String) = try {
        openStream(LocalServices.REVERSE, cmd).use {
            Log.d(TAG, "reverse:$cmd -> ${it.openInputStream().getResponse(1000)}")
        }
    } catch (e: Exception) {
        Log.d(TAG, "reverse 失败: ${e.message}")
    }

    private fun genDataPackage(cmd: String, param: Int, data: ByteArray, size: Int): ByteArray {
        val buf = ByteBuffer.allocate(cmd.length + size + 4)
        buf.put(cmd.toByteArray(Charsets.US_ASCII))
        buf.put((param shr 0).toByte())
        buf.put((param shr 8).toByte())
        buf.put((param shr 16).toByte())
        buf.put((param shr 24).toByte())
        buf.put(data, 0, size)
        return buf.array()
    }

    private fun genDataPackage(cmd: String, data: ByteArray, size: Int): ByteArray {
        return genDataPackage(cmd, size, data, size)
    }

    private fun genDataPackage(cmd: String, data: ByteArray): ByteArray {
        return genDataPackage(cmd, data.size, data, data.size)
    }

    private fun genDataPackage(cmd: String, data: String): ByteArray {
        return genDataPackage(cmd, data.toByteArray(Charsets.US_ASCII))
    }

    private fun genDataPackage(cmd: String, param: Int): ByteArray {
        val buf = ByteBuffer.allocate(cmd.length + 4)
        buf.put(cmd.toByteArray(Charsets.US_ASCII))
        buf.put((param shr 0).toByte())
        buf.put((param shr 8).toByte())
        buf.put((param shr 16).toByte())
        buf.put((param shr 24).toByte())
        return buf.array()
    }

    fun push(local: InputStream, remote: String): Boolean {
        try {
            openStream("sync:recv:").use {
                Log.d(TAG, "push to $remote")
                val buf = ByteArray(65500)
                val writer = it.openOutputStream()
                val reader = it.openInputStream()
                writer.write(genDataPackage("SEND", "$remote,33188"))
                while (true) {
                    val len = local.read(buf)
                    if (len <= 0) break
                    writer.write(genDataPackage("DATA", buf, len))
                }
                writer.write(genDataPackage("DONE", (System.currentTimeMillis() / 1000).toInt()))
                val ret = reader.getResponse(1000)
                return ret
            }
        } catch (e: Exception) {
            Log.d(TAG, "push 失败: ${e.message}")
            return false
        }
    }

    companion object {
        private const val TAG = "gscp-adb"
    }
}
