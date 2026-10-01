package com.gscp.desktop;

import android.graphics.Bitmap;
import android.graphics.SurfaceTexture;
import android.opengl.EGL14;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLES30;
import android.view.Surface;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/**
 * overlay 内容烘焙核心（纯 Java：APK 与离线 app_process 测试 jar 共用同一份源码）。
 *
 * 效果链与旧 CPU 烘焙（ArActivity.convertOverlayFrame）逐项对齐，全部 GPU 化：
 *  - 解码直写 SurfaceTexture（OES，零拷贝进 GPU）；
 *  - 内容：裁上下空带（480×640 → 中段 480²）+ 水平镜像 + BT.601 limited YUV→RGB
 *    （OES 采样器内转换）+ keyLut 黑键羽化（luma 5.1..86.7 → pow 1.2）+ 饱和度 ×1.12，
 *    预乘输出（等价 CPU setPixels → GLUtils 预乘上传）；
 *  - debug.gscp.ovsr=1（默认）：内容在键控前先过 GL 原生 ESPCN ×2 灰度超分
 *    （ncnn-benchmark sr_benchmark 同网络同权重，内容方区 480² → 960²），文字/图标
 *    显著增锐；色度走 aux 双线性移植。conv1(1→64)+ReLU → conv2(64→32)+ReLU →
 *    conv3(32→4)+PixelShuffle×2，全部同步 GPU pass（数毫秒），无线程、无读回、
 *    像素级确定。（历史：ncnn 执行载体在此设备不可用——Vulkan fp16 输出损坏、
 *    fp32 偶发漂移、CPU 320ms/帧且同步跑会冻结包括相机在内的整个合成线程，故弃用。）
 *  - 光晕 tint：全分辨率遮罩编码 → 两级面积平均 → 小图读回（延迟读、每 4 帧刷新）；
 *  - 光晕金字塔：1/4 剪影 → 1/16 往返两轮 → 1/32 远距层，与 CPU 位图尺寸一致；
 *  - 产物 contentTex/glowTex/bloomTex 为预乘纹理，供 ArFrontGl/ArRearGl 的
 *    FRAG_MESH 合成（行 0 = 图像顶行，与 GLUtils 位图上传约定一致）。
 *
 * SR 实现要点：多通道激活写入"切片纹理"（每 4 通道一个 480² 视口切片的 RGBA16F，
 * 64ch = 16 切片），规避 GLSL 动态 sampler 索引；权重在 R32F 纹理用 texelFetch 读取。
 * 需 ES3 上下文 + EXT_color_buffer_float（RGBA16F 渲染目标），否则自动退回非 SR。
 *
 * 线程约束：GL 方法必须在持有 EGL context 的线程（合成器 GL 线程）调用。
 */
public final class OverlayBakeCore {
    private static final String TAG = "ov-bake";

    /** 黑键羽化区间（与 CPU keyLut 同参：0.02..0.34 ×255）。 */
    private static final float KEY_LO = Math.max(0.08f - 0.06f, 0f) * 255f;
    private static final float KEY_SPAN = Math.min(0.28f + 0.06f, 1f) * 255f - KEY_LO;
    private static final String KEY_LO_F = Float.toString(KEY_LO);
    private static final String KEY_SPAN_F = Float.toString(KEY_SPAN);

    // ── 输入路径（解码端）────────────────────────────────────
    private int oesTex = 0;
    private SurfaceTexture st;
    private Surface inputSurface;
    private int streamW = 0, streamH = 0;
    private android.os.Handler listenerHandler;

    // ── 烘焙产物（预乘纹理 + FBO）───────────────────────────
    private int contentTex, contentFbo, contentW, contentH;
    private int glowTex, glowFbo, glowW, glowH;       // 1/4 剪影 + 金字塔往返
    private int halfTex, halfFbo;                     // 1/16 低通中间层
    private int farTex, farFbo, farW, farH;           // 1/32 远距光晕
    private int encTex, encFbo, enc2Tex, enc2Fbo;     // tint 遮罩链
    private int encMidTex, encMidFbo, encMidW, encMidH;
    private int encW, encH, enc2W, enc2H;
    private boolean avgDirty = false;
    private boolean lastOpaque = false;
    private int avgTick = 0;

    // ── SR 链路（GL 原生 ESPCN ×2，sr=true 时启用）────────────────
    private boolean srActive = false;
    private int auxTex, auxFbo, auxW, auxH;           // SR 输入（YUV→RGB 方区）
    private int srcW, srcH;
    private int act1Tex, act1Fbo;                     // 64ch：480 × (16*480) 切片
    private int act2Tex, act2Fbo;                     // 32ch：480 × (8*480)
    private int wt1Tex, wt2Tex, wt3Tex;               // 权重 R32F：(9×64)/(9×2048)/(9×129)
    private int pAux, pS1, pS2, pFinal;
    private int locAuxOes, locXfAux, locBandAux, locStreamAux;
    private int locA1Tex, locA2Tex, locAuxChroma, locW1, locW2, locW3;
    private float[] srB0, srB1;                       // conv1/conv2 bias（final 的 b2 打进 WT3 末行）

    private volatile boolean pending = false;
    private volatile boolean baked = false;
    private volatile boolean released = false;
    /** 仅离线测试注入：非空时替代 SurfaceTexture.getTransformMatrix。 */
    private volatile float[] xfOverride = null;
    private long version = 0;
    private long bakedFrames = 0;
    private final float[] avg = new float[3];
    private final ByteBuffer readBuf;
    /** 仅离线测试：内容 pass 后读回中心像素诊断。 */
    public boolean debugProbe = false;

    /** 帧回调只置标志；updateTexImage/bake 必须在 GL 线程。 */
    private final SurfaceTexture.OnFrameAvailableListener listener =
        new SurfaceTexture.OnFrameAvailableListener() {
            @Override public void onFrameAvailable(SurfaceTexture surfaceTexture) {
                if (!released) pending = true;
            }
        };

    public OverlayBakeCore() {
        readBuf = ByteBuffer.allocateDirect(64 * 64 * 4).order(ByteOrder.nativeOrder());
    }

    // ── 生命周期（GL 线程）──────────────────────────────────

    /** 建立输入路径与全部产物纹理。重复调用按新建处理（旧会话资源全释放）。
     *  handler：帧回调投递线程（合成器 GL 线程，须有 Looper）。
     *  sr：true 时启用 GL 原生 ESPCN ×2（内容方区 480² → 960²）；权重缺失或
     *  浮点渲染目标不可用则自动退回非 SR 路径。
     *  srWeights：espcn_x2.f32（fp32 LE：w0[576] b0[64] w1[18432] b1[32] w2[1152] b2[4]）。
     *  返回解码端输入 Surface。 */
    public Surface open(int sw, int sh, android.os.Handler handler, boolean sr, byte[] srWeights) {
        listenerHandler = handler;
        close();
        streamW = sw;
        streamH = sh;
        // 内容方区：流高大于宽时取中段方区（CPU convertOverlayFrame 的 topRow/h 规则）
        srcW = sw;
        srcH = sh > sw ? sw : sh;
        int topRow = sh > sw ? (sh - sw) / 2 : 0;
        bandV0 = (float) topRow / sh;
        bandV1 = (float) (topRow + srcH) / sh;

        srActive = false;
        if (sr && srWeights != null && srWeights.length == 81040 && floatRenderable()) {
            try {
                initWeights(srWeights);
                srActive = true;
            } catch (Throwable t) {
                LogI(TAG, "sr init fail: " + t);
                srActive = false;
            }
        }
        contentW = srcW << (srActive ? 1 : 0);
        contentH = srcH << (srActive ? 1 : 0);

        oesTex = genOesTexture();
        st = new SurfaceTexture(oesTex);
        st.setDefaultBufferSize(sw, sh);
        st.setOnFrameAvailableListener(listener, handler);
        inputSurface = new Surface(st);

        contentTex = texRgba(contentW, contentH);
        contentFbo = fboFor(contentTex);
        if (srActive) {
            auxW = srcW; auxH = srcH;
            auxTex = texRgba(auxW, auxH); auxFbo = fboFor(auxTex);
            act1Tex = texHalf(auxW, 16 * auxH); act1Fbo = fboFor(act1Tex);
            act2Tex = texHalf(auxW, 8 * auxH); act2Fbo = fboFor(act2Tex);
        }
        glowW = Math.max(8, contentW >> 2);
        glowH = Math.max(8, contentH >> 2);
        halfW = Math.max(4, contentW >> 4);
        halfH = Math.max(4, contentH >> 4);
        farW = Math.max(2, contentW >> 5);
        farH = Math.max(2, contentH >> 5);
        glowTex = texRgba(glowW, glowH); glowFbo = fboFor(glowTex);
        halfTex = texRgba(halfW, halfH); halfFbo = fboFor(halfTex);
        farTex = texRgba(farW, farH); farFbo = fboFor(farTex);
        encW = contentW; encH = contentH;
        encMidW = Math.max(1, contentW >> 2);
        encMidH = Math.max(1, contentH >> 2);
        enc2W = Math.max(1, contentW >> 4);
        enc2H = Math.max(1, contentH >> 4);
        encTex = texRgba(encW, encH); encFbo = fboFor(encTex);
        encMidTex = texRgba(encMidW, encMidH); encMidFbo = fboFor(encMidTex);
        enc2Tex = texRgba(enc2W, enc2H); enc2Fbo = fboFor(enc2Tex);

        if (pContent == 0) {
            pContent = buildProgram(VERT_SRC, CONTENT_FRAG);
            pGlow = buildProgram(VERT_SRC, GLOW_FRAG);
            pCopy = buildProgram(VERT_SRC, COPY_FRAG);
            pEncode = buildProgram(VERT_SRC, ENCODE_FRAG);
            locContentTex = GLES20.glGetUniformLocation(pContent, "uTex");
            locXf = GLES20.glGetUniformLocation(pContent, "uXf");
            locBand = GLES20.glGetUniformLocation(pContent, "uBand");
            locStream = GLES20.glGetUniformLocation(pContent, "uStream");
            locGlowTex = GLES20.glGetUniformLocation(pGlow, "uTex");
            locAvg = GLES20.glGetUniformLocation(pGlow, "uAvg");
            locCopyTex = GLES20.glGetUniformLocation(pCopy, "uTex");
            locEncTex = GLES20.glGetUniformLocation(pEncode, "uTex");
            quadBuf = ByteBuffer.allocateDirect(quadArr.length * 4)
                .order(ByteOrder.nativeOrder()).asFloatBuffer();
            GLES20.glEnableVertexAttribArray(0);
        }
        if (srActive && pAux == 0) {
            pAux = buildProgram(VERT_SRC, AUX_FRAG);
            pS1 = buildProgram(VERT300_SRC, SR_L1_FRAG);
            pS2 = buildProgram(VERT300_SRC, SR_L2_FRAG);
            pFinal = buildProgram(VERT300_SRC, SR_FINAL_FRAG);
            locAuxOes = GLES20.glGetUniformLocation(pAux, "uTex");
            locXfAux = GLES20.glGetUniformLocation(pAux, "uXf");
            locBandAux = GLES20.glGetUniformLocation(pAux, "uBand");
            locStreamAux = GLES20.glGetUniformLocation(pAux, "uStream");
            locA1Tex = GLES20.glGetUniformLocation(pS2, "uA1");
            locA2Tex = GLES20.glGetUniformLocation(pFinal, "uA2");
            locAuxChroma = GLES20.glGetUniformLocation(pFinal, "uAux");
            locW1 = GLES20.glGetUniformLocation(pS1, "uW");
            locW2 = GLES20.glGetUniformLocation(pS2, "uW");
            locW3 = GLES20.glGetUniformLocation(pFinal, "uW");
        }
        baked = false;
        LogI(TAG, "open stream=" + sw + "x" + sh + " content=" + contentW + "x" + contentH
            + " glow=" + glowW + "x" + glowH + " far=" + farW + "x" + farH
            + " sr=" + (srActive ? "on(gl)" : "off"));
        return inputSurface;
    }

    /** 释放输入路径并清空产物纹理（会话收尾/重开）。保留编译好的 program。 */
    public void close() {
        if (st != null) {
            st.setOnFrameAvailableListener(null, listenerHandler);
            st.release();
            st = null;
        }
        if (inputSurface != null) {
            inputSurface.release();
            inputSurface = null;
        }
        if (oesTex != 0) { GLES20.glDeleteTextures(1, new int[]{oesTex}, 0); oesTex = 0; }
        int[] texs = {contentTex, glowTex, halfTex, farTex, encTex, encMidTex, enc2Tex,
            auxTex, act1Tex, act2Tex, wt1Tex, wt2Tex, wt3Tex};
        int[] fbos = {contentFbo, glowFbo, halfFbo, farFbo, encFbo, encMidFbo, enc2Fbo,
            auxFbo, act1Fbo, act2Fbo};
        GLES20.glDeleteTextures(texs.length, texs, 0);
        GLES20.glDeleteFramebuffers(fbos.length, fbos, 0);
        contentTex = glowTex = halfTex = farTex = encTex = encMidTex = enc2Tex = 0;
        auxTex = act1Tex = act2Tex = wt1Tex = wt2Tex = wt3Tex = 0;
        contentFbo = glowFbo = halfFbo = farFbo = encFbo = encMidFbo = enc2Fbo = 0;
        auxFbo = act1Fbo = act2Fbo = 0;
        streamW = streamH = 0;
        pending = false;
        baked = false;
        avgDirty = false;
        version++;
    }

    public void release() {
        released = true;
        close();
    }

    // ── 烘焙（GL 线程）──────────────────────────────────────

    /** 有待处理解码帧时执行一次烘焙。返回是否烘焙了新帧。 */
    public boolean bakeIfPending() {
        if (released || !pending || st == null) return false;
        pending = false;
        long t0 = android.os.SystemClock.uptimeMillis();
        st.updateTexImage();
        if (xfOverride != null) {
            System.arraycopy(xfOverride, 0, xf, 0, 16);
        } else {
            st.getTransformMatrix(xf);
        }
        // 上一帧编码结果的延迟读回（把 glReadPixels 停顿错峰到本帧绘制之前；
        // 色调缓变，每 4 帧刷新一次）
        boolean hasOpaque = lastOpaque;
        if (avgDirty && (bakedFrames < 2 || ++avgTick % 4 == 0)) {
            hasOpaque = consumeAvg();
            avgDirty = false;
            lastOpaque = hasOpaque;
        }

        if (srActive) {
            // ── SR：YUV→RGB(方区) → GL 原生 ESPCN ×2（conv1 64ch → conv2 32ch →
            //    conv3+PixelShuffle），末 pass 合并色度移植 + 黑键 + 饱和度（960²）──
            GLES20.glDisable(GLES20.GL_BLEND);
            bindFbo(auxFbo, auxW, auxH);
            GLES20.glUseProgram(pAux);
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTex);
            GLES20.glUniform1i(locAuxOes, 0);
            GLES20.glUniformMatrix4fv(locXfAux, 1, false, xf, 0);
            GLES20.glUniform2f(locBandAux, bandV0, bandV1);
            GLES20.glUniform2f(locStreamAux, streamW, streamH);
            drawQuad();
            // conv1+ReLU：16 个 4 通道切片 → act1（切片 p = 通道 4p..4p+3）
            GLES20.glUseProgram(pS1);
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, auxTex);
            GLES20.glUniform1i(locW1, 1);
            GLES20.glActiveTexture(GLES20.GL_TEXTURE1);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, wt1Tex);
            bindFbo(act1Fbo, auxW, auxH);          // 16 切片写进 act1（此前误留 auxFbo）
            for (int p = 0; p < 16; p++) {
                GLES20.glUniform1i(GLES20.glGetUniformLocation(pS1, "uOc0"), p * 4);
                GLES20.glUniform4f(GLES20.glGetUniformLocation(pS1, "uB"),
                    srB0[p * 4], srB0[p * 4 + 1], srB0[p * 4 + 2], srB0[p * 4 + 3]);
                GLES20.glViewport(0, p * auxH, auxW, auxH);
                drawQuad();
            }
            // conv2+ReLU：8 个 4 通道切片 → act2
            GLES20.glUseProgram(pS2);
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, act1Tex);
            GLES20.glUniform1i(locA1Tex, 0);
            GLES20.glUniform1i(locW2, 1);
            GLES20.glActiveTexture(GLES20.GL_TEXTURE1);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, wt2Tex);
            bindFbo(act2Fbo, auxW, auxH);          // 8 切片写进 act2
            for (int p = 0; p < 8; p++) {
                GLES20.glUniform1i(GLES20.glGetUniformLocation(pS2, "uOc0"), p * 4);
                GLES20.glUniform4f(GLES20.glGetUniformLocation(pS2, "uB"),
                    srB1[p * 4], srB1[p * 4 + 1], srB1[p * 4 + 2], srB1[p * 4 + 3]);
                GLES20.glViewport(0, p * auxH, auxW, auxH);
                drawQuad();
            }
            // conv3 + PixelShuffle + 色度移植 + 黑键 + 饱和度 → content（960²）
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, contentFbo);
            GLES20.glViewport(0, 0, contentW, contentH);
            GLES20.glUseProgram(pFinal);
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, act2Tex);
            GLES20.glUniform1i(locA2Tex, 0);
            GLES20.glActiveTexture(GLES20.GL_TEXTURE1);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, auxTex);
            GLES20.glUniform1i(locAuxChroma, 1);
            GLES20.glUniform1i(locW3, 2);
            GLES20.glActiveTexture(GLES20.GL_TEXTURE2);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, wt3Tex);
            drawQuad();
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
        } else {
            // 内容：裁空带 + 镜像 + YUV→RGB + 黑键 + 饱和度，预乘输出
            bindFbo(contentFbo, contentW, contentH);
            GLES20.glUseProgram(pContent);
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTex);
            GLES20.glUniform1i(locContentTex, 0);
            GLES20.glUniformMatrix4fv(locXf, 1, false, xf, 0);
            GLES20.glUniform2f(locBand, bandV0, bandV1);
            GLES20.glUniform2f(locStream, streamW, streamH);
            drawQuad();
        }
        if (debugProbe) {
            int err = GLES20.glGetError();
            StringBuilder sb = new StringBuilder("probe xf=[");
            for (float v : xf) sb.append(String.format("%.2f,", v));
            sb.append("] err=0x").append(Integer.toHexString(err));
            ByteBuffer pb = ByteBuffer.allocateDirect(4 * 5).order(ByteOrder.nativeOrder());
            GLES20.glReadPixels(contentW / 2, contentH / 2 - 2, 5, 1, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, pb);
            sb.append(" px:");
            for (int i = 0; i < 5; i++) {
                sb.append(String.format(" %02x%02x%02x%02x", pb.get(i * 4) & 0xFF,
                    pb.get(i * 4 + 1) & 0xFF, pb.get(i * 4 + 2) & 0xFF, pb.get(i * 4 + 3) & 0xFF));
            }
            LogI(TAG, sb.toString());
        }
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
        // tint：全分辨率遮罩编码 → 两级面积平均 → 读回**延迟到下一帧开头**（把
        // glReadPixels 的管线停顿挪出本帧关键路径），每 4 帧刷新一次（色调缓变）
        bindFbo(encFbo, encW, encH);
        GLES20.glUseProgram(pEncode);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, contentTex);
        GLES20.glUniform1i(locEncTex, 0);
        drawQuad();
        bindFbo(encMidFbo, encMidW, encMidH);
        GLES20.glUseProgram(pCopy);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, encTex);
        GLES20.glUniform1i(locCopyTex, 0);
        drawQuad();
        bindFbo(enc2Fbo, enc2W, enc2H);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, encMidTex);
        drawQuad();
        avgDirty = true;

        // 剪影 → 金字塔低通两轮 → 远距层（尺寸与 CPU 烘焙一致）
        bindFbo(glowFbo, glowW, glowH);
        GLES20.glUseProgram(pGlow);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, contentTex);
        GLES20.glUniform1i(locGlowTex, 0);
        GLES20.glUniform3f(locAvg, avg[0], avg[1], avg[2]);
        drawQuad();
        GLES20.glUseProgram(pCopy);
        GLES20.glUniform1i(locCopyTex, 0);
        for (int round = 0; round < 2; round++) {
            bindFbo(halfFbo, halfW, halfH);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, glowTex);
            drawQuad();
            bindFbo(glowFbo, glowW, glowH);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, halfTex);
            drawQuad();
        }
        if (!hasOpaque) {
            GLES20.glClearColor(0f, 0f, 0f, 0f);
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);   // glow 已清；far 同步清空
        }
        bindFbo(farFbo, farW, farH);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, glowTex);
        drawQuad();

        baked = true;
        version++;
        bakedFrames++;
        if (bakedFrames % 150L == 1L) {
            LogI(TAG, "bake #" + bakedFrames + " "
                + (android.os.SystemClock.uptimeMillis() - t0) + "ms avg=("
                + (int) (avg[0] * 255) + "," + (int) (avg[1] * 255) + "," + (int) (avg[2] * 255) + ")");
        }
        return true;
    }

    /** 面积加权平均：enc 链存的是 (unprem·m, m)，两者同一滤波 → Σrgb_m/Σm 恰为
     *  CPU 语义的不透明平均色（不能再乘 m——rgb 已含遮罩，二次预乘会压暗色调）。 */
    private boolean consumeAvg() {
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, enc2Fbo);
        readBuf.rewind();
        GLES20.glReadPixels(0, 0, enc2W, enc2H, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, readBuf);
        long sumR = 0, sumG = 0, sumB = 0, sumM = 0;
        for (int i = 0; i < enc2W * enc2H; i++) {
            sumR += readBuf.get(i * 4) & 0xFF;
            sumG += readBuf.get(i * 4 + 1) & 0xFF;
            sumB += readBuf.get(i * 4 + 2) & 0xFF;
            sumM += readBuf.get(i * 4 + 3) & 0xFF;
        }
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
        if (sumM < 8) {
            avg[0] = avg[1] = avg[2] = 0f;
            return false;
        }
        avg[0] = sumR / (float) sumM;
        avg[1] = sumG / (float) sumM;
        avg[2] = sumB / (float) sumM;
        return true;
    }

    // ── 产物访问（任意线程）────────────────────────────────

    public boolean hasBaked() { return baked; }
    public boolean peekPending() { return pending; }
    public long version() { return version; }
    public int contentTex() { return contentTex; }
    public int glowTex() { return glowTex; }
    public int bloomTex() { return farTex; }
    public int contentW() { return contentW; }
    public int contentH() { return contentH; }
    public Surface inputSurface() { return inputSurface; }

    /** 仅离线测试：最近一次烘焙的平均色。 */
    public float[] avgForTest() { return avg; }

    /** 仅离线测试：act1 FBO（切片 0 = conv1 通道 0..3），数值探针用。 */
    public int debugAct1Fbo() { return act1Fbo; }
    public int debugAuxW() { return auxW; }
    public int debugAuxH() { return auxH; }

    /** 仅离线测试注入变换矩阵（见字段注释）。 */
    public void setXfOverride(float[] m) { xfOverride = m; }

    /** 产物转储（GL 线程；真机 A/B 取证）。 */
    public boolean dump(String prefix) {
        if (!baked) return false;
        dumpTex(contentTex, contentW, contentH, prefix + "_content.png");
        dumpTex(glowTex, glowW, glowH, prefix + "_glow.png");
        dumpTex(farTex, farW, farH, prefix + "_far.png");
        return true;
    }

    private void dumpTex(int tex, int w, int h, String name) {
        try {
            int[] oldFbo = new int[1];
            GLES20.glGetIntegerv(GLES20.GL_FRAMEBUFFER_BINDING, oldFbo, 0);
            int tmpTex = texRgba(w, h);
            int tmpFbo = fboFor(tmpTex);
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, tmpFbo);
            GLES20.glUseProgram(pCopy);
            GLES20.glUniform1i(locCopyTex, 0);
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, tex);
            drawQuad();
            Bitmap bm = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
            ByteBuffer bb = ByteBuffer.allocateDirect(w * h * 4).order(ByteOrder.nativeOrder());
            GLES20.glReadPixels(0, 0, w, h, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, bb);
            bm.copyPixelsFromBuffer(bb);
            java.io.FileOutputStream fos = new java.io.FileOutputStream(name);
            bm.compress(Bitmap.CompressFormat.PNG, 100, fos);
            fos.close();
            bm.recycle();
            GLES20.glDeleteTextures(1, new int[]{tmpTex}, 0);
            GLES20.glDeleteFramebuffers(1, new int[]{tmpFbo}, 0);
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, oldFbo[0]);
            LogI(TAG, "dump " + name);
        } catch (Throwable t) {
            LogI(TAG, "dump fail " + name + ": " + t);
        }
    }

    // ── GL 基础 ─────────────────────────────────────────────

    private float bandV0, bandV1;
    private int halfW, halfH;
    private int locContentTex, locXf, locBand, locStream;
    private int locGlowTex, locAvg, locCopyTex, locEncTex;
    private FloatBuffer quadBuf;
    private final float[] quadArr = new float[6 * 4];
    private final float[] xf = new float[16];
    private int pContent, pGlow, pCopy, pEncode;

    private void bindFbo(int fbo, int w, int h) {
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbo);
        GLES20.glViewport(0, 0, w, h);
    }

    /** 全屏双三角：clip [-1,1]，uv v=0 对应 clip y=-1（FBO 内存行 0）。
     *  FBO→FBO 采样为恒等映射，产物纹理行 0 = 图像顶行（GLUtils 约定）。 */
    private void drawQuad() {
        float[] a = quadArr;
        int k = 0;
        a[k++] = -1f; a[k++] = -1f; a[k++] = 0f; a[k++] = 0f;
        a[k++] = 1f; a[k++] = -1f; a[k++] = 1f; a[k++] = 0f;
        a[k++] = 1f; a[k++] = 1f; a[k++] = 1f; a[k++] = 1f;
        a[k++] = -1f; a[k++] = -1f; a[k++] = 0f; a[k++] = 0f;
        a[k++] = 1f; a[k++] = 1f; a[k++] = 1f; a[k++] = 1f;
        a[k++] = -1f; a[k++] = 1f; a[k++] = 0f; a[k++] = 1f;
        quadBuf.position(0);
        quadBuf.put(a);
        quadBuf.position(0);
        GLES20.glVertexAttribPointer(0, 4, GLES20.GL_FLOAT, false, 0, quadBuf);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, 6);
    }

    private static int texRgba(int w, int h) {
        int[] t = new int[1];
        GLES20.glGenTextures(1, t, 0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, t[0]);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, w, h, 0,
            GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null);
        int[] f = new int[1];
        GLES20.glGenFramebuffers(1, f, 0);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, f[0]);
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
            GLES20.GL_TEXTURE_2D, t[0], 0);
        if (GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER) != GLES20.GL_FRAMEBUFFER_COMPLETE) {
            throw new IllegalStateException("fbo incomplete " + w + "x" + h);
        }
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
        return t[0];
    }

    /** RGBA16F 纹理 + FBO（SR 中间激活；需 EXT_color_buffer_float）。 */
    private static int texHalf(int w, int h) {
        int[] t = new int[1];
        GLES30.glGenTextures(1, t, 0);
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, t[0]);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_NEAREST);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_NEAREST);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE);
        GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA16F, w, h, 0,
            GLES30.GL_RGBA, GLES30.GL_HALF_FLOAT, null);
        int[] f = new int[1];
        GLES30.glGenFramebuffers(1, f, 0);
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, f[0]);
        GLES30.glFramebufferTexture2D(GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0,
            GLES30.GL_TEXTURE_2D, t[0], 0);
        if (GLES30.glCheckFramebufferStatus(GLES30.GL_FRAMEBUFFER) != GLES30.GL_FRAMEBUFFER_COMPLETE) {
            throw new IllegalStateException("rgba16f fbo incomplete " + w + "x" + h);
        }
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0);
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0);
        return t[0];
    }

    /** R32F 权重纹理（texelFetch 采样）。 */
    private static int texR32f(FloatBuffer data, int w, int h) {
        int[] t = new int[1];
        GLES30.glGenTextures(1, t, 0);
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, t[0]);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_NEAREST);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_NEAREST);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE);
        data.position(0);
        GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_R32F, w, h, 0,
            GLES30.GL_RED, GLES30.GL_FLOAT, data);
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0);
        return t[0];
    }

    /** SR 权重初始化：f32 字节流（fp32 LE）→ 3 张 R32F 纹理 + bias 数组。
     *  布局：w0[64][9] b0[64] w1[32][64][9] b1[32] w2[4][32][9] b2[4]（oc 主序）。 */
    private void initWeights(byte[] raw) {
        FloatBuffer fb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        float[] all = new float[raw.length / 4];
        fb.get(all);
        int o = 0;
        float[] w0 = java.util.Arrays.copyOfRange(all, o, o + 576); o += 576;
        srB0 = java.util.Arrays.copyOfRange(all, o, o + 64); o += 64;
        float[] w1 = java.util.Arrays.copyOfRange(all, o, o + 18432); o += 18432;
        srB1 = java.util.Arrays.copyOfRange(all, o, o + 32); o += 32;
        float[] w2 = java.util.Arrays.copyOfRange(all, o, o + 1152); o += 1152;
        float[] b2 = java.util.Arrays.copyOfRange(all, o, o + 4);
        wt1Tex = texR32f(toBuffer(w0), 9, 64);
        wt2Tex = texR32f(toBuffer(w1), 9, 64 * 32);
        // WT3：行 (ic + 32*oc) 存 w2，末行 128 存 b2（x = oc）
        float[] w3 = new float[9 * 129];
        for (int oc = 0; oc < 4; oc++) {
            for (int ic = 0; ic < 32; ic++)
                for (int k = 0; k < 9; k++)
                    w3[(ic + 32 * oc) * 9 + k] = w2[oc * 288 + ic * 9 + k];
            w3[128 * 9 + oc] = b2[oc];
        }
        wt3Tex = texR32f(toBuffer(w3), 9, 129);
    }

    private static FloatBuffer toBuffer(float[] a) {
        return ByteBuffer.allocateDirect(a.length * 4).order(ByteOrder.nativeOrder())
            .asFloatBuffer().put(a);
    }

    /** SR 可用性：浮点渲染目标（RGBA16F FBO）支持检查。 */
    private static boolean floatRenderable() {
        String ext = GLES30.glGetString(GLES30.GL_EXTENSIONS);
        return ext != null && (ext.contains("EXT_color_buffer_float")
            || ext.contains("EXT_color_buffer_half_float"));
    }

    private static int fboFor(int tex) {
        int[] f = new int[1];
        GLES20.glGenFramebuffers(1, f, 0);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, f[0]);
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
            GLES20.GL_TEXTURE_2D, tex, 0);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
        return f[0];
    }

    private static int genOesTexture() {
        int[] t = new int[1];
        GLES20.glGenTextures(1, t, 0);
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, t[0]);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
        return t[0];
    }

    private static int buildProgram(String vs, String fs) {
        int v = loadShader(GLES20.GL_VERTEX_SHADER, vs);
        int f = loadShader(GLES20.GL_FRAGMENT_SHADER, fs);
        int program = GLES20.glCreateProgram();
        GLES20.glAttachShader(program, v);
        GLES20.glAttachShader(program, f);
        GLES20.glBindAttribLocation(program, 0, "aPosUv");
        GLES20.glLinkProgram(program);
        int[] status = new int[1];
        GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, status, 0);
        if (status[0] != GLES20.GL_TRUE) {
            throw new RuntimeException("program link error: " + GLES20.glGetProgramInfoLog(program));
        }
        return program;
    }

    private static int loadShader(int type, String src) {
        int shader = GLES20.glCreateShader(type);
        GLES20.glShaderSource(shader, src);
        GLES20.glCompileShader(shader);
        int[] status = new int[1];
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, status, 0);
        if (status[0] != GLES20.GL_TRUE) {
            throw new RuntimeException("shader compile error: " + GLES20.glGetShaderInfoLog(shader)
                + "\n" + src);
        }
        return shader;
    }

    private static void LogI(String tag, String msg) {
        System.out.println("[" + tag + "] " + msg);
        android.util.Log.i(tag, msg);
    }

    // ── shaders ─────────────────────────────────────────────

    /** 顶点（GLSL 100，全项目统一）：xy = clip，zw = uv（v=0 为 FBO 首行）。 */
    private static final String VERT_SRC =
        "attribute vec4 aPosUv;\n" +
        "varying vec2 vUv;\n" +
        "void main() {\n" +
        "    vUv = aPosUv.zw;\n" +
        "    gl_Position = vec4(aPosUv.xy, 0.0, 1.0);\n" +
        "}\n";

    /** SR 顶点（GLSL 300 es：与 300 es 片元配对）。 */
    private static final String VERT300_SRC =
        "#version 300 es\n" +
        "in vec4 aPosUv;\n" +
        "out vec2 vUv;\n" +
        "void main() {\n" +
        "    vUv = aPosUv.zw;\n" +
        "    gl_Position = vec4(aPosUv.xy, 0.0, 1.0);\n" +
        "}\n";

    /** 内容 pass（非 SR）：流坐标裁空带 + 镜像 + OES YUV→RGB + 黑键 + 饱和度，预乘输出。 */
    private static final String CONTENT_FRAG =
        "#extension GL_OES_EGL_image_external : require\n" +
        "precision highp float;\n" +
        "varying highp vec2 vUv;\n" +
        "uniform samplerExternalOES uTex;\n" +
        "uniform mat4 uXf;\n" +
        "uniform vec2 uBand;\n" +
        "uniform vec2 uStream;\n" +
        "void main() {\n" +
        "    vec2 f = vUv;\n" +
        "    f.x = 1.0 - (floor(f.x * uStream.x) + 0.5) / uStream.x;\n" +
        "    f.y = 1.0 - mix(uBand.x, uBand.y, f.y);\n" +
        "    vec2 suv = (uXf * vec4(f, 0.0, 1.0)).xy;\n" +
        "    vec3 rgb = texture2D(uTex, suv).rgb;\n" +
        "    float luma = dot(rgb, vec3(0.299, 0.587, 0.114));\n" +
        "    float a = pow(clamp((luma * 255.0 - " + KEY_LO_F + ") / " + KEY_SPAN_F + ", 0.0, 1.0), 1.2);\n" +
        "    vec3 sat = clamp(luma + (rgb - luma) * (28.0 / 25.0), 0.0, 1.0);\n" +
        "    gl_FragColor = vec4(sat * a, a);\n" +
        "}\n";

    /** 剪影：内容 alpha×8 饱和 + 统一平均色，预乘输出（CPU ColorMatrix 同义）。 */
    private static final String GLOW_FRAG =
        "precision mediump float;\n" +
        "varying vec2 vUv;\n" +
        "uniform sampler2D uTex;\n" +
        "uniform vec3 uAvg;\n" +
        "void main() {\n" +
        "    float a = texture2D(uTex, vUv).a;\n" +
        "    float a8 = min(a * 8.0, 1.0);\n" +
        "    gl_FragColor = vec4(uAvg * a8, a8);\n" +
        "}\n";

    /** 纯双线性搬运（金字塔降/升采样 + 转储中转）。 */
    private static final String COPY_FRAG =
        "precision mediump float;\n" +
        "varying vec2 vUv;\n" +
        "uniform sampler2D uTex;\n" +
        "void main() {\n" +
        "    gl_FragColor = texture2D(uTex, vUv);\n" +
        "}\n";

    /** tint 遮罩编码：a≥64/255 不透明像素记录非预乘色，其余清零（alpha 供降采样计数）。 */
    private static final String ENCODE_FRAG =
        "precision mediump float;\n" +
        "varying vec2 vUv;\n" +
        "uniform sampler2D uTex;\n" +
        "void main() {\n" +
        "    vec4 s = texture2D(uTex, vUv);\n" +
        "    float m = step(64.0 / 255.0, s.a);\n" +
        "    vec3 rgb = s.a > 0.004 ? s.rgb / s.a : vec3(0.0);\n" +
        "    gl_FragColor = vec4(rgb * m, m);\n" +
        "}\n";

    /** SR 输入 pass：YUV→RGB（方区、镜像，无键控；SR 亮度源 + 末 pass 色度源）。 */
    private static final String AUX_FRAG =
        "#extension GL_OES_EGL_image_external : require\n" +
        "precision highp float;\n" +
        "varying highp vec2 vUv;\n" +
        "uniform samplerExternalOES uTex;\n" +
        "uniform mat4 uXf;\n" +
        "uniform vec2 uBand;\n" +
        "uniform vec2 uStream;\n" +
        "void main() {\n" +
        "    vec2 f = vUv;\n" +
        "    f.x = 1.0 - (floor(f.x * uStream.x) + 0.5) / uStream.x;\n" +
        "    f.y = 1.0 - mix(uBand.x, uBand.y, f.y);\n" +
        "    vec2 suv = (uXf * vec4(f, 0.0, 1.0)).xy;\n" +
        "    gl_FragColor = vec4(texture2D(uTex, suv).rgb, 1.0);\n" +
        "}\n";

    /** ESPCN conv1+ReLU（GLSL 300 es）：输入 aux 亮度（inline 计算），输出 4ch 切片。 */
    private static final String SR_L1_FRAG =
        "#version 300 es\n" +
        "precision highp float;\n" +
        "uniform highp sampler2D uAux;\n" +
        "uniform highp sampler2D uW;\n" +
        "uniform int uOc0;\n" +
        "uniform vec4 uB;\n" +
        "out vec4 o;\n" +
        "void main() {\n" +
        "    int sliceH = textureSize(uAux, 0).y;\n" +
        "    ivec2 sz = textureSize(uAux, 0);\n" +
        "    ivec2 P = ivec2(int(gl_FragCoord.x), int(gl_FragCoord.y) - (uOc0 / 4) * sliceH);\n" +
        "    vec4 s = uB;\n" +
        "    for (int k = 0; k < 9; k++) {\n" +
        "        ivec2 d = ivec2(k % 3 - 1, k / 3 - 1);\n" +
        "        ivec2 q = clamp(P + d, ivec2(0), sz - 1);\n" +
        "        float l = dot(texelFetch(uAux, q, 0).rgb, vec3(0.299, 0.587, 0.114));\n" +
        "        s[0] += texelFetch(uW, ivec2(k, uOc0), 0).r * l;\n" +
        "        s[1] += texelFetch(uW, ivec2(k, uOc0 + 1), 0).r * l;\n" +
        "        s[2] += texelFetch(uW, ivec2(k, uOc0 + 2), 0).r * l;\n" +
        "        s[3] += texelFetch(uW, ivec2(k, uOc0 + 3), 0).r * l;\n" +
        "    }\n" +
        "    o = max(s, vec4(0.0));\n" +
        "}\n";

    /** ESPCN conv2+ReLU：输入 act1（64ch 切片纹理），输出 4ch 切片。 */
    private static final String SR_L2_FRAG =
        "#version 300 es\n" +
        "precision highp float;\n" +
        "uniform highp sampler2D uA1;\n" +
        "uniform highp sampler2D uW;\n" +
        "uniform int uOc0;\n" +
        "uniform vec4 uB;\n" +
        "out vec4 o;\n" +
        "void main() {\n" +
        "    ivec2 sz = textureSize(uA1, 0);\n" +
        "    int sliceH = sz.y / 16;\n" +
        "    ivec2 P = ivec2(int(gl_FragCoord.x), int(gl_FragCoord.y) - (uOc0 / 4) * sliceH);\n" +
        "    vec4 s = uB;\n" +
        "    for (int sl = 0; sl < 16; sl++) {\n" +
        "        int sy = sl * sliceH;\n" +
        "        vec4 a[9];\n" +
        "        for (int k = 0; k < 9; k++) {\n" +
        "            ivec2 d = ivec2(k % 3 - 1, k / 3 - 1);\n" +
        "            ivec2 q = ivec2(clamp(P.x + d.x, 0, sz.x - 1), clamp(P.y + d.y, 0, sliceH - 1));\n" +
        "            a[k] = texelFetch(uA1, ivec2(q.x, sy + q.y), 0);\n" +
        "        }\n" +
        "        for (int c = 0; c < 4; c++) {\n" +
        "            int ic = sl * 4 + c;\n" +
        "            int row = ic + 64 * (uOc0 + c);\n" +
        "            float acc = 0.0;\n" +
        "            for (int k = 0; k < 9; k++)\n" +
        "                acc += texelFetch(uW, ivec2(k, row), 0).r * a[k][c];\n" +
        "            s[c] += acc;\n" +
        "        }\n" +
        "    }\n" +
        "    o = max(s, vec4(0.0));\n" +
        "}\n";

    /** ESPCN conv3 + PixelShuffle×2 + 色度移植 + 黑键 + 饱和度 → content（960²）。
     *  PixelShuffle：out(2h+py, 2w+px) = conv3 通道 (px + 2*py) @ (h,w)。 */
    private static final String SR_FINAL_FRAG =
        "#version 300 es\n" +
        "precision highp float;\n" +
        "uniform highp sampler2D uA2;\n" +
        "uniform mediump sampler2D uAux;\n" +
        "uniform highp sampler2D uW;\n" +
        "in vec2 vUv;\n" +
        "out vec4 o;\n" +
        "void main() {\n" +
        "    ivec2 P = ivec2(gl_FragCoord.xy);\n" +
        "    int px = P.x & 1, py = P.y & 1;\n" +
        "    int oc = px + 2 * py;\n" +
        "    int hx = P.x >> 1, hy = P.y >> 1;\n" +
        "    ivec2 sz = textureSize(uA2, 0);\n" +
        "    int sliceH = sz.y / 8;\n" +
        "    float s = texelFetch(uW, ivec2(oc, 128), 0).r;\n" +
        "    for (int sl = 0; sl < 8; sl++) {\n" +
        "        int sy = sl * sliceH;\n" +
        "        vec4 a0 = texelFetch(uA2, ivec2(clamp(hx - 1, 0, sz.x - 1), sy + clamp(hy - 1, 0, sliceH - 1)), 0);\n" +
        "        vec4 a1 = texelFetch(uA2, ivec2(clamp(hx, 0, sz.x - 1), sy + clamp(hy - 1, 0, sliceH - 1)), 0);\n" +
        "        vec4 a2 = texelFetch(uA2, ivec2(clamp(hx + 1, 0, sz.x - 1), sy + clamp(hy - 1, 0, sliceH - 1)), 0);\n" +
        "        vec4 a3 = texelFetch(uA2, ivec2(clamp(hx - 1, 0, sz.x - 1), sy + clamp(hy, 0, sliceH - 1)), 0);\n" +
        "        vec4 a4 = texelFetch(uA2, ivec2(hx, hy), 0);\n" +
        "        vec4 a5 = texelFetch(uA2, ivec2(clamp(hx + 1, 0, sz.x - 1), sy + clamp(hy, 0, sliceH - 1)), 0);\n" +
        "        vec4 a6 = texelFetch(uA2, ivec2(clamp(hx - 1, 0, sz.x - 1), sy + clamp(hy + 1, 0, sliceH - 1)), 0);\n" +
        "        vec4 a7 = texelFetch(uA2, ivec2(clamp(hx, 0, sz.x - 1), sy + clamp(hy + 1, 0, sliceH - 1)), 0);\n" +
        "        vec4 a8 = texelFetch(uA2, ivec2(clamp(hx + 1, 0, sz.x - 1), sy + clamp(hy + 1, 0, sliceH - 1)), 0);\n" +
        "        int ic = sl * 4;\n" +
        "        float w; vec4 a;\n" +
        "        w = texelFetch(uW, ivec2(0, ic + 32 * oc), 0).r; a = a0; s += w * a[0]; s += w * a[1]; s += w * a[2]; s += w * a[3];\n" +
        "        w = texelFetch(uW, ivec2(1, ic + 32 * oc), 0).r; a = a1; s += w * a[0]; s += w * a[1]; s += w * a[2]; s += w * a[3];\n" +
        "        w = texelFetch(uW, ivec2(2, ic + 32 * oc), 0).r; a = a2; s += w * a[0]; s += w * a[1]; s += w * a[2]; s += w * a[3];\n" +
        "        w = texelFetch(uW, ivec2(3, ic + 32 * oc), 0).r; a = a3; s += w * a[0]; s += w * a[1]; s += w * a[2]; s += w * a[3];\n" +
        "        w = texelFetch(uW, ivec2(4, ic + 32 * oc), 0).r; a = a4; s += w * a[0]; s += w * a[1]; s += w * a[2]; s += w * a[3];\n" +
        "        w = texelFetch(uW, ivec2(5, ic + 32 * oc), 0).r; a = a5; s += w * a[0]; s += w * a[1]; s += w * a[2]; s += w * a[3];\n" +
        "        w = texelFetch(uW, ivec2(6, ic + 32 * oc), 0).r; a = a6; s += w * a[0]; s += w * a[1]; s += w * a[2]; s += w * a[3];\n" +
        "        w = texelFetch(uW, ivec2(7, ic + 32 * oc), 0).r; a = a7; s += w * a[0]; s += w * a[1]; s += w * a[2]; s += w * a[3];\n" +
        "        w = texelFetch(uW, ivec2(8, ic + 32 * oc), 0).r; a = a8; s += w * a[0]; s += w * a[1]; s += w * a[2]; s += w * a[3];\n" +
        "    }\n" +
        "    float y = clamp(s, 0.0, 1.0);\n" +
        "    vec2 uvc = (vec2(hx, hy) + 0.5) / vec2(textureSize(uAux, 0));\n" +
        "    vec3 rgbLow = texture(uAux, uvc).rgb;\n" +
        "    float yLow = dot(rgbLow, vec3(0.299, 0.587, 0.114));\n" +
        "    vec3 rgb = clamp(y + (rgbLow - yLow), 0.0, 1.0);\n" +
        "    float luma = dot(rgb, vec3(0.299, 0.587, 0.114));\n" +
        "    float a = pow(clamp((luma * 255.0 - " + KEY_LO_F + ") / " + KEY_SPAN_F + ", 0.0, 1.0), 1.2);\n" +
        "    vec3 sat = clamp(luma + (rgb - luma) * (28.0 / 25.0), 0.0, 1.0);\n" +
        "    o = vec4(sat * a, a);\n" +
        "}\n";
}
