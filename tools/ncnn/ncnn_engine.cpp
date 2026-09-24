// ncnn_engine.cpp — JNI shim for NCNN(-vulkan) det_10g + 1k3d68，GPU 上 fp32 激活存储
// （Mali 上 fp16 存储致 SCRFD 逐帧闪跳、fp16 权重致 landmark 漂移，真机+ht §5.13b 实测）。
// 除原生张量注入(nativeRun)外，还提供整条「det→landmark→Procrustes→pose」链的
// 确定性 C++ 移植 nativePose()：一次调用返回 R 的行(r1,r2,r3；人脸轴取 R 的列) + 世界位(cm)。
#ifdef __APPLE__
// host 离线测试构建（jar harness）：桩掉 android log / system property，其余逻辑与设备一致。
#include <jni.h>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
enum { ANDROID_LOG_INFO = 4, ANDROID_LOG_WARN = 5 };
static inline int __android_log_print(int, const char* tag, const char* fmt, ...)
{
    if (getenv("GSCP_HOST_LOG")) { va_list ap; va_start(ap, fmt); fprintf(stderr, "[%s] ", tag);
        vfprintf(stderr, fmt, ap); fputc('\n', stderr); va_end(ap); }
    return 0;
}
static inline int __system_property_get(const char*, char* v) { v[0] = 0; return 0; }
#else
#include <jni.h>
#include <android/log.h>
#include <sys/system_properties.h>
#endif
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <ctime>
#include <thread>
#include "net.h"
#include "gpu.h"

#define TAG "gscp-ncnn"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)

// host 构建链接的 libncnn.a 是 CPU-only（NCNN_VULKAN=OFF），无 get_gpu_count 符号。
#ifdef __APPLE__
static inline int nc_gpu_count() { return 0; }
#else
static inline int nc_gpu_count() { return ncnn::get_gpu_count(); }
#endif

static ncnn::Net* load_net(const std::string& param, const std::string& bin, bool gpu, bool allow_fp16) {
    (void)nc_gpu_count();
    ncnn::Net* net = new ncnn::Net();
    ncnn::Option o = net->opt;
    o.num_threads = (int)std::thread::hardware_concurrency();
    if (o.num_threads > 4) o.num_threads = 4;
    char omp[8]={0}; __system_property_get("debug.gscp.omp", omp);
    int ompn = omp[0] ? (int)(omp[0]-'0') : 0;
    if (ompn >= 1) o.num_threads = ompn;
    // ht 项目 insight_pipe 的 Vulkan 固化配置：lightmode=false（不激进释放/复用中间内存，
    // 否则 det 多输出在 Vulkan 下 extract 会拿到被复用过的空形 [C,1,1] stub）。
    o.lightmode = false;
    o.use_packing_layout = true;
    char pb[64]={0}; __system_property_get("debug.gscp.novulkan", pb);
    // 引擎选择：默认（未设）全 GPU，fp16 与否由各调用方 allow_fp16 决定（当前 det/lmk 均 fp32，
    // 见 NcnnEngine.initDet 注释——Mali 上 fp16 激活存储致 SCRFD 逐帧闪跳，真机实测）。
    // debug.gscp.novulkan 1=强制全 CPU，2=全 GPU fp32，3=fp16 存储+fp32 算术（对照实验档）。
    int mode = pb[0] ? pb[0] - '0' : -1;
    if (mode == 1) {
        gpu = false;               // 全 CPU
    } else {                        // 默认/-1、0、2、3 → 全 GPU
        gpu = true;
    }
    if (gpu && nc_gpu_count() > 0) {
        o.use_vulkan_compute = true;
        // 两个网各自独立开 fp16，便于真机 A/B：debug.gscp.f16.det / debug.gscp.f16.lm = 0/1
        // 覆盖调用方的 allow_fp16（未设 → 沿用默认，当前两网都走 fp32）。Mali 上 fp32 算术是
        // det 160ms/lm 58ms 的主因（CPU 口径分别 386/143ms），fp16 才有带宽与算力收益。
        char f16s[8] = {0};
        __system_property_get(param.find("det_") != std::string::npos
                                  ? "debug.gscp.f16.det" : "debug.gscp.f16.lm", f16s);
        const bool wantFp16 = f16s[0] == '0' ? false : f16s[0] == '1' ? true : allow_fp16;
        const bool stor = wantFp16 && mode != 2;
        o.use_fp16_packed = stor;
        o.use_fp16_storage = stor;
        o.use_fp16_arithmetic = stor && mode != 3;
        o.use_int8_storage = false;
        o.use_int8_arithmetic = false;
    } else {
        o.use_vulkan_compute = false;
        o.use_fp16_packed = false;
        o.use_fp16_storage = false;
        o.use_fp16_arithmetic = false;
    }
    net->opt = o;
    int pr = net->load_param(param.c_str());
    int mb = net->load_model(bin.c_str());
    if (pr != 0 || mb != 0) {
        LOGW("load fail pr=%d mb=%d %s", pr, mb, param.c_str());
        delete net; net = nullptr;
        return nullptr;
    }
    LOGI("loaded %s gpu=%d vulkanAvail=%d fp16Arith=%d", param.c_str(),
         (int)o.use_vulkan_compute, (int)(nc_gpu_count() > 0), (int)o.use_fp16_arithmetic);
    return net;
}

// ── nativePose 分阶段累计耗时（ms）────────────────────────────────────────────
// 0=det 前处理 1=det 前向 2=解码+NMS+选脸 3=lm 裁剪 4=lm 前向 5=解算+Procrustes+定位
// 每段各自的样本数单独记：检不出脸的帧也真的付了 det 前向的时间，却在解算前 break，
// 若用「累计时间 / 成功帧数」会把 miss 帧摊进成功帧的均值里（时间被高估）。
static double g_prof[6] = {0, 0, 0, 0, 0, 0};
static long g_pc[6] = {0, 0, 0, 0, 0, 0};

static inline double nc_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}
// 每段独立计数：某段之后的时间只在跑到下一段的帧里出现。
#define NCP(mark) do { double _n = nc_ms(); g_prof[mark] += _n - _t; g_pc[mark]++; _t = _n; } while (0)

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_gscp_desktop_NcnnEngine_nativeInit(JNIEnv* env, jobject,
                                            jstring param, jstring bin, jboolean useGpu,
                                            jboolean allowFp16) {
    const char* p = env->GetStringUTFChars(param, nullptr);
    const char* b = env->GetStringUTFChars(bin, nullptr);
    ncnn::Net* net = p && b ? load_net(p, b, useGpu == JNI_TRUE, allowFp16 == JNI_TRUE) : nullptr;
    if (p) env->ReleaseStringUTFChars(param, p);
    if (b) env->ReleaseStringUTFChars(bin, b);
    return (jlong)net;
}

JNIEXPORT void JNICALL
Java_com_gscp_desktop_NcnnEngine_nativeRelease(JNIEnv*, jobject, jlong handle) {
    delete (ncnn::Net*)handle;
}

JNIEXPORT jfloatArray JNICALL
Java_com_gscp_desktop_NcnnEngine_nativeRun(JNIEnv* env, jobject, jlong handle,
                                          jstring inputName, jfloatArray data,
                                          jobjectArray names, jintArray dimsOut) {
    ncnn::Net* net = (ncnn::Net*)handle;
    if (!net) return nullptr;
    const char* in = inputName ? env->GetStringUTFChars(inputName, nullptr) : nullptr;
    jsize ndata = env->GetArrayLength(data);
    jfloat* dptr = env->GetFloatArrayElements(data, nullptr);
    jsize nNames = names ? env->GetArrayLength(names) : 0;
    if (!in || ndata == 0 || nNames == 0) {
        if (in) env->ReleaseStringUTFChars(inputName, in);
        if (dptr) env->ReleaseFloatArrayElements(data, dptr, JNI_ABORT);
        return nullptr;
    }
    int C = 3;
    int plane = ndata / C;
    int side = (int)std::sqrt((double)plane);
    ncnn::Mat x(side, side, C);
    size_t ps = (size_t)(side * side);
    for (int c = 0; c < C; c++)
        memcpy((float*)x.channel(c), dptr + (size_t)c * ps, ps * sizeof(float));
    std::string inName = in ? in : "";
    env->ReleaseStringUTFChars(inputName, in);

    ncnn::Extractor ex = net->create_extractor();
    int ri = ex.input(inName.c_str(), x);
    env->ReleaseFloatArrayElements(data, dptr, JNI_ABORT);
    if (ri != 0) { LOGW("extractor.input fail %d", ri); return nullptr; }

    std::vector<float> out;
    std::vector<int> dims((size_t)nNames, 0);
    char d1[8]={0}; __system_property_get("debug.gscp.detone", d1);
    bool detone = (d1[0]=='1') && nNames > 1;
    for (jsize i = 0; i < nNames; i++) {
        jstring js = (jstring)env->GetObjectArrayElement(names, i);
        const char* nm = js ? env->GetStringUTFChars(js, nullptr) : nullptr;
        ncnn::Mat m;
        if (detone) {
            ncnn::Extractor ex2 = net->create_extractor();
            ex2.input(inName.c_str(), x);
            int r2 = ex2.extract(nm, m);
            LOGI("detone[%d]%s rc=%d tot=%d w=%d h=%d c=%d es=%d", i, nm?nm:"?",
                 r2, (int)m.total(), (int)m.w, (int)m.h, (int)m.c, (int)m.elemsize);
        } else {
            int rc = ex.extract(nm, m);
            if (rc != 0) { LOGW("extract %s fail rc=%d", nm ? nm : "?", rc); }
        }
        ncnn::Mat m1 = m;
        if (m.elempack != 1) { ncnn::Mat mp; ncnn::convert_packing(m, mp, 1); m1 = mp; }
        if (m1.elemsize == 2) { ncnn::Mat mf; ncnn::cast_float16_to_float32(m1, mf); m1 = mf; }
        const float* mp = (const float*)m1.data;
        size_t t = m1.total();
        dims[i] = (int)t;
        for (size_t k = 0; k < t; k++) out.push_back(mp[k]);
        if (js) env->ReleaseStringUTFChars(js, nm);
    }

    jint* ds = env->GetIntArrayElements(dimsOut, nullptr);
    jsize nds = env->GetArrayLength(dimsOut);
    for (jsize i = 0; i < nds; i++) ds[i] = (i < nNames) ? dims[i] : 0;
    env->ReleaseIntArrayElements(dimsOut, ds, 0);

    jfloatArray ret = env->NewFloatArray((jsize)out.size());
    if (ret && !out.empty()) env->SetFloatArrayRegion(ret, 0, (jsize)out.size(), out.data());
    return ret;
}

// ============================================================================
// 整条 face→landmark→pose 链的确定性 C++ 移植（源自 ht/insightface/Kotlin）。
// nativePose(frame BGR, w, h) → out[0..8]=R 的行 (r1,r2,r3)，R 把 canonical 帧旋到 landmark 系 L
//   (X 右 / Y 下 / Z 朝相机)；人脸自身三轴 = R 的「列」：横轴=(o0,o3,o6)、下轴=(o1,o4,o7)、
//   朝外法线=(o2,o5,o8)。不做朝相机强制翻号（翻号会让 yaw 反号，与 CoreML/insightface 不一致）。
// out[9..11]=pos3(cm, x右/y上/z朝前，含 3×IPD 前移)，out[12..25]=bbox14(x1,y1,x2,y2,5×2 kps 全帧 px)。
// 返回 1=有脸并解出；0=失败。
// ============================================================================

#define NCPOSE_DET 640   // 必须与 det 导出的固定 Reshape 尺寸一致（param 里 12800/3200/800=640²）
#define NCPOSE_LM 192
#define NCPOSE_THR 0.45f
#define NCPOSE_NMS 0.4f
#define NCPOSE_IPD_CM 6.2f
#define NCPOSE_FACE_CM 24.0f

// canonical 68pt 模板（与 insightface / Kotlin 完全一致）
static const float NC_CANONICAL[68 * 3] = {
    -0.626695f,-0.2927f,-0.314002f,-0.599665f,-0.122503f,-0.292441f,-0.571026f,0.051187f,-0.254947f,
    -0.533857f,0.212848f,-0.186633f,-0.479733f,0.350646f,-0.0477f,-0.395756f,0.446512f,0.073287f,
    -0.298808f,0.510661f,0.179778f,-0.188387f,0.55444f,0.31666f,0.001471f,0.584439f,0.388416f,
    0.190991f,0.551707f,0.314331f,0.326929f,0.489576f,0.1684f,0.440026f,0.402358f,0.035962f,
    0.506879f,0.311625f,-0.094761f,0.540895f,0.204526f,-0.202671f,0.574118f,0.045703f,-0.284176f,
    0.599142f,-0.145859f,-0.296496f,0.627544f,-0.307748f,-0.301995f,-0.474668f,-0.437605f,0.236487f,
    -0.41666f,-0.471756f,0.315993f,-0.347539f,-0.484078f,0.366113f,-0.26064f,-0.476391f,0.399232f,
    -0.167123f,-0.457775f,0.416609f,0.123174f,-0.45874f,0.425062f,0.206362f,-0.480419f,0.415784f,
    0.286673f,-0.490149f,0.391954f,0.362397f,-0.47686f,0.352786f,0.425568f,-0.450058f,0.295319f,
    -0.007628f,-0.323089f,0.461944f,-0.007877f,-0.25574f,0.510469f,-0.007689f,-0.19918f,0.552546f,
    -0.007345f,-0.142617f,0.598667f,-0.144961f,0.033127f,0.419691f,-0.084343f,0.031274f,0.473174f,
    -0.0055f,0.039751f,0.51467f,0.063463f,0.046135f,0.479224f,0.133984f,0.02204f,0.419079f,
    -0.386751f,-0.313398f,0.259632f,-0.316661f,-0.350072f,0.328527f,-0.234138f,-0.354914f,0.333493f,
    -0.155162f,-0.315249f,0.314328f,-0.230922f,-0.28427f,0.325583f,-0.317509f,-0.285165f,0.309882f,
    0.138957f,-0.309824f,0.318284f,0.219456f,-0.353192f,0.338028f,0.301747f,-0.349665f,0.333102f,
    0.376653f,-0.313518f,0.263229f,0.296695f,-0.287144f,0.322013f,0.214624f,-0.290528f,0.331242f,
    -0.201438f,0.237361f,0.379537f,-0.137321f,0.185786f,0.465253f,-0.076486f,0.151193f,0.503572f,
    -0.002536f,0.168727f,0.516438f,0.064421f,0.15088f,0.504524f,0.126465f,0.179476f,0.468592f,
    0.218248f,0.238992f,0.375674f,0.132883f,0.283928f,0.440051f,0.068022f,0.297354f,0.477404f,
    -0.000469f,0.300041f,0.487109f,-0.069343f,0.296969f,0.48054f,-0.14252f,0.274303f,0.43808f,
    -0.178135f,0.230591f,0.396359f,-0.074031f,0.214719f,0.465326f,-0.002636f,0.214142f,0.48323f,
    0.059816f,0.210764f,0.472244f,0.1669f,0.231271f,0.396789f,0.059809f,0.223764f,0.466413f,
    -0.001434f,0.225759f,0.475244f,-0.075221f,0.230656f,0.4671475f,
};

struct NCDet { float x1, y1, x2, y2, score; float kps[10]; };

// 张量 → fp32 elempack=1 连续（指针在调用区间内有效）
static const float* nc_unpack(const ncnn::Mat& src, ncnn::Mat& tmp) {
    ncnn::Mat m = src;
    if (m.elempack != 1) { ncnn::Mat mp; ncnn::convert_packing(m, mp, 1); m = mp; }
    if (m.elemsize == 2) { ncnn::Mat mf; ncnn::cast_float16_to_float32(m, mf); m = mf; }
    tmp = m;
    return (const float*)m.data;
}

// det 输入的等比缩放系数（aspect-preserve，长边→640）。节流帧不跑 det 也要用它把
// 640px 空间的锁定框换回帧坐标，所以单独成函数，与 nc_det_input 严格同源。
static float nc_det_scale(int w, int h) {
    const float rr = (float)h / w;
    const int nh = (rr > 1) ? NCPOSE_DET : (int)(NCPOSE_DET * rr);
    return (float)nh / h;
}

// det 输入：BGR 帧 aspect-preserve letterbox 到 640（图像占左上区）；返回归一化 RGB (x-127.5)/128。
static ncnn::Mat nc_det_input(const unsigned char* bgr, int w, int h, float& scale) {
    const int D = NCPOSE_DET;
    float rr = (float)h / w;
    int nh, nw;
    if (rr > 1) { nh = D; nw = (int)(D / rr); } else { nw = D; nh = (int)(D * rr); }
    scale = (float)nh / h;
    ncnn::Mat in(D, D, 3);
    const float fill = (0.0f - 127.5f) * (1.0f / 128.0f);
    const float inv = 1.0f / 128.0f;
    // 平面指针提出循环：原来每个像素都调 3 次 in.channel(c)（虚调用 + 对齐计算），
    // 640×480 帧就是 ~92 万次；letterbox 边框用 std::fill 向量化填。
    float* pR = (float*)in.channel(0);
    float* pG = (float*)in.channel(1);
    float* pB = (float*)in.channel(2);
    std::fill(pR, pR + (size_t)D * D, fill);
    std::fill(pG, pG + (size_t)D * D, fill);
    std::fill(pB, pB + (size_t)D * D, fill);
    const float iscale = 1.0f / scale;   // 乘法替代内层除法
    for (int y = 0; y < nh; y++) {
        int sy = (int)(y * iscale); if (sy >= h) sy = h - 1;
        const unsigned char* row = bgr + ((size_t)sy * w) * 3;
        size_t off = (size_t)y * D;
        for (int x = 0; x < nw; x++) {
            int sx = (int)(x * iscale); if (sx >= w) sx = w - 1;
            const unsigned char* p = row + sx * 3;
            pR[off + x] = (p[2] - 127.5f) * inv;   // RGB: 0=R
            pG[off + x] = (p[1] - 127.5f) * inv;
            pB[off + x] = (p[0] - 127.5f) * inv;
        }
    }
    return in;
}

// det 反解：9 输出 → 候选框（640px 坐标）
static void nc_decode(const ncnn::Mat* out, std::vector<NCDet>& dets) {
    const int strides[3] = {8, 16, 32};
    for (int k = 0; k < 3; k++) {
        int s = strides[k];
        int g = NCPOSE_DET / s;
        int per = g * g, total = per * 2;
        ncnn::Mat t;
        const float* sc = nc_unpack(out[k], t);
        const float* bb = nc_unpack(out[3 + k], t);
        const float* kk = nc_unpack(out[6 + k], t);
        for (int p = 0; p < total; p++) {
            float v = sc[p];
            if (v < NCPOSE_THR) continue;
            int cell = p / 2;
            int col = cell % g, row = cell / g;
            float ax = (float)col * s, ay = (float)row * s;
            int b4 = p * 4;
            if (b4 + 3 >= (int)(out[3 + k].total())) continue;
            NCDet d;
            d.x1 = ax - bb[b4] * s; d.y1 = ay - bb[b4 + 1] * s;
            d.x2 = ax + bb[b4 + 2] * s; d.y2 = ay + bb[b4 + 3] * s;
            int k4 = p * 10;
            for (int i = 0; i < 5 && k4 + i * 2 + 1 < (int)(out[6 + k].total()); i++) {
                d.kps[i * 2] = ax + kk[k4 + i * 2] * s;
                d.kps[i * 2 + 1] = ay + kk[k4 + i * 2 + 1] * s;
            }
            d.score = v;
            dets.push_back(d);
        }
    }
}

// —— 跨帧检测关联（锁定上一帧人脸，IoU 关联），避免 pick 在多张脸/窗口间来回切换 ——
// 框与关键点都在 det 的 640 letterbox 像素空间；ex/ey/sh 是上一帧 landmark 观测到的
// 特征（同空间），用来把框平移+缩放到下一帧，从而支持「det 隔帧跑」的节流。
struct NCTrack {
    bool on = false; float x1 = 0, y1 = 0, x2 = 0, y2 = 0; float kps[10] = {0};
    float ex = 0, ey = 0, sh = 0;        // 眼中心 + 眼→下巴竖直跨度（尺度参考）
    float vx = 0, vy = 0, vsr = 0;       // 每帧速度：由相邻两帧观测 EWMA 差分
    bool hasF = false;
    int sinceDet = 0;                    // 距上次真正跑 SCRFD 的帧数
};
static NCTrack g_track;
// det 节流档位：每 N 帧跑一次 SCRFD。属性 debug.gscp.detN（真机 A/B）、环境变量 GSCP_DET_N
// （host 离线；host 的 __system_property_get 是桩）覆盖，未设时用下面的默认值。
// 默认 3 的依据（vivo/Mali fp32 实测）：N=1 4.9 fps、N=3 7.5 fps、N=6 9.7 fps；离线连续帧
// 序列上 N=3 相对逐帧检测的偏差 p50 0.07–0.08°/p95 ≤0.66°、覆盖不掉帧（见 ht 报告 §5.13d）。
// 设备帧间隔约 130 ms，所以 N=3 的外推时间窗 ≈ 离线（25fps 视频）N=6，属已验证范围。
static int g_detEvery = 3;
static int g_sinceProp = 0;   // 属性读取节流计数（免每帧走 property 查找）
static long g_poseCalls = 0;  // nativePose 调用次数
static long g_detCalls = 0;   // 其中真正跑了 det 的次数
// 节流退回原因计数（nc_track_predict 每帧归一类）：0=到重检测周期 1=未锁定 2=无 landmark
// 参考 3=框过小 4=位移过快 5=尺度突变 6=外推框退化。用来在真机上判断「det 每 2.4 帧一次」
// 到底是节奏本身还是退回条件太紧，进而决定往哪个方向放宽。
static long g_tkR[7] = {0, 0, 0, 0, 0, 0, 0};

static void nc_det_prop() {
    if (g_sinceProp++ % 30) return;
    char s[8] = {0};
    __system_property_get("debug.gscp.detN", s);
    const char* e = getenv("GSCP_DET_N");
    int v = e && *e ? atoi(e) : (s[0] ? atoi(s) : 3);
    g_detEvery = v < 1 ? 1 : (v > 16 ? 16 : v);
}

// 用上帧观测到的运动把锁定框搬到本帧；返回 true = 本帧不必跑 SCRFD。
// 节奏：sinceDet = 距上次 det 已过的帧数（det 帧归零）。跳过帧 1..N-1、第 N 帧重检测
// → 恰好「每 N 帧一次 det」。任何不该外推的情形（未锁定/无特征/运动或尺度突变）都返回
// false 退回检测 —— 节流的全部风险都在这一步，宁可多跑 det。N=1 时恒 false，与逐帧检测等价。
static bool nc_track_predict(NCDet& box) {
    if (!g_track.on || g_detEvery <= 1) { g_tkR[1]++; return false; }
    if (g_track.sinceDet >= g_detEvery) { g_tkR[0]++; return false; }  // 到重检测周期
    if (!g_track.hasF) { g_tkR[2]++; return false; }
    const float bw = g_track.x2 - g_track.x1, bh = g_track.y2 - g_track.y1;
    if (!(bw > 8.f && bh > 8.f)) { g_tkR[3]++; return false; }
    if (std::abs(g_track.vx) > 0.15f * bw || std::abs(g_track.vy) > 0.15f * bh) {
        g_tkR[4]++; return false;
    }
    const float rho = 1.f + g_track.vsr;
    if (rho < 0.85f || rho > 1.18f) { g_tkR[5]++; return false; }
    const float cx = (g_track.x1 + g_track.x2) * 0.5f + g_track.vx;
    const float cy = (g_track.y1 + g_track.y2) * 0.5f + g_track.vy;
    const float hw = bw * 0.5f * rho, hh = bh * 0.5f * rho;
    box.x1 = cx - hw; box.y1 = cy - hh; box.x2 = cx + hw; box.y2 = cy + hh;
    box.score = 1.f;   // 复用框：置信度只用于日志，不参与挑选
    for (int i = 0; i < 5; i++) {
        box.kps[i * 2] = cx + (g_track.kps[i * 2] - (g_track.x1 + g_track.x2) * 0.5f) * rho;
        box.kps[i * 2 + 1] = cy + (g_track.kps[i * 2 + 1] - (g_track.y1 + g_track.y2) * 0.5f) * rho;
    }
    if (!(box.x2 > box.x1 && box.y2 > box.y1)) { g_tkR[6]++; return false; }
    return true;
}

static float bb_iou(float a1x, float a1y, float a2x, float a2y,
                    float b1x, float b1y, float b2x, float b2y) {
    float xx1 = std::max(a1x, b1x), yy1 = std::max(a1y, b1y),
          xx2 = std::min(a2x, b2x), yy2 = std::min(a2y, b2y);
    float iw = std::max(0.f, xx2 - xx1), ih = std::max(0.f, yy2 - yy1);
    float ix = iw * ih;
    float u = (a2x - a1x) * (a2y - a1y) + (b2x - b1x) * (b2y - b1y) - ix;
    return u > 1e-9f ? ix / u : 0.f;
}

// NMS + 锁定式跟踪：一旦锁定某个人脸，只跟随与其重叠(IoU>0.3)的候选；
// 短暂找不到时不乱跳到随机框(保持锁定)，连续超阈值帧仍无重叠才放弃并重新获取。
static int g_lostCnt = 0;   // 已无法关联到的连续帧数

JNIEXPORT void JNICALL
Java_com_gscp_desktop_NcnnEngine_nativeResetTrack(JNIEnv*, jobject)
{
    g_track.on = false; g_lostCnt = 0;
}
static int nc_pick(std::vector<NCDet>& dets) {
    if (dets.empty()) {   // 本帧det没输出可用框
        if (g_track.on) { g_lostCnt++; if (g_lostCnt > 16) { g_track.on = false; g_lostCnt = 0; } }
        return -1;
    }
    std::vector<int> order(dets.size());
    for (size_t i = 0; i < dets.size(); i++) order[i] = (int)i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return dets[a].score > dets[b].score; });
    std::vector<int> keep;
    while (!order.empty()) {
        int i = order.front();
        order.erase(order.begin());
        keep.push_back(i);
        float iArea = (dets[i].x2 - dets[i].x1) * (dets[i].y2 - dets[i].y1) + 1e-6f;
        for (size_t q = 0; q < order.size();) {
            int j = order[q];
            float xx1 = std::max(dets[i].x1, dets[j].x1), yy1 = std::max(dets[i].y1, dets[j].y1);
            float xx2 = std::min(dets[i].x2, dets[j].x2), yy2 = std::min(dets[i].y2, dets[j].y2);
            float ww = std::max(0.f, xx2 - xx1 + 1.f), hh = std::max(0.f, yy2 - yy1 + 1.f);
            if (ww * hh / iArea > NCPOSE_NMS) order.erase(order.begin() + q);
            else q++;
        }
    }
    if (keep.empty()) return -1;
    if (g_track.on) {
        // 已锁定：找与锁定框重叠(IoU>0.3)的候选，只跟随它
        int bestPick = -1; float bestIo = 0.1f;
        for (int k : keep) {
            float iou = bb_iou(g_track.x1, g_track.y1, g_track.x2, g_track.y2,
                               dets[k].x1, dets[k].y1, dets[k].x2, dets[k].y2);
            if (iou > bestIo) { bestIo = iou; bestPick = k; }
        }
        if (bestPick >= 0) { g_lostCnt = 0; return bestPick; }
        // 锁住的目标不在画面：短时不乱跳；连续超额帧仍没有才弃锁
        g_lostCnt++;
        if (g_lostCnt > 16) {
            g_track = NCTrack(); g_lostCnt = 0;
            // 允许重新获取：选当前最高分
            int bi = keep[0]; float bs = -1.f;
            for (int k : keep) if (dets[k].score > bs) { bs = dets[k].score; bi = k; }
            g_track.on = true; g_track.x1 = dets[bi].x1; g_track.y1 = dets[bi].y1;
            g_track.x2 = dets[bi].x2; g_track.y2 = dets[bi].y2;
            return bi;
        }
        return -1;
    } else {
        // 初始/重新获取：选 score≥max(THR, .9·best) 中面积最大者作为新锁定
        g_lostCnt = 0;
        float best = -1.f; for (int k : keep) best = std::max(best, dets[k].score);
        float thr = std::max(NCPOSE_THR, 0.9f * best);
        int pick = keep[0], area = -1;
        for (int k : keep) if (dets[k].score >= thr) {
            float a = (dets[k].x2 - dets[k].x1) * (dets[k].y2 - dets[k].y1);
            if (a > area) { area = (int)a; pick = k; }
        }
        g_track.on = true; g_track.x1 = dets[pick].x1; g_track.y1 = dets[pick].y1;
        g_track.x2 = dets[pick].x2; g_track.y2 = dets[pick].y2;
        return pick;
    }
}

// 1k3d68 输入：BGR 帧按仿射 M=[s0..] 裁剪到 192（raw RGB 0..255）
static ncnn::Mat lm_input(const unsigned char* bgr, int w, int h, float bx1, float by1, float bx2, float by2,
                          double& i00, double& i01, double& i02, double& i10, double& i11, double& i12, double& zS) {
    const int L = NCPOSE_LM;
    float w0 = bx2 - bx1, h0 = by2 - by1;
    float cx = (bx2 + bx1) * 0.5f, cy = (by2 + by1) * 0.5f;
    float sM = L / (std::max(w0, h0) * 1.5f);
    // M = [sM,0, L/2 - cx*sM; 0, sM, L/2 - cy*sM], 行主序
    double a00 = sM, a01 = 0.0, a02 = L * 0.5 - (double)cx * sM;
    double a10 = 0.0, a11 = sM, a12 = L * 0.5 - (double)cy * sM;
    double det = a00 * a11 - a01 * a10;
    i00 = a11 / det; i01 = -a01 / det; i02 = (a01 * a12 - a02 * a11) / det;
    i10 = -a10 / det; i11 = a00 / det; i12 = (a02 * a10 - a00 * a12) / det;
    zS = std::sqrt(i00 * i00 + i01 * i01);

    ncnn::Mat o(L, L, 3);
    for (int c = 0; c < 3; c++) { float* pc = (float*)o.channel(c); for (int p = 0; p < L * L; p++) pc[p] = 0.f; }
    // 重采样必须用 dst→src 的逆仿射(i**)：等价于 cv2.warpAffine(M) 的内部映射。
    // 曾用正变换 a**（步进 sM<1）→ 只采到以 cx,cy 为中心 ~85px 窗（远小于 1.5×bbox），
    // landmark 网络输入错误，姿态近常数（jar 离线测试复现）。
    for (int py = 0; py < L; py++) {
        double v = i11 * py + i12;
        for (int px = 0; px < L; px++) {
            double u = i00 * px + i02;
            int u0 = (int)std::floor(u), v0 = (int)std::floor(v);
            double fu = u - u0, fv = v - v0;
            float rb = 0, rg = 0, rr = 0;
            if (u >= 0 && v >= 0 && u0 + 1 < w && v0 + 1 < h) {
                const unsigned char* p00 = bgr + (((size_t)v0) * w + u0) * 3;
                const unsigned char* p10 = p00 + 3;
                const unsigned char* p01 = p00 + ((size_t)w) * 3;
                const unsigned char* p11 = p01 + 3;
                float b = (1 - fu) * (1 - fv) * p00[0] + fu * (1 - fv) * p10[0] + (1 - fu) * fv * p01[0] + fu * fv * p11[0];
                float g = (1 - fu) * (1 - fv) * p00[1] + fu * (1 - fv) * p10[1] + (1 - fu) * fv * p01[1] + fu * fv * p11[1];
                float r = (1 - fu) * (1 - fv) * p00[2] + fu * (1 - fv) * p10[2] + (1 - fu) * fv * p01[2] + fu * fv * p11[2];
                rb = b; rg = g; rr = r;
            }
            size_t idx = (size_t)py * L + px;
            ((float*)o.channel(0))[idx] = rr;   // R
            ((float*)o.channel(1))[idx] = rg;   // G
            ((float*)o.channel(2))[idx] = rb;   // B
        }
    }
    return o;
}

// 高斯消元解 4×3（pivot）；失败返回 false
static bool nc_solve4(const double A[4][4], const double b[4][3], double X[4][3]) {
    double m[4][4], x[4][3];
    for (int r = 0; r < 4; r++) { for (int c = 0; c < 4; c++) m[r][c] = A[r][c];
        for (int c = 0; c < 3; c++) x[r][c] = b[r][c]; }
    for (int col = 0; col < 4; col++) {
        int piv = col; double best = std::abs(m[col][col]);
        for (int r = col + 1; r < 4; r++) { double v = std::abs(m[r][col]); if (v > best) { best = v; piv = r; } }
        if (best < 1e-12) return false;
        if (piv != col) { for (int c = 0; c < 4; c++) std::swap(m[col][c], m[piv][c]);
            for (int c = 0; c < 3; c++) std::swap(x[col][c], x[piv][c]); }
        double pv = m[col][col];
        for (int c = 0; c < 4; c++) m[col][c] /= pv;
        for (int c = 0; c < 3; c++) x[col][c] /= pv;
        for (int r = 0; r < 4; r++) if (r != col) {
            double f = m[r][col]; if (std::abs(f) < 1e-15) continue;
            for (int c = 0; c < 4; c++) m[r][c] -= f * m[col][c];
            for (int c = 0; c < 3; c++) x[r][c] -= f * x[col][c];
        }
    }
    for (int r = 0; r < 4; r++) for (int c = 0; c < 3; c++) X[r][c] = x[r][c];
    return true;
}

static void cross3(const double a[3], const double b3[3], double o[3]) {
    o[0] = a[1] * b3[2] - a[2] * b3[1];
    o[1] = a[2] * b3[0] - a[0] * b3[2];
    o[2] = a[0] * b3[1] - a[1] * b3[0];
}

static float eye_center_x(const float* lmk, int i0, int i1) {
    double sx = 0; for (int i = i0; i <= i1; i++) sx += lmk[i * 3];
    return (float)(sx / (i1 - i0 + 1));
}
static float eye_center_y(const float* lmk, int i0, int i1) {
    double sy = 0; for (int i = i0; i <= i1; i++) sy += lmk[i * 3 + 1];
    return (float)(sy / (i1 - i0 + 1));
}

JNIEXPORT jint JNICALL
Java_com_gscp_desktop_NcnnEngine_nativePose(JNIEnv* env, jobject,
                                            jlong detHandle, jlong lmHandle,
                                            jbyteArray frame, jint w, jint h,
                                            jfloatArray out) {
    ncnn::Net* det = (ncnn::Net*)detHandle;
    ncnn::Net* lm = (ncnn::Net*)lmHandle;
    if (!det || !lm || w <= 0 || h <= 0 || !out) return 0;
    jsize olen = env->GetArrayLength(out);
    if (olen < 26) return 0;
    jfloat* o = env->GetFloatArrayElements(out, nullptr);
    if (!o) return 0;
    int R = 0;
    do {
        jbyte* b = env->GetByteArrayElements(frame, nullptr);
        if (!b) break;
        const unsigned char* bgr = (const unsigned char*)b;

        double _t = nc_ms();
        g_poseCalls++;
        nc_det_prop();
        if (g_track.on) g_track.sinceDet++;   // 距上次检测的帧数；跑 det 的帧会归零
        float scale = nc_det_scale(w, h);
        NCDet dm;
        // det 节流：SCRFD 640×640 是单帧最大成本（真机 Mali fp32 实测 det 131–160 ms，
        // 1k3d68@192 只 32–42 ms），而位姿由 landmark 决定。每 g_detEvery 帧跑一次 SCRFD，
        // 其余帧用上一帧 landmark 观测到的平移/缩放把锁定框外推出来（见 nc_track_predict）。
        // g_detEvery=1（默认/属性未设）时 predict 直接返回 false → 与逐帧检测完全等价。
        if (!nc_track_predict(dm)) {
            ncnn::Mat inm = nc_det_input(bgr, w, h, scale);
            NCP(0);

            // det forward → 9 输出
            ncnn::Mat outm[9];
            {
                ncnn::Extractor ex = det->create_extractor();
                if (ex.input("in0", inm) != 0) { env->ReleaseByteArrayElements(frame, b, JNI_ABORT); break; }
                const char* os[9] = {"out0","out1","out2","out3","out4","out5","out6","out7","out8"};
                bool ok = true;
                for (int i = 0; i < 9; i++) if (ex.extract(os[i], outm[i]) != 0) { ok = false; break; }
                if (!ok) { env->ReleaseByteArrayElements(frame, b, JNI_ABORT); break; }
            }
            NCP(1);
            g_detCalls++;

            std::vector<NCDet> dets;
            nc_decode(outm, dets);
            int pick = nc_pick(dets);
            if (pick < 0) { env->ReleaseByteArrayElements(frame, b, JNI_ABORT); break; }
            dm = dets[pick];
            if (!(dm.x1 >= 0.f)) { env->ReleaseByteArrayElements(frame, b, JNI_ABORT); break; }   // NaN 校验
            g_track.sinceDet = 0;
            NCP(2);
        } else {
            NCP(2);
        }
        const float bw = dm.x2 - dm.x1, bh = dm.y2 - dm.y1;
        if (!(bw > 0.f && bh > 0.f)) { env->ReleaseByteArrayElements(frame, b, JNI_ABORT); break; }
        // 更新跨帧锁定（640px 坐标 + 关键点，供下一帧外推）：保持同一张脸
        g_track.on = true; g_track.x1 = dm.x1; g_track.y1 = dm.y1; g_track.x2 = dm.x2; g_track.y2 = dm.y2;
        for (int i = 0; i < 10; i++) g_track.kps[i] = dm.kps[i];

        float bx1 = dm.x1 / scale, by1 = dm.y1 / scale, bx2 = dm.x2 / scale, by2 = dm.y2 / scale;

        // —— 1k3d68 前向 ——
        double i00, i01, i02, i10, i11, i12, zS;
        ncnn::Mat lmIn = lm_input(bgr, w, h, bx1, by1, bx2, by2, i00, i01, i02, i10, i11, i12, zS);
        NCP(3);
        env->ReleaseByteArrayElements(frame, b, JNI_ABORT);   // 像素用完，释放
        ncnn::Mat predM;
        {
            ncnn::Extractor ex = lm->create_extractor();
            if (ex.input("in0", lmIn) != 0) break;
            if (ex.extract("out0", predM) != 0) break;
        }
        NCP(4);
        ncnn::Mat pu;
        const float* pred = nc_unpack(predM, pu);
        // 逻辑长度（w*h*c=3309=1103点×3），不能用 predM.total()：ncnn 会按 SIMD 对齐
        // 补到 3312（1104点），使 base=npred/3-68 多算 1 点 → x/y/z 三元组整体错位
        // 1 点，姿态被锁死成近常数（jar 离线测试对 ht/CoreML 实测复现）。
        size_t npred = (size_t)predM.w * predM.h * predM.c;
        if (npred < 3 * 69) break;
        size_t base = npred / 3 - 68;
        float half = NCPOSE_LM * 0.5f;
        float lmk[68 * 3];
        for (int i = 0; i < 68; i++) {
            size_t j = base + i;
            float x = (pred[j * 3] + 1.0f) * half;
            float y = (pred[j * 3 + 1] + 1.0f) * half;
            float z = pred[j * 3 + 2] * half;
            lmk[i * 3] = (float)(i00 * x + i01 * y + i02);
            lmk[i * 3 + 1] = (float)(i10 * x + i11 * y + i12);
            lmk[i * 3 + 2] = (float)(z * zS);
        }
        bool nany = false; for (int i = 0; i < 68 * 3; i++) if (!(lmk[i] == lmk[i])) { nany = true; break; }
        if (nany) break;
        // —— 节流用的特征观测（与锁定框同在 640px 空间：lmk 是帧坐标，×scale 还原）——
        // 眼中心给平移，眼→下巴竖直跨度给尺度：跨度沿 y 方向，受 yaw 影响小，比瞳距稳。
        {
            const float exx = (eye_center_x(lmk, 36, 40) + eye_center_x(lmk, 42, 46)) * 0.5f * scale;
            const float eyy = (eye_center_y(lmk, 36, 40) + eye_center_y(lmk, 42, 46)) * 0.5f * scale;
            const float shh = lmk[8 * 3 + 1] * scale - eyy;
            if (shh > 4.f) {
                if (g_track.hasF) {
                    const float a = 0.5f;   // EWMA：不把单帧 landmark 噪声全转成速度
                    g_track.vx += (exx - g_track.ex - g_track.vx) * a;
                    g_track.vy += (eyy - g_track.ey - g_track.vy) * a;
                    g_track.vsr += (shh / g_track.sh - 1.f - g_track.vsr) * a;
                }
                g_track.ex = exx; g_track.ey = eyy; g_track.sh = shh; g_track.hasF = true;
            }
        }
#ifdef __APPLE__
        if (getenv("GSCP_LMK")) {
            double mz = 0; for (int i = 0; i < 68; i++) mz += lmk[i * 3 + 2];
            fprintf(stderr, "[lmk] npred=%zu base=%zu lm0=(%.1f,%.1f,%.2f) lm30=(%.1f,%.1f,%.2f) meanz=%.2f\n",
                    (size_t)(predM.total()), base, lmk[0], lmk[1], lmk[2], lmk[90], lmk[91], lmk[92], mz / 68);
            if (getenv("GSCP_RAW")) {
                fprintf(stderr, "[raw] w=%d h=%d c=%d pack=%d es=%d tot=%zu zS=%.4f i00=%.4f i02=%.2f i11=%.4f i12=%.2f\n",
                        predM.w, predM.h, predM.c, predM.elempack, predM.elemsize, predM.total(), zS, i00, i02, i11, i12);
                fprintf(stderr, "[raw] first12="); for (int i=0;i<12;i++) fprintf(stderr," %.5f", pred[i]); fprintf(stderr,"\n");
                fprintf(stderr, "[raw] last6pts="); for (int i=0;i<2;i++) fprintf(stderr," (%.4f,%.4f,%.4f)", pred[base*3+i*3],pred[base*3+i*3+1],pred[base*3+i*3+2]); fprintf(stderr,"\n");
            }
            if (getenv("GSCP_RAW2")) {
                FILE* f = fopen("/tmp/lmk_raw.txt", "a");
                if (f) { for (int i = 0; i < 68; i++) fprintf(f, "%.5f %.5f %.5f\n",
                          pred[(base+i)*3], pred[(base+i)*3+1], pred[(base+i)*3+2]); fclose(f); }
            }
            if (getenv("GSCP_LMK2")) {
                FILE* f = fopen("/tmp/lmk_dump.txt", "a");
                if (f) { for (int i = 0; i < 68; i++) fprintf(f, "%.4f %.4f %.4f\n", lmk[i*3], lmk[i*3+1], lmk[i*3+2]);
                         fprintf(f, "# bbox %.2f %.2f %.2f %.2f img %dx%d\n", bx1, by1, bx2, by2, w, h); fclose(f); }
            }
        }
#endif

        // —— Procrustes（法方程）→ basis ——
        double A[4][4] = {{0}}, RHS[4][3] = {{0}};
        for (int i = 0; i < 68; i++) {
            double s[4] = { NC_CANONICAL[i * 3], NC_CANONICAL[i * 3 + 1], NC_CANONICAL[i * 3 + 2], 1.0 };
            double d[3] = { lmk[i * 3], lmk[i * 3 + 1], lmk[i * 3 + 2] };
            for (int r = 0; r < 4; r++) for (int t = 0; t < 4; t++) A[r][t] += s[r] * s[t];
            for (int r = 0; r < 4; r++) for (int t = 0; t < 3; t++) RHS[r][t] += s[r] * d[t];
        }
        double B[4][3];
        if (!nc_solve4(A, RHS, B)) break;

        double r1[3] = { B[0][0], B[1][0], B[2][0] };
        double r2[3] = { B[0][1], B[1][1], B[2][1] };
        double n1 = std::sqrt(r1[0]*r1[0] + r1[1]*r1[1] + r1[2]*r1[2]);
        double n2 = std::sqrt(r2[0]*r2[0] + r2[1]*r2[1] + r2[2]*r2[2]);
        if (n1 < 1e-9 || n2 < 1e-9) break;
        double r1n[3] = { r1[0]/n1, r1[1]/n1, r1[2]/n1 };
        double r2n[3] = { r2[0]/n2, r2[1]/n2, r2[2]/n2 };
        double r3n[3]; cross3(r1n, r2n, r3n);
        double n3 = std::sqrt(r3n[0]*r3n[0] + r3n[1]*r3n[1] + r3n[2]*r3n[2]);
        if (n3 < 1e-9) break;
        float nx = (float)(r3n[0]/n3), ny = (float)(r3n[1]/n3), nz = (float)(r3n[2]/n3);
        // 此处曾强制 nz>0（法线朝相机）：与 insightface P2sRt 的 r3 约定不一致，
        // 会使 yaw 反号（jar 离线测试对 CoreML 实测）。朝向修正只放在渲染层做。
        float rx = (float)r1n[0], ry = (float)r1n[1], rz = (float)r1n[2];
        float ux = (float)r2n[0], uy = (float)r2n[1], uz = (float)r2n[2];

        // —— 位置（人脸框高度 + 垂直 FOV 50° 针孔，3 瞳距前移）——
        // 深度由检测框竖直跨度推得：框高受锁定时比 IPD 稳定，避免远处小脸的
        // 眼距/landmark 噪声让深度在帧间大幅跳动。
        float fbh = by2 - by1;
        if (!(fbh > 1.f)) break;
        double focal = (h * 0.5) / std::tan(25.0 * (3.14159265358979323846 / 180.0));
        float depth = NCPOSE_FACE_CM * (float)focal / fbh;
        float bcx = (bx1 + bx2) * 0.5f, bcy = (by1 + by2) * 0.5f;
        float wxx = (bcx - w * 0.5f) / (float)focal * depth;
        float wy = (h * 0.5f - bcy) / (float)focal * depth;
        float d3 = 3.0f * NCPOSE_IPD_CM;
        // landmark/Procrustes 系 L：X 右 / Y 下 / Z 朝相机（canonical 鼻尖 z 最大 → 脸法线 = +R 第3列）。
        // 位置系 P：X 右 / Y 上 / Z 朝前（渲染器再取 -z 得 GL 系）。
        // 朝相机位移 = +法线在 P 系 = (rz, -uz, -nz)；旧实现误用 r3 行并强制 +z，overlay 被推到脸后 18.6cm。
        float px = wxx + rz * d3, py1 = wy - uz * d3, pz = depth - nz * d3;

        // —— 写出 ——
        o[0] = rx; o[1] = ry; o[2] = rz;
        o[3] = ux; o[4] = uy; o[5] = uz;
        o[6] = nx; o[7] = ny; o[8] = nz;
        o[9] = px; o[10] = py1; o[11] = pz;
        o[12] = bx1; o[13] = by1; o[14] = bx2; o[15] = by2;
        for (int i = 0; i < 5 && 4 + i * 2 + 1 < 14; i++) { o[16 + i * 2] = dm.kps[i * 2] / scale; o[17 + i * 2] = dm.kps[i * 2 + 1] / scale; }
        NCP(5);
        R = 1;
    } while (0);
    env->ReleaseFloatArrayElements(out, o, R ? 0 : JNI_ABORT);
    return R;
}

/** nativePose 阶段均值（µs）+ 计数：
 *  [0..5]=六段均值 [6]=成功帧数 [7]=调用次数 [8]=其中真正跑了 det 的次数
 *  [9..15]=节流退回原因计数（见 g_tkR 注释：0=周期 1=未锁 2=无特征 3=框小 4=运动 5=尺度 6=退化）。
 *  各段样本数不同（见 g_pc 注释）；节流后 det 均值只按 det 帧算，摊到每帧要乘 [8]/[7]。 */
JNIEXPORT jdoubleArray JNICALL
Java_com_gscp_desktop_NcnnEngine_nativeProf(JNIEnv* env, jobject) {
    jdoubleArray a = env->NewDoubleArray(9 + 7);
    if (!a) return nullptr;
    double v[9 + 7];
    for (int i = 0; i < 6; i++) v[i] = g_prof[i] / (g_pc[i] > 0 ? g_pc[i] : 1) * 1e3;
    v[6] = (double)g_pc[5];
    v[7] = (double)g_poseCalls;
    v[8] = (double)g_detCalls;
    for (int i = 0; i < 7; i++) v[9 + i] = (double)g_tkR[i];
    env->SetDoubleArrayRegion(a, 0, 9 + 7, v);
    return a;
}

JNIEXPORT void JNICALL
Java_com_gscp_desktop_NcnnEngine_nativeProfReset(JNIEnv*, jobject) {
    for (int i = 0; i < 6; i++) { g_prof[i] = 0; g_pc[i] = 0; }
    g_poseCalls = 0; g_detCalls = 0;
    for (int i = 0; i < 7; i++) g_tkR[i] = 0;
}

}  // extern "C"