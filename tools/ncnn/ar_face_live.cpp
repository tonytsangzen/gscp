// ar_face_live.cpp — gscp AR 模式 native 管线，自 ncnn-benchmark 项目 face_live.cpp 移植。
// 主路径 = 已在真机验证的 algo3 融合：SCRFD det_10g(448) 逐帧检测 → MediaPipe FaceMesh
// (468 点, 192²) → 9 点 Horn Kabsch 求姿 → 大角度(35°→55° 线性)混入 hopenet(48² 灰度) →
// One-Euro 帧间滤波(旋转/平移/overlay 像素) → 几何锚定 overlay 平面四角。
// 与蓝本的差异仅两处：JNI 包名改为 com.gscp.desktop.ArNative；裁掉 PFLD/tddfa/insight
// 备选路径与回放模式（评测用），姿态源固定为融合(3)，hopenet(1) 保留作对照。
// 解码（与 insightface scrfd.py 一致）：score 图内已 Sigmoid；bbox/kps 解码乘 stride；
// anchor 中心 = (格点 x, 格点 y) * stride（无 0.5 偏移）；blob 线性序 i = (y*W + x)*2 + a。
#include <jni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "ncnn/net.h"
#include "ncnn/cpu.h"
#include "ncnn/gpu.h"
#include "ncnn/benchmark.h"
#include "ncnn/datareader.h"
#include "ncnn/mat.h"

#include "out_norm.h"

#define LOG_TAG "ar-native"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

const char* kDetParam = "det_10g.ncnn.param";
const char* kDetBin = "det_10g.ncnn.bin";
const char* kDetInput = "in0";
// 与 param 中 reshape 对应：out0..2=score, out3..5=bbox(w=4), out6..8=kps(w=10)
const char* kScoreBlob[3] = {"out0", "out1", "out2"};
const char* kBoxBlob[3] = {"out3", "out4", "out5"};
const char* kKpsBlob[3] = {"out6", "out7", "out8"};
const int kStride[3] = {8, 16, 32};
const float kScoreThresh = 0.50f;
const float kNmsIou = 0.45f;
const int kMaxFaces = 20;
// det_10g 图内写死了 448 输入的 FPN 上采样尺寸，固定按 448 运行
const int kIn = 448;

const char* kBackendName[4] = {"CPU NEON FP32", "CPU NEON FP16", "Vulkan FP16", "Vulkan FP32"};

// hopenet 48x48 灰度 + 融合（mesh Kabsch + hopenet 大角度）——真机评估选定
const char* kHopParam = "hopenet_fp32.ncnn.param";
const char* kHopBin = "hopenet_fp32.ncnn.bin";

// MediaPipe FaceMesh 468 点（192² RGB，(x-127.5)/127.5）
const char* kFmParam = "facemesh.ncnn.param";
const char* kFmBin = "facemesh.ncnn.bin";
const int kFmCrop = 192;
const int kFmPoints = 468;

std::mutex g_mutex;      // 串行化 open/detect/close
ncnn::Net* g_net = nullptr;

// overlay 平面距离（cm）：属性 debug.gscp.ovcm（真机免重编译调参）/ 环境变量
// GSCP_OV_CM（host）可覆盖，默认 25，钳制 [5,200]。
static int g_ovCm = 25;
static int g_ovPropCnt = 0;
static void nc_ov_prop() {
    if (g_ovPropCnt++ % 30) return;
    char s[8] = {0};
    __system_property_get("debug.gscp.ovcm", s);
    const char* e = getenv("GSCP_OV_CM");
    int v = e && *e ? atoi(e) : (s[0] ? atoi(s) : 25);
    g_ovCm = v < 5 ? 5 : (v > 200 ? 200 : v);
}

// overlay 投影大小倍数（投影宽 = 倍数 × 脸宽）：属性 debug.gscp.ovscale /
// 环境变量 GSCP_OV_SCALE 可覆盖，默认 3（脸宽 3 倍，钳制上限），钳制 [0.5,3]。
// 距离只挪位置不改大小，大小由本倍数唯一决定（投影恒 1:1）。
static float g_ovScale = 3.f;
static int g_ovScaleCnt = 0;
static void nc_ov_scale_prop() {
    if (g_ovScaleCnt++ % 30) return;
    char s[16] = {0};
    __system_property_get("debug.gscp.ovscale", s);
    const char* e = getenv("GSCP_OV_SCALE");
    float v = e && *e ? (float)atof(e) : (s[0] ? (float)atof(s) : 3.f);
    g_ovScale = v < 0.5f ? 0.5f : (v > 3.f ? 3.f : v);
}

// overlay 平移（cm，板平面内沿横轴/竖轴）：属性 debug.gscp.ovpanx / ovpany
// （环境变量 GSCP_OV_PAN_X/Y 可覆盖），默认水平 0 / 垂直 5，钳制 [-10,10]。
// 沿板轴平移随头部姿态自然移动；正值方向以真机实测为准（横轴 = 眼线方向）。
static float g_ovPanX = 0.f, g_ovPanY = 5.f;
static int g_ovPanCnt = 0;
static void nc_ov_pan_prop() {
    if (g_ovPanCnt++ % 30) return;
    char sx[16] = {0}, sy[16] = {0};
    __system_property_get("debug.gscp.ovpanx", sx);
    __system_property_get("debug.gscp.ovpany", sy);
    const char* ex = getenv("GSCP_OV_PAN_X");
    const char* ey = getenv("GSCP_OV_PAN_Y");
    float vx = ex && *ex ? (float)atof(ex) : (sx[0] ? (float)atof(sx) : 0.f);
    float vy = ey && *ey ? (float)atof(ey) : (sy[0] ? (float)atof(sy) : 5.f);
    g_ovPanX = vx < -10.f ? -10.f : (vx > 10.f ? 10.f : vx);
    g_ovPanY = vy < -10.f ? -10.f : (vy > 10.f ? 10.f : vy);
}
int g_backend = -1;
int g_input = 0;
int g_threads = 1;
bool g_gpuOk = false;
bool g_shutdown = false;
ncnn::Net* g_fmNet = nullptr;      // facemesh（融合姿态源）
ncnn::Net* g_hopeNet = nullptr;    // hopenet（大角度融合项）
ncnn::VulkanDevice* g_vk = nullptr;

int g_poseAlgo = 3;                // 3=融合（默认/唯一主路径） 1=hopenet（对照）
static volatile int g_ovW = 480, g_ovH = 480;   // 背景层恒为 480×480（首个流帧到达前也按此比例）
static const char* poseAlgoName() { return g_poseAlgo == 3 ? "fusion" : "hopenet"; }

std::vector<unsigned char> readAsset(AAssetManager* am, const char* name)
{
    std::vector<unsigned char> data;
    if (!am) return data;
    AAsset* a = AAssetManager_open(am, name, AASSET_MODE_BUFFER);
    if (!a) return data;
    const off_t len = AAsset_getLength(a);
    data.resize((size_t)len);
    const int got = AAsset_read(a, data.data(), (size_t)len);
    AAsset_close(a);
    if (got != (int)data.size()) data.clear();
    return data;
}

double nowMs()
{
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// ---- YUV -> BGR（BT.601 limited range，Android YUV_420_888 标准约定） ----
void yuvToBgr(const unsigned char* Y, const unsigned char* U, const unsigned char* V,
              int w, int h, int yStride, int uStride, int vStride, int uPix, int vPix,
              unsigned char* bgr)
{
    for (int y = 0; y < h; y++) {
        const unsigned char* yrow = Y + (size_t)y * yStride;
        const unsigned char* urow = U + (size_t)(y >> 1) * uStride;
        const unsigned char* vrow = V + (size_t)(y >> 1) * vStride;
        unsigned char* out = bgr + (size_t)y * w * 3;
        for (int x = 0; x < w; x++) {
            int yy = yrow[x] - 16;
            int uu = urow[(x >> 1) * uPix] - 128;
            int vv = vrow[(x >> 1) * vPix] - 128;
            int r = (298 * yy + 409 * vv + 128) >> 8;
            int g = (298 * yy - 100 * uu - 208 * vv + 128) >> 8;
            int b = (298 * yy + 516 * uu + 128) >> 8;
            out[x * 3 + 0] = (unsigned char)(b < 0 ? 0 : b > 255 ? 255 : b);
            out[x * 3 + 1] = (unsigned char)(g < 0 ? 0 : g > 255 ? 255 : g);
            out[x * 3 + 2] = (unsigned char)(r < 0 ? 0 : r > 255 ? 255 : r);
        }
    }
}

// 顺时针旋转 0/90/180/270 度成直立图；rot 90/270 时输出宽高互换
// 90:  dst(x,y) = src(y, h-1-x)   180: dst(x,y) = src(w-1-x, h-1-y)
// 270: dst(x,y) = src(w-1-y, x)
void rotateBgr(const unsigned char* src, int w, int h, int rot, unsigned char* dst,
               int* rw, int* rh)
{
    if (rot == 90) {
        *rw = h; *rh = w;
        for (int y = 0; y < w; y++) {
            for (int x = 0; x < h; x++) {
                const unsigned char* s = src + ((size_t)(h - 1 - x) * w + y) * 3;
                unsigned char* d = dst + ((size_t)y * h + x) * 3;
                d[0] = s[0]; d[1] = s[1]; d[2] = s[2];
            }
        }
    } else if (rot == 180) {
        *rw = w; *rh = h;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                memcpy(dst + ((size_t)y * w + x) * 3,
                       src + ((size_t)(h - 1 - y) * w + (w - 1 - x)) * 3, 3);
    } else if (rot == 270) {
        *rw = h; *rh = w;
        for (int y = 0; y < w; y++) {
            for (int x = 0; x < h; x++) {
                const unsigned char* s = src + ((size_t)x * w + (w - 1 - y)) * 3;
                unsigned char* d = dst + ((size_t)y * h + x) * 3;
                d[0] = s[0]; d[1] = s[1]; d[2] = s[2];
            }
        }
    } else {
        *rw = w; *rh = h;
        if (dst != src) memcpy(dst, src, (size_t)w * h * 3);
    }
}

struct Face {
    float box[4];   // x1 y1 x2 y2（直立原图坐标）
    float kps[10];  // 5 个关键点 x/y
    float score;
    bool hasPose = false;
    float ypr[3];       // pitch/yaw/roll（度）
    float normal[2][2]; // 面部法线：鼻尖起点 + 朝向端点（直立原图坐标）
    bool hasOv = false;
    float ov[4][2];     // overlay 平面四角（顺时针，直立原图坐标）
    float ovB[4][2];    // 背板四角：ov 沿平面法线后推 thicknessCm（立体厚度用）
    float ovC[2];       // 平面中心（用于标签）
    float ovD = 0;      // 实际绘制距离（cm）
};

// 5 点 PnP 回退模型（眼/眼/鼻/嘴角/嘴角；mesh 失败时兜底）
static const float kModel5[5][3] = {
    {-180, -170, 135}, { 180, -170, 135}, {0, 0, 0}, {-150, 150, 125}, {150, 150, 125}
};
static const float kEyeSpan5 = 360.f;

// 解码一个 stride 的输出（blob 线性序 i = (y*W + x)*2 + a，anchor 下标在最内层）
void decodeStride(const ncnn::Mat& sc, const ncnn::Mat& bb, const ncnn::Mat& kp,
                  int stride, std::vector<Face>* out)
{
    const int gw = g_input / stride;
    if (gw <= 0) return;
    const int g = gw * gw;
    const int rows = std::min(sc.h, g * 2);
    for (int i = 0; i < rows; i++) {
        const float s = sc.row(i)[0];
        if (s < kScoreThresh) continue;
        const int a = i & 1;
        const int rem = i >> 1;                 // = y*gw + x
        const int y = rem / gw, x = rem - y * gw;
        const float cx = (float)x * stride, cy = (float)y * stride;
        const float* d = bb.row(i);
        const float* k = kp.row(i);
        Face f;
        f.score = s;
        f.box[0] = cx - d[0] * stride; f.box[1] = cy - d[1] * stride;
        f.box[2] = cx + d[2] * stride; f.box[3] = cy + d[3] * stride;
        for (int j = 0; j < 5; j++) {
            f.kps[2 * j] = cx + k[2 * j] * stride;
            f.kps[2 * j + 1] = cy + k[2 * j + 1] * stride;
        }
        out->push_back(f);
    }
}

float iou(const Face& a, const Face& b)
{
    const float x1 = std::max(a.box[0], b.box[0]);
    const float y1 = std::max(a.box[1], b.box[1]);
    const float x2 = std::min(a.box[2], b.box[2]);
    const float y2 = std::min(a.box[3], b.box[3]);
    const float inter = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
    const float areaA = (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]);
    const float areaB = (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]);
    const float den = areaA + areaB - inter;
    return den > 0 ? inter / den : 0.f;
}

std::string f1(double v) { char b[32]; snprintf(b, sizeof(b), "%.1f", v); return b; }
std::string f2(double v) { char b[32]; snprintf(b, sizeof(b), "%.2f", v); return b; }
std::string f3(double v) { char b[32]; snprintf(b, sizeof(b), "%.3f", v); return b; }

static void rodrigues(const float* w, float R[9]) {
    float th2 = w[0]*w[0] + w[1]*w[1] + w[2]*w[2];
    float th = sqrtf(th2);
    float A = th < 1e-6f ? 1.f : sinf(th) / th;
    float B = th < 1e-6f ? 0.5f : (1.f - cosf(th)) / th2;
    float c = 1.f - B * th2;
    float x = w[0], y = w[1], z = w[2];
    R[0] = c + B*x*x;      R[1] = -A*z + B*x*y;   R[2] =  A*y + B*x*z;
    R[3] =  A*z + B*x*y;   R[4] = c + B*y*y;      R[5] = -A*x + B*y*z;
    R[6] = -A*y + B*x*z;   R[7] =  A*x + B*y*z;   R[8] = c + B*z*z;
}

static void r2w(const float R[9], float* w) {
    float tr = (R[0] + R[4] + R[8] - 1.f) * 0.5f;
    tr = std::max(-1.f, std::min(1.f, tr));
    float th = acosf(tr);
    float s = th > 1e-6f ? th / (2.f * sinf(th)) : 0.5f;
    w[0] = s * (R[7] - R[5]);
    w[1] = s * (R[2] - R[6]);
    w[2] = s * (R[3] - R[1]);
}

// R = Rz(roll)*Ry(yaw)*Rx(pitch) 拆解（度）
static void yprFromR(const float R[9], float* ypr) {
    ypr[1] = atan2f(-R[6], sqrtf(R[7]*R[7] + R[8]*R[8])) * 57.2958f;  // yaw
    ypr[0] = atan2f(R[7], R[8]) * 57.2958f;                            // pitch
    ypr[2] = atan2f(R[3], R[0]) * 57.2958f;                            // roll
}

// obs/model: n 组 2D-3D 对应点（直立图像素）；输出 R(模型->相机)、t、欧拉角(yaw,pitch,roll)度。
// 近共面点存在 PnP 双解模糊：用 frontal/±30° 三组初始俯仰各跑一次 GN，取重投影误差最小者。
static bool solveHeadPose(const float obs[][2], const float (*model)[3], int n,
                          float eyeSpan, int rw, int rh,
                          float R[9], float t[3], float* ypr,
                          const float* prevW = nullptr)
{
    const float fx = (float)rw, fy = (float)rh, cx = rw * 0.5f, cy = rh * 0.5f;
    float ex = obs[1][0] - obs[0][0], ey = obs[1][1] - obs[0][1];
    float eyeD = sqrtf(ex*ex + ey*ey);
    if (eyeD < 8.f) return false;

    float roll0 = atan2f(ey, ex);
    float Z = fx * eyeSpan / eyeD;
    float mu = 0, mv = 0;
    for (int i = 0; i < n; i++) { mu += obs[i][0]; mv += obs[i][1]; }
    mu /= n; mv /= n;
    float mc[3] = {0.f, 0.f, 0.f};
    for (int i = 0; i < n; i++)
        for (int q = 0; q < 3; q++) mc[q] += model[i][q];
    for (int q = 0; q < 3; q++) mc[q] /= n;

    bool anyOk = false;
    float bestRms = 1e30f;
    const float initPitch[3] = { 0.f, 0.5236f, -0.5236f };   // 0 / ±30°
    for (int k = prevW ? -1 : 0; k < 3; k++) {
        float w2[3];
        if (k < 0) { w2[0] = prevW[0]; w2[1] = prevW[1]; w2[2] = prevW[2]; }
        else { w2[0] = initPitch[k]; w2[1] = 0.f; w2[2] = roll0; }
        float R2[9], t2[3];
        rodrigues(w2, R2);
        float Rc[3] = { R2[0]*mc[0] + R2[1]*mc[1] + R2[2]*mc[2],
                        R2[3]*mc[0] + R2[4]*mc[1] + R2[5]*mc[2],
                        R2[6]*mc[0] + R2[7]*mc[1] + R2[8]*mc[2] };
        t2[0] = (mu - cx) * Z / fx - Rc[0];
        t2[1] = (mv - cy) * Z / fy - Rc[1];
        t2[2] = Z - Rc[2];

        bool ok = true;
        for (int it = 0; it < 15 && ok; it++) {
            float JTJ[36], JTr[6];
            for (int i = 0; i < 36; i++) JTJ[i] = 0.f;
            for (int i = 0; i < 6; i++) JTr[i] = 0.f;
            for (int i = 0; i < n; i++) {
                float X[3] = { R2[0]*model[i][0] + R2[1]*model[i][1] + R2[2]*model[i][2] + t2[0],
                               R2[3]*model[i][0] + R2[4]*model[i][1] + R2[5]*model[i][2] + t2[1],
                               R2[6]*model[i][0] + R2[7]*model[i][1] + R2[8]*model[i][2] + t2[2] };
                if (X[2] < 1.f) { ok = false; break; }
                float x = X[0], y = X[1], z = X[2], iz2 = 1.f / (z*z);
                float u = cx + fx*x/z, v = cy + fy*y/z;
                float rows[2][6] = {
                    { fx*(-x*y)*iz2,      fx*(z*z + x*x)*iz2, fx*(-y*z)*iz2, fx/z, 0.f,  -fx*x*iz2 },
                    { fy*(-(z*z + y*y))*iz2, fy*(x*y)*iz2,    fy*(x*z)*iz2,  0.f, fy/z, -fy*y*iz2 }
                };
                const float Jr[2] = { -(u - obs[i][0]), -(v - obs[i][1]) };
                for (int a = 0; a < 6; a++) {
                    JTr[a] += rows[0][a]*Jr[0] + rows[1][a]*Jr[1];
                    for (int b = 0; b < 6; b++) JTJ[a*6+b] += rows[0][a]*rows[0][b] + rows[1][a]*rows[1][b];
                }
            }
            if (!ok) break;
            float lm = 1e-4f;
            for (int a = 0; a < 6; a++) {
                float mx = 1e-9f;
                for (int b = 0; b < 6; b++) mx = std::max(mx, fabsf(JTJ[a*6+b]));
                JTJ[a*6+a] += lm*mx + 1e-9f;
            }
            // 高斯消元（部分主元）解 (JTJ) d = JTr
            float A6[6][7];
            for (int a = 0; a < 6; a++) { for (int b = 0; b < 6; b++) A6[a][b] = JTJ[a*6+b]; A6[a][6] = JTr[a]; }
            for (int col = 0; col < 6; col++) {
                int piv = col;
                for (int r2 = col + 1; r2 < 6; r2++)
                    if (fabsf(A6[r2][col]) > fabsf(A6[piv][col])) piv = r2;
                if (fabsf(A6[piv][col]) < 1e-12f) { ok = false; break; }
                if (piv != col) for (int b = 0; b < 7; b++) std::swap(A6[col][b], A6[piv][b]);
                for (int r2 = 0; r2 < 6; r2++) {
                    if (r2 == col) continue;
                    float fk = A6[r2][col] / A6[col][col];
                    for (int b = col; b < 7; b++) A6[r2][b] -= fk * A6[col][b];
                }
            }
            if (!ok) break;
            float d[6];
            for (int a = 0; a < 6; a++) d[a] = A6[a][6] / A6[a][a];
            w2[0] += d[0]; w2[1] += d[1]; w2[2] += d[2];
            t2[0] += d[3]; t2[1] += d[4]; t2[2] += d[5];
            rodrigues(w2, R2);
            float dn = 0;
            for (int a = 0; a < 6; a++) dn += d[a]*d[a];
            if (dn < 1e-10f) break;
        }
        if (!ok) continue;
        float ss = 0.f;
        for (int i = 0; i < n; i++) {
            float X[3] = { R2[0]*model[i][0] + R2[1]*model[i][1] + R2[2]*model[i][2] + t2[0],
                           R2[3]*model[i][0] + R2[4]*model[i][1] + R2[5]*model[i][2] + t2[1],
                           R2[6]*model[i][0] + R2[7]*model[i][1] + R2[8]*model[i][2] + t2[2] };
            float du = cx + fx*X[0]/X[2] - obs[i][0];
            float dv = cy + fy*X[1]/X[2] - obs[i][1];
            ss += du*du + dv*dv;
        }
        float rms = sqrtf(ss / n);
        if (rms < bestRms) {
            bestRms = rms;
            for (int a = 0; a < 9; a++) R[a] = R2[a];
            for (int a = 0; a < 3; a++) t[a] = t2[a];
            anyOk = true;
        }
        if (prevW && k < 0 && rms < 8.f) break;
    }
    if (!anyOk) return false;
    yprFromR(R, ypr);
    return true;
}

// 投影面部法线（朝向）：起点=鼻尖（模型原点），终点=原点 + R·(0,0,-L)
static void projectNormal(const float R[9], const float t[3], int rw, int rh,
                          float nrm[2][2])
{
    const float fx = (float)rw, fy = (float)rh, cx = rw * 0.5f, cy = rh * 0.5f;
    const float L = 450.f;
    float pts[2][3] = {
        { t[0], t[1], t[2] },
        { -R[2]*L + t[0], -R[5]*L + t[1], -R[8]*L + t[2] },
    };
    for (int a = 0; a < 2; a++) {
        nrm[a][0] = cx + fx * pts[a][0] / pts[a][2];
        nrm[a][1] = cy + fy * pts[a][1] / pts[a][2];
    }
}

// ---- facemesh：MediaPipe FaceMesh 468 点：方框裁剪 192² RGB (x-127.5)/127.5 → 468 点 ----
static bool facemeshRun(const unsigned char* bgr, int iw, int ih,
                        const float box[4], float* obs468, float* obs468z)
{
    if (!g_fmNet) return false;
    float bw = box[2] - box[0], bh = box[3] - box[1];
    float cx = (box[0] + box[2]) * 0.5f, cy = (box[1] + box[3]) * 0.5f;
    float side = std::max(bw, bh);
    float x1 = cx - side / 2, y1 = cy - side / 2;
    static std::vector<unsigned char> crop;   // g_mutex 串行，复用缓冲
    crop.resize(kFmCrop * kFmCrop * 3);
    for (int j = 0; j < kFmCrop; j++) {
        int sy = std::min(ih - 1, std::max(0, (int)(y1 + (j + 0.5f) * side / kFmCrop)));
        for (int i = 0; i < kFmCrop; i++) {
            int sx = std::min(iw - 1, std::max(0, (int)(x1 + (i + 0.5f) * side / kFmCrop)));
            const unsigned char* s = bgr + ((size_t)sy * iw + sx) * 3;
            unsigned char* d = &crop[(j * kFmCrop + i) * 3];
            d[0] = s[0]; d[1] = s[1]; d[2] = s[2];
        }
    }
    ncnn::Mat in = ncnn::Mat::from_pixels(crop.data(), ncnn::Mat::PIXEL_RGB, kFmCrop, kFmCrop);
    const float means[3] = { 127.5f, 127.5f, 127.5f };
    const float norms[3] = { 1 / 127.5f, 1 / 127.5f, 1 / 127.5f };
    in.substract_mean_normalize(means, norms);
    ncnn::Extractor ex = g_fmNet->create_extractor();
    ex.input("in0", in);
    ncnn::Mat out;
    if (ex.extract("out0", out) != 0) return false;
    if (out.w * out.h * out.c < kFmPoints * 3) return false;
    const float kScale = side / (float)kFmCrop;
    for (int i = 0; i < kFmPoints; i++) {
        obs468[2*i]   = out[3*i]   * kScale + x1;
        obs468[2*i+1] = out[3*i+1] * kScale + y1;
        obs468z[i]    = out[3*i+2] * kScale;
    }
    return true;
}

// ---- mesh 9 点 Kabsch 闭式姿态（Horn 四元数）：z 翻转到相机坐标系 →
// 中心化 + 按眼角距归一 → Horn 闭式解（无迭代、无分支翻转）
static const float kMeshTpl9[9*3] = {
#include "mesh_template9.inc"
};
static const int kMeshIdx9[9] = {10, 1, 152, 33, 263, 61, 291, 234, 454};


// 4x4 对称阵 Jacobi 特征分解：返回最大特征值的单位特征向量
static void jacobiMaxEig4(double N[4][4], double* q)
{
    for (int i = 0; i < 4; i++) q[i] = 0;
    double a[4][4];
    double v[4][4] = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) a[i][j] = N[i][j];
    for (int sweep = 0; sweep < 24; sweep++) {
        double off = 0;
        for (int i = 0; i < 4; i++) for (int j = i+1; j < 4; j++) off += a[i][j]*a[i][j];
        if (off < 1e-22) break;
        for (int p = 0; p < 3; p++) {
            for (int qq = p+1; qq < 4; qq++) {
                if (fabs(a[p][qq]) < 1e-15) continue;
                double theta = 0.5 * atan2(2*a[p][qq], a[qq][qq]-a[p][p]);
                double c = cos(theta), s = sin(theta);
                for (int i = 0; i < 4; i++) {
                    double aip = a[i][p], aiq = a[i][qq];
                    a[i][p] = c*aip - s*aiq; a[i][qq] = s*aip + c*aiq;
                }
                for (int j = 0; j < 4; j++) {
                    double apj = a[p][j], aqj = a[qq][j];
                    a[p][j] = c*apj - s*aqj; a[qq][j] = s*apj + c*aqj;
                }
                for (int i = 0; i < 4; i++) {
                    double vip = v[i][p], viq = v[i][qq];
                    v[i][p] = c*vip - s*viq; v[i][qq] = s*vip + c*viq;
                }
            }
        }
    }
    int best = 0; double bestv = -1e30;
    for (int i = 0; i < 4; i++) if (a[i][i] > bestv) { bestv = a[i][i]; best = i; }
    for (int i = 0; i < 4; i++) q[i] = v[i][best];
    double nq = sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    if (nq < 1e-12) { q[0] = 1; return; }
    for (int i = 0; i < 4; i++) q[i] /= nq;
}

static bool meshProcrustesR(const float* obs468, const float* obs468z, float R[9])
{
    double P[9*3];   // 观测：z 翻转到相机坐标系（z 进场景，鼻尖最近）
    for (int k = 0; k < 9; k++) {
        int li = kMeshIdx9[k];
        P[3*k]   =  obs468[2*li];
        P[3*k+1] =  obs468[2*li+1];
        P[3*k+2] = -obs468z[li];
    }
    // 中心化 + 按观测外眼角距(点3-点4)归一（与模板构建一致）
    double cP[3] = {0,0,0}, cT[3] = {0,0,0};
    for (int k = 0; k < 9; k++)
        for (int q = 0; q < 3; q++) { cP[q] += P[3*k+q]; cT[q] += kMeshTpl9[3*k+q]; }
    for (int q = 0; q < 3; q++) { cP[q] /= 9; cT[q] /= 9; }
    double ex = P[3*3]-P[3*4], ey = P[3*3+1]-P[3*4+1], ez = P[3*3+2]-P[3*4+2];
    double sP = sqrt(ex*ex+ey*ey+ez*ez) + 1e-9;
    double X[9*3], C[9*3];
    for (int k = 0; k < 9; k++)
        for (int q = 0; q < 3; q++) {
            X[3*k+q] = (P[3*k+q]-cP[q]) / sP;
            C[3*k+q] = kMeshTpl9[3*k+q] - cT[q];
        }
    // Horn 闭式：M = Σ x_k c_k^T（x=观测, c=模板），x ≈ R c
    double Mxx=0,Mxy=0,Mxz=0,Myx=0,Myy=0,Myz=0,Mzx=0,Mzy=0,Mzz=0;
    for (int k = 0; k < 9; k++) {
        double x[3] = { X[3*k], X[3*k+1], X[3*k+2] };
        double c[3] = { C[3*k], C[3*k+1], C[3*k+2] };
        Mxx += x[0]*c[0]; Mxy += x[0]*c[1]; Mxz += x[0]*c[2];
        Myx += x[1]*c[0]; Myy += x[1]*c[1]; Myz += x[1]*c[2];
        Mzx += x[2]*c[0]; Mzy += x[2]*c[1]; Mzz += x[2]*c[2];
    }
    double sigma = Mxx + Myy + Mzz;
    double ax = Myz - Mzy, ay = Mzx - Mxy, az = Mxy - Myx;
    double N[4][4] = {
        { sigma,   ax,   ay,   az },
        {    ax, Mxx - Myy - Mzz, Mxy + Myx, Mzx + Mxz },
        {    ay, Mxy + Myx, -Mxx + Myy - Mzz, Myz + Mzy },
        {    az, Mzx + Mxz, Myz + Mzy, -Mxx - Myy + Mzz } };
    double q[4];
    jacobiMaxEig4(N, q);
    double q0 = q[0], qx = q[1], qy = q[2], qz = q[3];
    R[0] = (float)(q0*q0 + qx*qx - qy*qy - qz*qz);
    R[1] = (float)(2*(qx*qy - q0*qz));
    R[2] = (float)(2*(qx*qz + q0*qy));
    R[3] = (float)(2*(qx*qy + q0*qz));
    R[4] = (float)(q0*q0 - qx*qx + qy*qy - qz*qz);
    R[5] = (float)(2*(qy*qz - q0*qx));
    R[6] = (float)(2*(qx*qz - q0*qy));
    R[7] = (float)(2*(qy*qz + q0*qx));
    R[8] = (float)(q0*q0 - qx*qx - qy*qy + qz*qz);
    return true;
}

// ---- hopenet：方框裁剪 -> 48x48 灰度（无归一化）-> 3x66 bins 期望角 ----
static bool hopenetRun(const unsigned char* bgr, int iw, int ih,
                       const float box[4], float* yaw, float* pitch, float* roll)
{
    if (!g_hopeNet) return false;
    float bw = box[2] - box[0], bh = box[3] - box[1];
    float cx = (box[0] + box[2]) * 0.5f, cy = (box[1] + box[3]) * 0.5f;
    float side = std::max(bw, bh);
    float x1 = cx - side / 2, y1 = cy - side / 2;
    static std::vector<unsigned char> gray;   // g_mutex 串行，复用缓冲
    gray.resize(48 * 48);
    for (int j = 0; j < 48; j++) {
        int sy = std::min(ih - 1, std::max(0, (int)(y1 + (j + 0.5f) * side / 48.f)));
        for (int i = 0; i < 48; i++) {
            int sx = std::min(iw - 1, std::max(0, (int)(x1 + (i + 0.5f) * side / 48.f)));
            const unsigned char* p = bgr + ((size_t)sy * iw + sx) * 3;
            gray[j * 48 + i] = (unsigned char)((p[0] * 299 + p[1] * 587 + p[2] * 114) / 1000);
        }
    }
    ncnn::Mat in = ncnn::Mat::from_pixels(gray.data(), ncnn::Mat::PIXEL_GRAY, 48, 48);
    ncnn::Extractor ex = g_hopeNet->create_extractor();
    ex.input("data", in);
    ncnn::Mat out;
    if (ex.extract("hybridsequential0_multitask0_dense0_fwd", out) != 0) return false;
    if (out.w * out.h * out.c < 3 * 66) return false;
    const float* o = out;
    auto soft = [](float* z, int n) {
        float mx = -1e30f;
        for (int i = 0; i < n; i++) mx = std::max(mx, z[i]);
        float s = 0;
        for (int i = 0; i < n; i++) { z[i] = expf(z[i] - mx); s += z[i]; }
        for (int i = 0; i < n; i++) z[i] /= s;
    };
    float buf[66];
    double ang[3];
    for (int a = 0; a < 3; a++) {
        for (int i = 0; i < 66; i++) buf[i] = o[a * 66 + i];
        soft(buf, 66);
        double e = 0;
        for (int i = 0; i < 66; i++) e += (i + 1) * buf[i];
        ang[a] = e * 3.0 - 99.0;
    }
    // hopenet 的 pitch 正方向与本项目约定相反，实测同帧对比确认后翻转
    *pitch = -ang[0]; *roll = (float)ang[1]; *yaw = (float)ang[2];
    return true;
}

// 欧拉角 -> R（约定与 yprFromR 一致：R = Rz(roll)*Ry(yaw)*Rx(pitch)）
static void eulerToR(float yawDeg, float pitchDeg, float rollDeg, float R[9])
{
    float y = yawDeg * 0.01745f, p = pitchDeg * 0.01745f, r = rollDeg * 0.01745f;
    float cy = cosf(y), sy = sinf(y), cp = cosf(p), sp = sinf(p), cr = cosf(r), sr = sinf(r);
    R[0] =  cr*cy;  R[1] = -sr*cp + cr*sy*sp;  R[2] =  sr*sp + cr*sy*cp;
    R[3] =  sr*cy;  R[4] =  cr*cp + sr*sy*sp;  R[5] = -cr*sp + sr*sy*cp;
    R[6] = -sy;     R[7] =  cy*sp;             R[8] =  cy*cp;
}

// ---- 姿态网加载（facemesh + hopenet）：真机上薄层小网走 Vulkan FP16，失败回退 CPU 稳健配置 ----
struct GpuVariant {
    const char* name;
    bool fp16p, fp16s, fp16a, slm, cm, wino;
};
static ncnn::Net* buildGpuNet(const std::string& param, const std::vector<unsigned char>& bin,
                              const GpuVariant& v, std::string* err);

static void ensurePoseNets(AAssetManager* am)
{
    static std::once_flag once;
    std::call_once(once, [am] {
        if (!am) return;
        if (ncnn::get_gpu_count() > 0) {
            static std::once_flag vkOnce;
            std::call_once(vkOnce, [] { g_vk = ncnn::get_gpu_device(0); });
        }
        // 加载函数：优先 Vulkan FP16，失败回退 CPU 稳健配置（fp32+单线程+普通卷积）。
        // 真机（天玑 ARM 内核对 pnnx 模型）确定性 SIGSEGV 的规避与蓝本一致。
        auto loadVkOrSafeCpu = [&](const char* tag, const char* pp, const char* bp, ncnn::Net** out) {
            std::vector<unsigned char> p = readAsset(am, pp);
            std::vector<unsigned char> b = readAsset(am, bp);
            if (p.empty() || b.empty()) { LOGE("%s assets missing", tag); return; }
            std::string param((const char*)p.data(), p.size());
            param.push_back('\0');
            if (g_vk) {
                ncnn::Net* net = new ncnn::Net;
                net->opt.lightmode = true;
                net->opt.num_threads = 1;
                net->opt.use_winograd_convolution = true;
                net->opt.use_sgemm_convolution = true;
                net->opt.use_int8_inference = false;
                net->opt.use_bf16_storage = false;
                net->opt.use_vulkan_compute = true;
                net->opt.use_packing_layout = true;
                net->opt.use_fp16_packed = true;
                net->opt.use_fp16_storage = true;
                net->opt.use_fp16_arithmetic = true;
                net->opt.use_shader_local_memory = false;
                net->opt.use_cooperative_matrix = false;
                ncnn::VkBlobAllocator* vb = new ncnn::VkBlobAllocator(g_vk);
                ncnn::VkStagingAllocator* vs = new ncnn::VkStagingAllocator(g_vk);
                net->opt.blob_vkallocator = vb;
                net->opt.workspace_vkallocator = vb;
                net->opt.staging_vkallocator = vs;
                net->set_vulkan_device(g_vk);
                if (net->load_param_mem(param.c_str()) == 0) {
                    const unsigned char* mem = b.data();
                    ncnn::DataReaderFromMemory dr(mem);
                    if (net->load_model(dr) == 0) {
                        *out = net;
                        LOGI("%s net on Vulkan", tag);
                        return;
                    }
                }
                delete net;
                LOGE("%s vulkan load failed, fallback cpu", tag);
            }
            // CPU 稳健回退
            ncnn::Net* net = new ncnn::Net;
            net->opt.lightmode = true;
            net->opt.num_threads = 1;
            net->opt.use_fp16_storage = false;
            net->opt.use_fp16_packed = false;
            net->opt.use_fp16_arithmetic = false;
            net->opt.use_sgemm_convolution = false;
            net->opt.use_winograd_convolution = false;
            if (net->load_param_mem(param.c_str()) != 0) { delete net; LOGE("%s load_param failed", tag); return; }
            const unsigned char* mem = b.data();
            ncnn::DataReaderFromMemory dr(mem);
            if (net->load_model(dr) != 0) { delete net; LOGE("%s load_model failed", tag); return; }
            *out = net;
            LOGI("%s net on safe cpu", tag);
        };
        loadVkOrSafeCpu("hopenet", kHopParam, kHopBin, &g_hopeNet);
        loadVkOrSafeCpu("facemesh", kFmParam, kFmBin, &g_fmNet);
    });
}

// ---- 检测网 GPU 自动调优：正确性筛选 + 速度择优，结果持久化（蓝本同款） ----
static const GpuVariant kGpuVariants[] = {
    {"fp16",       true, true, true,   true,  true,  true },
    {"fp16s",      true, true, false,  true,  true,  true },
    {"fp16p",      true, false, false, true,  true,  true },
    {"fp32",       false, false, false, true,  true,  true },
    {"fp16-sl0",   true, true, true,   false, false, true },
    {"fp16-wino0", true, true, true,   true,  true,  false },
};
static const int kGpuVariantCount = (int)(sizeof(kGpuVariants) / sizeof(kGpuVariants[0]));
static int g_gpuVariant = -1;
static std::string s_filesDir;

static int readGpuVariantCache() {
    if (s_filesDir.empty()) return -1;
    FILE* f = fopen((s_filesDir + "/gpu_variant.txt").c_str(), "r");
    if (!f) return -1;
    int v = -1;
    if (fscanf(f, "%d", &v) != 1) v = -1;
    fclose(f);
    if (v < 0 || v >= kGpuVariantCount) return -1;
    return v;
}
static void writeGpuVariantCache(int v) {
    if (s_filesDir.empty()) return;
    FILE* f = fopen((s_filesDir + "/gpu_variant.txt").c_str(), "w");
    if (!f) return;
    fprintf(f, "%d", v);
    fclose(f);
}

static bool forwardDet(ncnn::Net* net, const ncnn::Mat& in, int warmup, int timed,
                       std::vector<nbench::FlatOut>* outs, double* avgMs)
{
    for (int i = 0; i < warmup; i++) {
        ncnn::Extractor ex = net->create_extractor();
        ex.input(kDetInput, in);
        ncnn::Mat t;
        for (int s = 0; s < 3; s++)
            if (ex.extract(kScoreBlob[s], t) != 0 || ex.extract(kBoxBlob[s], t) != 0 ||
                ex.extract(kKpsBlob[s], t) != 0) return false;
    }
    double t1 = nowMs();
    for (int i = 0; i < timed; i++) {
        ncnn::Extractor ex = net->create_extractor();
        ex.input(kDetInput, in);
        for (int s = 0; s < 3; s++) {
            ncnn::Mat m;
            if (ex.extract(kScoreBlob[s], m) != 0 || ex.extract(kBoxBlob[s], m) != 0 ||
                ex.extract(kKpsBlob[s], m) != 0) return false;
            if (outs && i == 0) outs->push_back(nbench::flatten(m));
        }
    }
    *avgMs = (nowMs() - t1) / std::max(1, timed);
    return true;
}

static bool cpuReference(const std::string& param, const std::vector<unsigned char>& bin,
                         const ncnn::Mat& in, std::vector<nbench::FlatOut>* ref)
{
    ncnn::UnlockedPoolAllocator blob, ws;
    ncnn::Net net;
    net.opt.lightmode = true;
    int nt = ncnn::get_physical_big_cpu_count();
    net.opt.num_threads = nt > 0 ? nt : 4;
    net.opt.blob_allocator = &blob;
    net.opt.workspace_allocator = &ws;
    net.opt.use_winograd_convolution = true;
    net.opt.use_sgemm_convolution = true;
    net.opt.use_int8_inference = false;
    net.opt.use_vulkan_compute = false;
    net.opt.use_bf16_storage = false;
    if (net.load_param_mem(param.c_str()) != 0) return false;
    const unsigned char* mem = bin.data();
    ncnn::DataReaderFromMemory dr(mem);
    if (net.load_model(dr) != 0) return false;
    ncnn::Extractor ex = net.create_extractor();
    ex.input(kDetInput, in);
    for (int s = 0; s < 3; s++) {
        ncnn::Mat m;
        if (ex.extract(kScoreBlob[s], m) != 0 || ex.extract(kBoxBlob[s], m) != 0 ||
            ex.extract(kKpsBlob[s], m) != 0) return false;
        ref->push_back(nbench::flatten(m));
    }
    return true;
}

static ncnn::Net* buildGpuNet(const std::string& param, const std::vector<unsigned char>& bin,
                              const GpuVariant& v, std::string* err)
{
    ncnn::Net* net = new ncnn::Net;
    net->opt.lightmode = true;
    net->opt.num_threads = 1;
    net->opt.use_winograd_convolution = v.wino;
    net->opt.use_sgemm_convolution = true;
    net->opt.use_int8_inference = false;
    net->opt.use_bf16_storage = false;
    net->opt.use_vulkan_compute = true;
    net->opt.use_packing_layout = true;
    net->opt.use_fp16_packed = v.fp16p;
    net->opt.use_fp16_storage = v.fp16s;
    net->opt.use_fp16_arithmetic = v.fp16a;
    net->opt.use_shader_local_memory = v.slm;
    net->opt.use_cooperative_matrix = v.cm;
    ncnn::VkBlobAllocator* vb = new ncnn::VkBlobAllocator(g_vk);
    ncnn::VkStagingAllocator* vs = new ncnn::VkStagingAllocator(g_vk);
    net->opt.blob_vkallocator = vb;
    net->opt.workspace_vkallocator = vb;
    net->opt.staging_vkallocator = vs;
    net->set_vulkan_device(g_vk);
    if (net->load_param_mem(param.c_str()) != 0) { *err = "load_param"; delete net; return nullptr; }
    const unsigned char* mem = bin.data();
    ncnn::DataReaderFromMemory dr(mem);
    if (net->load_model(dr) != 0) { *err = "load_model"; delete net; return nullptr; }
    return net;
}

// GPU 后端入口：自动调优并发布
static jstring openGpuAuto(JNIEnv* env, const std::string& param,
                           const std::vector<unsigned char>& bin, jint backend)
{
    const int gcnt = ncnn::get_gpu_count();
    if (gcnt <= 0) return env->NewStringUTF("{\"ok\":false,\"err\":\"no_vulkan\"}");
    static std::once_flag vkOnce;
    std::call_once(vkOnce, [] { g_vk = ncnn::get_gpu_device(0); });
    if (!g_vk) return env->NewStringUTF("{\"ok\":false,\"err\":\"no_vulkan_device\"}");

    // 调优输入：基准风格伪随机 448x448（内容不影响时序）
    ncnn::Mat in(kIn, kIn, 3);
    {
        uint32_t st = 0x2545F491u;
        for (int q = 0; q < 3; q++) {
            float* p = (float*)in.channel(q);
            for (int i = 0; i < kIn * kIn; i++) {
                st = st * 1664525u + 1013904223u;
                p[i] = 2.0f * (float)(st >> 8 & 0xffffff) / 16777215.0f - 1.0f;
            }
        }
    }

    auto publish = [&](ncnn::Net* net, int variantIdx, double ms, double maxAbs) -> jstring {
        std::lock_guard<std::mutex> lk(g_mutex);
        if (g_shutdown) { delete net; return env->NewStringUTF("{\"ok\":false,\"err\":\"closed\"}"); }
        delete g_net;
        g_net = net;
        g_backend = backend;
        g_input = kIn;
        g_threads = 1;
        g_gpuOk = true;
        g_gpuVariant = variantIdx;
        char mb[32];
        snprintf(mb, sizeof(mb), "%.3g", maxAbs);
        std::string j = std::string("{\"ok\":true,\"backend\":\"") + kBackendName[backend]
                      + "\",\"variant\":\"" + kGpuVariants[variantIdx].name + "\""
                      + ",\"gpuMs\":" + f2(ms) + ",\"gpuMaxAbs\":\"" + mb + "\""
                      + ",\"threads\":1,\"input\":" + std::to_string(kIn) + "}";
        LOGI("det net open(GPU): backend=%s variant=%s avg=%.2fms maxAbs=%s",
             kBackendName[backend], kGpuVariants[variantIdx].name, ms, mb);
        return env->NewStringUTF(j.c_str());
    };

    std::vector<nbench::FlatOut> ref;
    if (!cpuReference(param, bin, in, &ref))
        return env->NewStringUTF("{\"ok\":false,\"err\":\"cpu_ref_failed\"}");

    int chosen = g_gpuVariant;
    if (chosen < 0) chosen = readGpuVariantCache();
    if (chosen >= 0) {
        std::string err;
        ncnn::Net* net = buildGpuNet(param, bin, kGpuVariants[chosen], &err);
        if (net) {
            std::vector<nbench::FlatOut> outs;
            double ms = 0;
            if (forwardDet(net, in, 2, 4, &outs, &ms)) {
                nbench::OutDiff d = nbench::compareOutputs(ref, outs);
                LOGI("gpu tune cached variant=%s avg=%.2fms maxAbs=%.4g",
                     kGpuVariants[chosen].name, ms, d.maxAbs);
                if (d.maxAbs < 0.3f && d.badDiffs == 0)
                    return publish(net, chosen, ms, d.maxAbs);
            }
            delete net;
        }
        g_gpuVariant = -1;
    }

    double bestMs = 1e30f;
    int bestIdx = -1;
    double bestAbs = 0;
    for (int v = 0; v < kGpuVariantCount; v++) {
        {
            std::lock_guard<std::mutex> lk(g_mutex);
            if (g_shutdown) return env->NewStringUTF("{\"ok\":false,\"err\":\"closed\"}");
        }
        std::string err;
        ncnn::Net* net = buildGpuNet(param, bin, kGpuVariants[v], &err);
        if (!net) { LOGI("gpu tune %s: %s", kGpuVariants[v].name, err.c_str()); continue; }
        std::vector<nbench::FlatOut> outs;
        double ms = 0;
        if (!forwardDet(net, in, 2, 4, &outs, &ms)) {
            LOGI("gpu tune %s: forward failed", kGpuVariants[v].name);
            delete net;
            continue;
        }
        nbench::OutDiff d = nbench::compareOutputs(ref, outs);
        LOGI("gpu tune %s: avg=%.2fms maxAbs=%.4g badDiffs=%ld",
             kGpuVariants[v].name, ms, d.maxAbs, d.badDiffs);
        delete net;
        if (d.maxAbs > 0.2f || d.badDiffs > 0) continue;
        if (ms < bestMs) { bestMs = ms; bestIdx = v; bestAbs = d.maxAbs; }
    }
    if (bestIdx < 0) return env->NewStringUTF("{\"ok\":false,\"err\":\"vulkan_inaccurate\"}");
    writeGpuVariantCache(bestIdx);

    std::string err;
    ncnn::Net* net = buildGpuNet(param, bin, kGpuVariants[bestIdx], &err);
    if (!net) return env->NewStringUTF("{\"ok\":false,\"err\":\"load_model\"}");
    double ms = 0;
    if (!forwardDet(net, in, 2, 3, nullptr, &ms)) {
        delete net;
        return env->NewStringUTF("{\"ok\":false,\"err\":\"warmup_failed\"}");
    }
    return publish(net, bestIdx, ms, bestAbs);
}

}  // namespace

// AR 姿态轨迹状态（文件域：nativeFaceOpen/Close 跨会话复位）。
// 轨迹关联按脸中心最近邻；每脸独立 One-Euro 状态。
struct PoseTrack {
    bool has = false;
    float nose[2] = {0, 0};
    float w[3] = {0, 0, 0};        // 旋转矢量 One-Euro 状态
    float xPrev[3] = {0, 0, 0};
    float dPrev[3] = {0, 0, 0};
    bool hasEuro[3] = {false, false, false};
    bool hasW = false;
    bool hasT = false;             // 平移 One-Euro 状态
    float tPrev[3] = {0, 0, 0};
    float tDPrev[3] = {0, 0, 0};
    bool hasTD[3] = {false, false, false};
    bool hasO = false;             // overlay 像素 One-Euro：ovC(2) + 四角相对偏移(8)
    float ovPrev[10] = {0,0,0,0,0,0,0,0,0,0};
    float ovDP[10] = {0,0,0,0,0,0,0,0,0,0};
    bool hasOE[10] = {false,false,false,false,false,false,false,false,false,false};
    bool hasDim = false;           // 平滑框尺寸状态（overlay 屏尺寸稳定用）
    float bwS = 0, bhS = 0;
    bool hasEyeD = false;          // eyeD EWMA：kps 仅检测帧刷新，不平滑会周期阶跃
    float eyeDS = 0;
    bool hasR1 = false;            // 上一帧面内 x 轴（yaw≈±90° 投影退化时冻结防抖）
    float r1L[3] = {1.f, 0.f, 0.f};
    bool hopeOn = false;           // hopenet 接入滞回（38 开/30 关），防边界反复开关
    float rawW = 0;                // 未膨胀 mesh 包围盒宽 EWMA（板尺寸用，与 ROI 裕量解耦）
    float eyeDX = 1.f, eyeDY = 0.f;  // 观测眼线 33→263（相机系，板横轴用）
    bool hasRollE = false;           // 眼线角 One-Euro（板 roll 阻尼）
    float rollPrev = 0, rollDPrev = 0;
    int miss = 0;                    // 连续未被任何检测关联的帧数（轨迹寿命）
};
static PoseTrack s_tracks[3];

// SCRFD 节流：每 kDetEvery 帧跑一次全图检测，其余帧复用缓存框驱动 facemesh；
// mesh 成功后用 468 点包围盒回写缓存框（跟随人脸移动），检测帧全量校正。
static std::vector<Face> s_cached;
static int s_frm = 0;

static void tracksReset() {
    for (int i = 0; i < 3; i++) s_tracks[i] = PoseTrack();
    s_cached.clear();
    s_frm = 0;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_gscp_desktop_ArNative_nativeFaceOpen(JNIEnv* env, jobject, jobject assets,
                                              jstring filesDir, jint backend, jint input)
{
    if (filesDir) {
        const char* fd = env->GetStringUTFChars(filesDir, nullptr);
        if (fd) { s_filesDir = fd; env->ReleaseStringUTFChars(filesDir, fd); }
    }

    {
        std::lock_guard<std::mutex> lk(g_mutex);
        g_shutdown = false;
        tracksReset();   // 跨会话不继承轨迹/缓存（丢脸轨迹泄漏的会话级兜底）
        if (g_net && g_backend == backend) {
            std::string j = std::string("{\"ok\":true,\"backend\":\"") + kBackendName[backend]
                          + "\",\"threads\":" + std::to_string(g_threads) + ",\"cached\":true}";
            return env->NewStringUTF(j.c_str());
        }
    }

    AAssetManager* am = assets ? AAssetManager_fromJava(env, assets) : nullptr;
    ensurePoseNets(am);
    std::vector<unsigned char> p = readAsset(am, kDetParam);
    std::vector<unsigned char> b = readAsset(am, kDetBin);
    if (p.empty() || b.empty()) return env->NewStringUTF("{\"ok\":false,\"err\":\"assets_missing\"}");

    std::string param((const char*)p.data(), p.size());
    param.push_back('\0');

    if (backend == 2 || backend == 3) {
        return openGpuAuto(env, param, b, backend);
    }

    // CPU 后端（蓝本同款配置：fp16/packing 保持 ncnn 默认）
    int nt = ncnn::get_physical_big_cpu_count();
    if (nt <= 0) nt = ncnn::get_cpu_count();
    if (nt <= 0) nt = 4;
    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(nt);

    ncnn::Net* net = new ncnn::Net;
    ncnn::UnlockedPoolAllocator* nb = nullptr;
    ncnn::UnlockedPoolAllocator* nw = nullptr;
    net->opt.lightmode = true;
    net->opt.num_threads = nt;
    net->opt.use_winograd_convolution = true;
    net->opt.use_sgemm_convolution = true;
    net->opt.use_int8_inference = false;
    net->opt.use_vulkan_compute = false;
    net->opt.use_bf16_storage = false;

    if (backend == 1) {
        nb = new ncnn::UnlockedPoolAllocator;
        nw = new ncnn::UnlockedPoolAllocator;
        net->opt.blob_allocator = nb;
        net->opt.workspace_allocator = nw;
        net->opt.use_packing_layout = true;
        net->opt.use_fp16_packed = true;
        net->opt.use_fp16_storage = true;
        net->opt.use_fp16_arithmetic = true;
    } else {
        nb = new ncnn::UnlockedPoolAllocator;
        nw = new ncnn::UnlockedPoolAllocator;
        net->opt.blob_allocator = nb;
        net->opt.workspace_allocator = nw;
    }

    if (net->load_param_mem(param.c_str()) != 0) { delete net; if (nb) delete nb; if (nw) delete nw; return env->NewStringUTF("{\"ok\":false,\"err\":\"load_param\"}"); }
    const unsigned char* mem = b.data();
    ncnn::DataReaderFromMemory dr(mem);
    if (net->load_model(dr) != 0) { delete net; if (nb) delete nb; if (nw) delete nw; return env->NewStringUTF("{\"ok\":false,\"err\":\"load_model\"}"); }

    // 空跑两轮验证配置在本机可用
    {
        ncnn::Mat warm(kIn, kIn, 3);
        warm.fill(0.01f);
        for (int w2 = 0; w2 < 2; w2++) {
            ncnn::Extractor ex = net->create_extractor();
            ex.input(kDetInput, warm);
            for (int s = 0; s < 3; s++) {
                ncnn::Mat t;
                if (ex.extract(kScoreBlob[s], t) != 0 || ex.extract(kBoxBlob[s], t) != 0 ||
                    ex.extract(kKpsBlob[s], t) != 0) {
                    delete net;
                    if (nb) delete nb;
                    if (nw) delete nw;
                    return env->NewStringUTF("{\"ok\":false,\"err\":\"warmup_failed\"}");
                }
            }
        }
    }

    std::string j;
    {
        std::lock_guard<std::mutex> lk(g_mutex);
        if (g_shutdown) {
            delete net;
            if (nb) delete nb;
            if (nw) delete nw;
            return env->NewStringUTF("{\"ok\":false,\"err\":\"closed\"}");
        }
        delete g_net;
        g_net = net;
        g_backend = backend;
        g_input = kIn;
        g_threads = nt;
        j = std::string("{\"ok\":true,\"backend\":\"") + kBackendName[backend]
          + "\",\"threads\":" + std::to_string(g_threads)
          + ",\"input\":" + std::to_string(kIn) + "}";
    }

    LOGI("det net open: backend=%s input=%d threads=%d", kBackendName[backend], kIn, g_threads);
    return env->NewStringUTF(j.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_gscp_desktop_ArNative_nativeFaceClose(JNIEnv*, jobject)
{
    std::lock_guard<std::mutex> lk(g_mutex);
    g_shutdown = true;
    tracksReset();
    delete g_net;
    g_net = nullptr;
    g_backend = -1;
    g_input = 0;
}

extern "C" JNIEXPORT void JNICALL
Java_com_gscp_desktop_ArNative_nativeFaceSetPoseAlgo(JNIEnv*, jobject, jint algo)
{
    std::lock_guard<std::mutex> lk(g_mutex);
    // 3=融合（主路径） 1=hopenet（对照）；其余一律回融合
    g_poseAlgo = (algo == 1) ? 1 : 3;
}

/**
 * 一帧处理（蓝本同款）：YUV -> BGR -> 旋转 -> RGBA（显示）+ SCRFD 检测 + facemesh
 * 融合姿态。返回 JSON：
 * {"ok","backend","convMs","ms","w","h","rot","faces":[
 *   {"box":[x1,y1,x2,y2],"score","kps":[x,y x5],"pose":[p,y,r],
 *    "normal":[[起],[终]],"overlay":[[x,y]x4],"ovC":[x,y],"ovD":cm}]}
 */
extern "C" JNIEXPORT void JNICALL
Java_com_gscp_desktop_ArNative_nativeSetOverlaySize(
    JNIEnv*, jobject, jint w, jint h)
{
    std::lock_guard<std::mutex> lk(g_mutex);
    g_ovW = w; g_ovH = h;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_gscp_desktop_ArNative_nativeFaceDetect(
    JNIEnv* env, jobject, jbyteArray yArr, jbyteArray uArr, jbyteArray vArr,
    jint w, jint h, jint yStride, jint uStride, jint vStride, jint uPix, jint vPix,
    jint rot, jboolean detect, jobject rgbaOut)
{
    std::lock_guard<std::mutex> lk(g_mutex);

    nc_ov_prop();
    nc_ov_scale_prop();
    nc_ov_pan_prop();

    jbyte* yp = env->GetByteArrayElements(yArr, nullptr);
    jbyte* up = env->GetByteArrayElements(uArr, nullptr);
    jbyte* vp = env->GetByteArrayElements(vArr, nullptr);

    const int rw = (rot == 90 || rot == 270) ? h : w;
    const int rh = (rot == 90 || rot == 270) ? w : h;

    double t0 = nowMs();
    static std::vector<unsigned char> s_bgr, s_upr;
    if (s_bgr.size() < (size_t)w * h * 3) s_bgr.resize((size_t)w * h * 3);
    if (s_upr.size() < (size_t)rw * rh * 3) s_upr.resize((size_t)rw * rh * 3);
    yuvToBgr((const unsigned char*)yp, (const unsigned char*)up, (const unsigned char*)vp,
             w, h, yStride, uStride, vStride, uPix, vPix, s_bgr.data());
    int ckw = 0, ckh = 0;
    rotateBgr(s_bgr.data(), w, h, rot, s_upr.data(), &ckw, &ckh);
    double convMs = nowMs() - t0;

    // RGBA 显示缓冲
    void* rgba = rgbaOut ? env->GetDirectBufferAddress(rgbaOut) : nullptr;
    const jlong rgbaCap = rgbaOut ? env->GetDirectBufferCapacity(rgbaOut) : 0;
    if (rgba && rgbaCap >= (jlong)rw * rh * 4) {
        unsigned char* out = (unsigned char*)rgba;
        const unsigned char* in = s_upr.data();
        const size_t n = (size_t)rw * rh;
        for (size_t i = 0; i < n; i++) {
            out[i * 4 + 0] = in[i * 3 + 2];
            out[i * 4 + 1] = in[i * 3 + 1];
            out[i * 4 + 2] = in[i * 3 + 0];
            out[i * 4 + 3] = 255;
        }
    }

    std::vector<Face> faces;
    double inferMs = 0;
    float maxScore = -1.f;
    bool ok = true;
    const char* err = "";

    // SCRFD 节流（s_cached/s_frm 在文件域，nativeFaceOpen/Close 复位）；
    // mesh 成功后用 468 点包围盒回写缓存框（跟随人脸移动），检测帧全量校正。
    static float s_lastMax = -1.f;
    const int kDetEvery = 4;
    bool detRan = false;   // 检测帧标记（调试/JSON 用）；框口径已统一为 mesh
    if (!detect) s_cached.clear();
    if (detect) {
        if (!g_net || g_input <= 0) { ok = false; err = "not_open"; }
        else if (s_cached.empty() || (s_frm % kDetEvery) == 0) {
            detRan = true;
            double t1 = nowMs();
            // letterbox 到 g_input，pad 填均值 127.5（归一化后为 0）
            const float scale = std::min((float)g_input / rw, (float)g_input / rh);
            const int rw2 = std::max(1, (int)std::lround(rw * scale));
            const int rh2 = std::max(1, (int)std::lround(rh * scale));
            const int padX = (g_input - rw2) / 2;
            const int padY = (g_input - rh2) / 2;
            ncnn::Mat resized = ncnn::Mat::from_pixels_resize(s_upr.data(), ncnn::Mat::PIXEL_BGR,
                                                              rw, rh, rw2, rh2);
            ncnn::Mat in(g_input, g_input, 3);
            in.fill(127.5f);
            for (int q = 0; q < 3; q++) {
                const float* srcp = (const float*)resized.channel(q).data;
                float* dstp = (float*)in.channel(q).data;
                for (int y = 0; y < rh2; y++) {
                    memcpy(dstp + (size_t)(y + padY) * g_input + padX,
                           srcp + (size_t)y * rw2, rw2 * sizeof(float));
                }
            }
            const float mean[3] = {127.5f, 127.5f, 127.5f};
            const float norm[3] = {1 / 128.f, 1 / 128.f, 1 / 128.f};
            in.substract_mean_normalize(mean, norm);

            ncnn::Extractor ex = g_net->create_extractor();
            ex.input(kDetInput, in);
            ncnn::Mat sc[3], bb[3], kp[3];
            bool extOk = true;
            for (int s = 0; s < 3; s++) {
                if (ex.extract(kScoreBlob[s], sc[s]) != 0 || ex.extract(kBoxBlob[s], bb[s]) != 0 ||
                    ex.extract(kKpsBlob[s], kp[s]) != 0) { extOk = false; break; }
            }
            if (!extOk) { ok = false; err = "extract_failed"; }
            else {
                float maxCand = 0.f;
                for (int s = 0; s < 3; s++) decodeStride(sc[s], bb[s], kp[s], kStride[s], &faces);
                for (int s = 0; s < 3; s++) {
                    const float* p = (const float*)sc[s].data;
                    const size_t cnt = (size_t)sc[s].h * sc[s].w * sc[s].c;
                    for (size_t i = 0; i < cnt; i++)
                        if (p[i] > maxCand) maxCand = p[i];
                }
                maxScore = maxCand;
                std::sort(faces.begin(), faces.end(),
                          [](const Face& a, const Face& b) { return a.score > b.score; });
                std::vector<Face> kept;
                for (const Face& f : faces) {
                    if (kept.size() >= (size_t)kMaxFaces) break;
                    bool sup = false;
                    for (const Face& k : kept)
                        if (iou(f, k) > kNmsIou) { sup = true; break; }
                    if (!sup) kept.push_back(f);
                }
                faces.swap(kept);
                for (Face& f : faces) {
                    for (int k = 0; k < 4; k += 2) {
                        f.box[k] = std::min(std::max((f.box[k] - padX) / scale, 0.f), (float)(rw - 1));
                        f.box[k + 1] = std::min(std::max((f.box[k + 1] - padY) / scale, 0.f), (float)(rh - 1));
                    }
                    for (int k = 0; k < 10; k += 2) {
                        f.kps[k] = std::min(std::max((f.kps[k] - padX) / scale, 0.f), (float)(rw - 1));
                        f.kps[k + 1] = std::min(std::max((f.kps[k + 1] - padY) / scale, 0.f), (float)(rh - 1));
                    }
                }
            }
            inferMs = nowMs() - t1;
            // 热轨道合入：box 保持 mesh 口径不变（SCRFD 框与 mesh 框取景比例不同，
            // 交替会让 mesh 回归出现周期性偏置 → overlay 跳动）。SCRFD 只刷 kps/score；
            // 框尺寸偏差 >40% 视为漂移，用 SCRFD 框重同步；无匹配的新脸追加。
            if (s_cached.empty()) {
                s_cached = faces;
            } else {
                std::vector<bool> matched(s_cached.size(), false);
                for (const Face& nf : faces) {
                    const float ndx = (nf.box[0] + nf.box[2]) * 0.5f;
                    const float ndy = (nf.box[1] + nf.box[3]) * 0.5f;
                    const float ndiag = hypotf(nf.box[2] - nf.box[0], nf.box[3] - nf.box[1]);
                    int best = -1; float bd = 1e18f;
                    for (int tk = 0; tk < (int)s_cached.size(); tk++) {
                        const float dx = (s_cached[tk].box[0] + s_cached[tk].box[2]) * 0.5f - ndx;
                        const float dy = (s_cached[tk].box[1] + s_cached[tk].box[3]) * 0.5f - ndy;
                        const float d2 = dx * dx + dy * dy;
                        if (d2 < bd) { bd = d2; best = tk; }
                    }
                    const float nside = std::max(nf.box[2] - nf.box[0], nf.box[3] - nf.box[1]);
                    const float cside = std::max(s_cached[best].box[2] - s_cached[best].box[0],
                                                 s_cached[best].box[3] - s_cached[best].box[1]);
                    if (best >= 0 && bd < ndiag * ndiag * 0.5625f) {
                        memcpy(s_cached[best].kps, nf.kps, sizeof(nf.kps));
                        // 单向重同步：仅 ROI 过紧（mesh 裁切风险）才用 SCRFD 框救援。
                        // 双向会让 ROI 在某些角度每 kDetEvery 帧在大/小口径间振荡 → 抖动
                        if (nside > cside * 1.7f) {
                            s_cached[best].box[0] = nf.box[0]; s_cached[best].box[1] = nf.box[1];
                            s_cached[best].box[2] = nf.box[2]; s_cached[best].box[3] = nf.box[3];
                        }
                        matched[best] = true;
                    } else {
                        s_cached.push_back(nf);
                    }
                }
                int w = 0;
                for (int tk = 0; tk < (int)s_cached.size(); tk++)
                    if (matched[tk] || tk >= (int)matched.size()) s_cached[w++] = s_cached[tk];
                s_cached.resize(w);
            }
            if (!faces.empty()) s_lastMax = maxScore;
        }
        faces = s_cached;
        if (!detRan) maxScore = s_lastMax;
        s_frm++;
    }

    env->ReleaseByteArrayElements(yArr, yp, JNI_ABORT);
    env->ReleaseByteArrayElements(uArr, up, JNI_ABORT);
    env->ReleaseByteArrayElements(vArr, vp, JNI_ABORT);

    // 姿态：融合（facemesh 9 点 Kabsch + hopenet 大角度混入）→ One-Euro → overlay。
    // （PoseTrack/s_tracks/s_cached 定义在文件域：nativeFaceOpen/Close 跨会话复位）
    float wtmp3[3];

    const float fps = 25.f;
    // 轨迹寿命：未被关联的轨迹逐帧 +1，超 ~2s（24 帧 @12fps 实测）整槽复位释放。
    // 若只增不放，死槽永久占坑（has 恒真）→ 换位/重新入镜后 freeTrk 枯竭，
    // 人脸每帧被 continue 丢弃 = 丢失后永远无法再捕获。
    for (int tk = 0; tk < 3; tk++) {
        if (!s_tracks[tk].has) continue;
        if (++s_tracks[tk].miss > 24) s_tracks[tk] = PoseTrack();
    }
    for (int fi = 0; fi < (int)faces.size(); fi++) {
        Face& f = faces[fi];
        float R[9], tt[3];
        float obs5[5][2];
        for (int k = 0; k < 5; k++) { obs5[k][0] = f.kps[2*k]; obs5[k][1] = f.kps[2*k+1]; }

        // 轨迹关联：与现有轨迹按脸中心最近邻匹配（阈值=脸对角一半）
        float nose[2] = { (f.box[0]+f.box[2])*0.5f, (f.box[1]+f.box[3])*0.5f };
        float diag = hypotf(f.box[2]-f.box[0], f.box[3]-f.box[1]);
        PoseTrack* trk = nullptr;
        int freeTrk = -1;
        for (int tk = 0; tk < 3; tk++) {
            if (!s_tracks[tk].has) { if (freeTrk < 0) freeTrk = tk; continue; }
            float ddx = s_tracks[tk].nose[0] - nose[0], ddy = s_tracks[tk].nose[1] - nose[1];
            if (hypotf(ddx, ddy) < diag * 0.5f) { trk = &s_tracks[tk]; break; }
        }
        if (!trk) {
            if (freeTrk < 0) continue;
            trk = &s_tracks[freeTrk];
            *trk = PoseTrack();
            trk->has = true;
        } else if (trk->miss > 8) {
            // 丢失 ~0.6s 后重新关联：滤波/眼线/尺寸状态全部过期，按新轨迹重建，
            // 避免 overlay 从丢失前位置大幅甩动
            *trk = PoseTrack();
            trk->has = true;
        }
        trk->miss = 0;
        trk->nose[0] = nose[0]; trk->nose[1] = nose[1];

        // 姿态源：3=融合（主路径） 1=hopenet（对照）
        bool poseOk = false;
        if (g_poseAlgo == 3 && g_fmNet && g_hopeNet) {
            // 融合：mesh 9 点 Kabsch（正脸/中角度最准，ht 评测 frontal 8.26° 15-45 13.32°）
            // + hopenet（大角度最准 45-70 22.35° extreme 20.12°），按 mesh 姿态幅度平滑切换
            static float obsFM[468 * 2], obsFMz[468];
            if (facemeshRun(s_upr.data(), rw, rh, f.box, obsFM, obsFMz)
                && meshProcrustesR(obsFM, obsFMz, R)) {
                // mesh 468 点包围盒回写缓存框（×1.45 裕量）：所有帧统一 mesh 口径。
                // 若检测帧保留 SCRFD 框、非检测帧用 mesh 框，tt 中心与框尺寸会按
                // kDetEvery 周期在两种口径间阶跃（即「跳动」）；统一后仅剩 mesh 噪声。
                // faces 按 score 排序，检测帧与缓存可能不同序 → 按框中心最近邻匹配回写；
                // kps 不回写——eyeD 距离标定基于 SCRFD 点位口径，混用 mesh 外眼角会失真。
                {
                    float bx0 = 1e9f, by0 = 1e9f, bx1 = -1e9f, by1 = -1e9f;
                    for (int k = 0; k < kFmPoints; k++) {
                        bx0 = std::min(bx0, obsFM[2*k]);   by0 = std::min(by0, obsFM[2*k+1]);
                        bx1 = std::max(bx1, obsFM[2*k]);   by1 = std::max(by1, obsFM[2*k+1]);
                    }
                    const float mcx = (bx0 + bx1) * 0.5f, mcy = (by0 + by1) * 0.5f;
                    const float mw = (bx1 - bx0) * 1.45f, mh = (by1 - by0) * 1.45f;
                    int best = -1; float bd = 1e18f;
                    for (int tk = 0; tk < (int)s_cached.size(); tk++) {
                        const float dx = (s_cached[tk].box[0] + s_cached[tk].box[2]) * 0.5f
                                       - (f.box[0] + f.box[2]) * 0.5f;
                        const float dy = (s_cached[tk].box[1] + s_cached[tk].box[3]) * 0.5f
                                       - (f.box[1] + f.box[3]) * 0.5f;
                        const float d2 = dx * dx + dy * dy;
                        if (d2 < bd) { bd = d2; best = tk; }
                    }
                    if (best >= 0) {
                        Face& cf = s_cached[best];
                        cf.box[0] = mcx - mw * 0.5f; cf.box[1] = mcy - mh * 0.5f;
                        cf.box[2] = mcx + mw * 0.5f; cf.box[3] = mcy + mh * 0.5f;
                        // 板尺寸源：原始包围盒宽（不含 ×1.45 ROI 裕量）
                        const float rwRaw = bx1 - bx0;
                        PoseTrack* trkRaw = trk;
                        trkRaw->rawW = trkRaw->rawW > 0.f
                            ? trkRaw->rawW + (rwRaw - trkRaw->rawW) * 0.15f
                            : rwRaw;
                        // 眼线（33→263，相机系）：板横轴直接对齐可见人脸倾斜
                        trk->eyeDX = obsFM[2*263]   - obsFM[2*33];
                        trk->eyeDY = obsFM[2*263+1] - obsFM[2*33+1];
                    }
                    f.box[0] = mcx - mw * 0.5f; f.box[1] = mcy - mh * 0.5f;
                    f.box[2] = mcx + mw * 0.5f; f.box[3] = mcy + mh * 0.5f;
                }
                float yprM[3];
                yprFromR(R, yprM);
                float mag = std::max(fabsf(yprM[1]), fabsf(yprM[0]));
                // hopenet 接入滞回：38° 开 / 30° 关——35° 单阈值会在边界上反复开关，
                // mesh 与 hopenet 的系统偏差交替出现 = 特定角度抖动
                if (!trk->hopeOn) { if (mag >= 38.f && g_hopeNet) trk->hopeOn = true; }
                else if (mag < 30.f) trk->hopeOn = false;
                float yh = 0, ph2 = 0, rh2 = 0;
                bool haveH = false;
                if (trk->hopeOn)
                    haveH = hopenetRun(s_upr.data(), rw, rh, f.box, &yh, &ph2, &rh2);
                float wH = std::min(std::max((mag - 35.f) / 20.f, 0.f), 1.f);
                if (haveH && wH > 0.f) {
                    float yb = yprM[1] * (1-wH) + yh * wH;
                    float pb = yprM[0] * (1-wH) + ph2 * wH;
                    float rb = yprM[2] * (1-wH) + rh2 * wH;
                    eulerToR(yb, pb, rb, R);
                    f.ypr[0] = pb; f.ypr[1] = yb; f.ypr[2] = rb;
                } else {
                    yprFromR(R, f.ypr);
                }
                float eyeD = hypotf(obs5[1][0] - obs5[0][0], obs5[1][1] - obs5[0][1]);
                if (eyeD > 8.f) {
                    // eyeD EWMA（α=0.25）：SCRFD kps 每 kDetEvery 帧才刷新，直接用会阶跃
                    if (!trk->hasEyeD) { trk->eyeDS = eyeD; trk->hasEyeD = true; }
                    else trk->eyeDS += 0.25f * (eyeD - trk->eyeDS);
                    float Z = (float)rw * kEyeSpan5 / trk->eyeDS;
                    tt[0] = ((f.box[0] + f.box[2]) * 0.5f - rw * 0.5f) * Z / (float)rw;
                    tt[1] = ((f.box[1] + f.box[3]) * 0.5f - rh * 0.5f) * Z / (float)rh;
                    tt[2] = Z;
                    poseOk = true;
                }
            }
        } else if (g_poseAlgo == 1 && g_hopeNet) {
            float hy2 = 0, hp2 = 0, hr2 = 0;
            if (hopenetRun(s_upr.data(), rw, rh, f.box, &hy2, &hp2, &hr2)) {
                eulerToR(hy2, hp2, hr2, R);
                float eyeD = hypotf(obs5[1][0] - obs5[0][0], obs5[1][1] - obs5[0][1]);
                if (eyeD > 8.f) {
                    if (!trk->hasEyeD) { trk->eyeDS = eyeD; trk->hasEyeD = true; }
                    else trk->eyeDS += 0.25f * (eyeD - trk->eyeDS);
                    float Z = (float)rw * kEyeSpan5 / trk->eyeDS;
                    tt[0] = ((f.box[0] + f.box[2]) * 0.5f - rw * 0.5f) * Z / (float)rw;
                    tt[1] = ((f.box[1] + f.box[3]) * 0.5f - rh * 0.5f) * Z / (float)rh;
                    tt[2] = Z;
                    yprFromR(R, f.ypr);
                    poseOk = true;
                }
            }
        }
        if (!poseOk) {
            poseOk = solveHeadPose(obs5, kModel5, 5, kEyeSpan5, rw, rh, R, tt, f.ypr);
        }
        f.hasPose = poseOk;
        if (!f.hasPose) continue;
        const float rawRoll = f.ypr[2];   // One-Euro 前（诊断用）

        // One-Euro（旋转矢量）：静止强平滑、快速转动直通（逐分量自适应截止）
        r2w(R, wtmp3);
        for (int q = 0; q < 3; q++) {
            if (!trk->hasW) { trk->w[q] = wtmp3[q]; trk->xPrev[q] = wtmp3[q]; trk->hasW = true; continue; }
            float dx = (wtmp3[q] - trk->xPrev[q]) * fps;
            if (!trk->hasEuro[q]) { trk->dPrev[q] = dx; trk->hasEuro[q] = true; }
            // 旋转通道：0.25 是蓝本 25fps 口径；我们 ~12fps，等效滞后翻倍，
            // minCutoff 提到 0.5 让 roll 跟随延迟回到蓝本水准
            const float dCutoff = 1.2f, minCutoff = 0.5f, beta = 0.08f;
            float ad = 1.f / (1.f + (1.f/(2.f*3.14159265f*dCutoff)) * fps);
            float dhat = trk->dPrev[q] + ad * (dx - trk->dPrev[q]);
            trk->dPrev[q] = dhat;
            float a = 1.f / (1.f + (1.f/(2.f*3.14159265f*(minCutoff + beta*fabsf(dhat)))) * fps);
            float xh = trk->xPrev[q] + a * (wtmp3[q] - trk->xPrev[q]);
            trk->xPrev[q] = xh;
            trk->w[q] = xh;
        }
        rodrigues(trk->w, R);
        yprFromR(R, f.ypr);
        // 平移 One-Euro：overlay 锚点稳定的关键（t 抖动直接映射为平面漂移）
        for (int q = 0; q < 3; q++) {
            if (!trk->hasT) { trk->tPrev[q] = tt[q]; trk->tDPrev[q] = 0; trk->hasT = true; continue; }
            float dx = (tt[q] - trk->tPrev[q]) * fps;
            if (!trk->hasTD[q]) { trk->tDPrev[q] = dx; trk->hasTD[q] = true; }
            const float dCutoff = 1.2f, minCutoff = 0.6f, beta = 0.25f;
            float ad = 1.f / (1.f + (1.f/(2.f*3.14159265f*dCutoff)) * fps);
            float dhat = trk->tDPrev[q] + ad * (dx - trk->tDPrev[q]);
            trk->tDPrev[q] = dhat;
            float a = 1.f / (1.f + (1.f/(2.f*3.14159265f*(minCutoff + beta*fabsf(dhat)))) * fps);
            float xh = trk->tPrev[q] + a * (tt[q] - trk->tPrev[q]);
            trk->tPrev[q] = xh;
            tt[q] = xh;
        }
        projectNormal(R, tt, rw, rh, f.normal);
        float ovD = 0;
        // overlay 放置（法线跟随头部 + roll 跟随眼线 + 投影恒 1:1）：
        //  - 平面中心 = 脸中心 t + N·D，N = -R 第 3 列：板随头部俯仰/偏航倾斜；
        //    R 的 pitch 经模板去偏校准（正脸=0），姿态偏差不再带入板朝向。
        //  - 横轴 = 观测眼线（33→263）投影，对 N 正交化——roll 不经 Kabsch 约定。
        //  - 投影尺寸：宽 = 1.5×平滑包围盒宽（针孔反算）；高按内容纵横比给定，
        //    (0,0,0) 姿态投影严格 1:1，其他姿态遵循正常透视关系。
        {
            const float cmU = 450.f / 9.5f;        // 模型单位/厘米（外眼角 450 单位 ≈ 9.5cm）
            const float cx = rw * 0.5f, cy = rh * 0.5f;
            const float fxf = (float)rw, fyf = (float)rh;
            const float Dcm = (float)g_ovCm;       // 距离固定（默认 20cm，debug.gscp.ovcm 可调）
            const float D = Dcm * cmU;
            // 平滑框 EWMA（尺寸源稳定；rawW 未含 ROI 裕量）
            float bw = f.box[2] - f.box[0], bh = f.box[3] - f.box[1];
            if (!trk->hasDim) { trk->bwS = bw; trk->bhS = bh; trk->hasDim = true; }
            trk->bwS += (bw - trk->bwS) * 0.15f;
            trk->bhS += (bh - trk->bhS) * 0.15f;
            // 板法线 = -R 第 3 列（朝相机，跟随头部俯仰/偏航，pitch 已校准）
            float nx = -R[2], ny = -R[5], nz = -R[8];
            float nn = sqrtf(nx*nx + ny*ny + nz*nz);
            if (nn < 1e-6f) nn = 1.f;
            nx /= nn; ny /= nn; nz /= nn;
            // 锚点 = 脸中心 + N·D：板沿法线悬浮于脸前（法线一致——板朝向与
            // 位移方向一致，倾斜观感自然）；pitch 已由模板去偏校准。
            // 板轴 r1/r2 备好后沿板平面平移（ovpanx/ovpany），再投影中心。
            float C[3] = { nx*D + tt[0], ny*D + tt[1], nz*D + tt[2] };
            // 横轴 = 眼线投影，对 N 正交化；退化沿用上帧（侧脸兜底）。
            // 眼线与 R 第 0 列（模型 +x = 33→263）同向，无需 y 取反——
            // 取反会让板倾斜方向与预览中人脸倾斜相反（实测 roll 反向）
            float exd = trk->eyeDX, eyd = trk->eyeDY;
            // 板 roll 直接取眼线方向（33→263 raw 地标差，无滤波），地标逐帧噪声
            // 直通为板左右晃动。对眼线角做 One-Euro（环绕安全，±π wrap）：
            // 静止强平滑压抖动、快速转头经 beta 提截止保持跟随。
            if (!trk->hasRollE) {
                trk->rollPrev = atan2f(eyd, exd);
                trk->rollDPrev = 0.f;
                trk->hasRollE = true;
            } else {
                const float ang = atan2f(eyd, exd);
                float dA = ang - trk->rollPrev;
                if (dA > 3.14159265f) dA -= 2.f * 3.14159265f;
                if (dA < -3.14159265f) dA += 2.f * 3.14159265f;
                const float dCut = 1.0f, minCut = 0.3f, beta = 0.15f;
                const float dx = dA * fps;
                const float ad = 1.f / (1.f + (1.f / (2.f * 3.14159265f * dCut)) * fps);
                float dhat = trk->rollDPrev + ad * (dx - trk->rollDPrev);
                trk->rollDPrev = dhat;
                const float a = 1.f / (1.f +
                    (1.f / (2.f * 3.14159265f * (minCut + beta * fabsf(dhat)))) * fps);
                float xh = trk->rollPrev + a * dA;
                if (xh > 3.14159265f) xh -= 2.f * 3.14159265f;
                if (xh < -3.14159265f) xh += 2.f * 3.14159265f;
                trk->rollPrev = xh;
                exd = cosf(xh);
                eyd = sinf(xh);
            }
            float elen = sqrtf(exd*exd + eyd*eyd);
            if (elen < 1e-3f) { exd = 1.f; eyd = 0.f; elen = 1.f; }
            float r1x = exd / elen, r1y = eyd / elen, r1z = 0.f;
            float dN = r1x*nx + r1y*ny + r1z*nz;
            r1x -= dN*nx; r1y -= dN*ny; r1z -= dN*nz;
            float l1 = sqrtf(r1x*r1x + r1y*r1y + r1z*r1z);
            if (l1 < 0.15f && trk->hasR1) {
                r1x = trk->r1L[0]; r1y = trk->r1L[1]; r1z = trk->r1L[2];
                dN = r1x*nx + r1y*ny + r1z*nz;
                r1x -= dN*nx; r1y -= dN*ny; r1z -= dN*nz;
                l1 = sqrtf(r1x*r1x + r1y*r1y + r1z*r1z);
            }
            if (l1 < 1e-6f) l1 = 1.f;
            r1x /= l1; r1y /= l1; r1z /= l1;
            trk->r1L[0] = r1x; trk->r1L[1] = r1y; trk->r1L[2] = r1z; trk->hasR1 = true;
            float r2x = ny*r1z - nz*r1y;
            float r2y = nz*r1x - nx*r1z;
            float r2z = nx*r1y - ny*r1x;
            // 板平面内平移（cm→模型单位）：水平沿 r1（眼线方向）、竖直沿 r2
            const float panU = g_ovPanX * cmU, panV = g_ovPanY * cmU;
            C[0] += r1x*panU + r2x*panV;
            C[1] += r1y*panU + r2y*panV;
            C[2] += r1z*panU + r2z*panV;
            if (C[2] < 1.f) C[2] = 1.f;
            const float ccx = cx + fxf * C[0] / C[2];
            const float ccy = cy + fyf * C[1] / C[2];
            const float zPlane = C[2];
            // 高度：世界尺寸按内容纵横比给定。(0,0,0) 姿态时 n=(0,0,-1)、r1 水平、
            // r2 竖直且四角同深度 → 投影高 = fyf·2hh/z = fx·2hw/z = 投影宽，严格 1:1；
            // 其他姿态按正常透视投影，随倾斜自然缩短（不人为补偿）
            const float srcW = trk->rawW > 0.f ? trk->rawW : trk->bwS * 0.69f;
            const float hw = (srcW * 0.5f * g_ovScale) * zPlane / fxf;
            const float hh = hw * ((float)g_ovH / (float)g_ovW) * (fxf / fyf);
            // 背板 = 前板沿 -N 推 2mm（厚度暗示）
            const float thCm = 0.2f;
            const float Bx = C[0] + nx * thCm * cmU;
            const float By = C[1] + ny * thCm * cmU;
            const float Bz = C[2] + nz * thCm * cmU;
            const signed char W2[4][2] = { {-1,-1}, {1,-1}, {1,1}, {-1,1} };
            for (int i = 0; i < 4; i++) {
                const float mx = W2[i][0] * hw, my = W2[i][1] * hh;
                float X[3] = { C[0] + r1x*mx + r2x*my,
                               C[1] + r1y*mx + r2y*my,
                               C[2] + r1z*mx + r2z*my };
                if (X[2] < 1.f) X[2] = 1.f;
                f.ov[i][0] = cx + fxf * X[0] / X[2];
                f.ov[i][1] = cy + fyf * X[1] / X[2];
                float Y[3] = { Bx + r1x*mx + r2x*my,
                               By + r1y*mx + r2y*my,
                               Bz + r1z*mx + r2z*my };
                if (Y[2] < 1.f) Y[2] = 1.f;
                f.ovB[i][0] = cx + fxf * Y[0] / Y[2];
                f.ovB[i][1] = cy + fyf * Y[1] / Y[2];
            }
            f.ovC[0] = ccx; f.ovC[1] = ccy;
            ovD = Dcm;
        }
        // overlay 像素 One-Euro：中心绝对滤波，四角只滤相对中心的偏移（形状刚性）
        {
            float cur[10];
            cur[0] = f.ovC[0]; cur[1] = f.ovC[1];
            for (int i = 0; i < 4; i++) {
                cur[2 + 2*i] = f.ov[i][0] - f.ovC[0];
                cur[3 + 2*i] = f.ov[i][1] - f.ovC[1];
            }
            const float dCutoff = 1.2f, minCutoff = 0.9f, beta = 0.28f;
            float ad = 1.f / (1.f + (1.f/(2.f*3.14159265f*dCutoff)) * fps);
            for (int q2 = 0; q2 < 10; q2++) {
                if (!trk->hasO) {
                    trk->ovPrev[q2] = cur[q2]; trk->ovDP[q2] = 0; trk->hasOE[q2] = true;
                    continue;
                }
                float dx2 = (cur[q2] - trk->ovPrev[q2]) * fps;
                if (!trk->hasOE[q2]) { trk->ovDP[q2] = dx2; trk->hasOE[q2] = true; }
                float dhat = trk->ovDP[q2] + ad * (dx2 - trk->ovDP[q2]);
                trk->ovDP[q2] = dhat;
                float a = 1.f / (1.f + (1.f/(2.f*3.14159265f*(minCutoff + beta*fabsf(dhat)))) * fps);
                float xh = trk->ovPrev[q2] + a * (cur[q2] - trk->ovPrev[q2]);
                trk->ovPrev[q2] = xh;
                cur[q2] = xh;
            }
            trk->hasO = true;
            f.ovC[0] = cur[0]; f.ovC[1] = cur[1];
            for (int i = 0; i < 4; i++) {
                f.ov[i][0] = cur[0] + cur[2 + 2*i];
                f.ov[i][1] = cur[1] + cur[3 + 2*i];
            }
        }
        f.ovD = ovD;
        f.hasOv = true;
        static int rollDbg = 0;
        if (++rollDbg % 30 == 1)
            LOGI("ovc=%.1f,%.1f dims=%.0fx%.0f det=%d roll=%.1f ypr=%.0f/%.0f/%.0f",
                 f.ovC[0], f.ovC[1], f.box[2] - f.box[0], f.box[3] - f.box[1],
                 (int)detRan, rawRoll, f.ypr[0], f.ypr[1], f.ypr[2]);
    }

    std::string j = std::string("{\"ok\":") + (ok ? "true" : "false");
    if (!ok) j += std::string(",\"err\":\"") + err + "\"";
    if (g_backend >= 0) j += ",\"backend\":\"" + std::string(kBackendName[g_backend]) + "\"";
    j += ",\"detect\":" + std::string(detect ? "true" : "false");
    j += ",\"algo\":\"" + std::string(poseAlgoName()) + "\"";
    j += ",\"convMs\":" + f2(convMs) + ",\"ms\":" + f2(inferMs);
    j += ",\"threads\":" + std::to_string(g_threads);
    j += ",\"input\":" + std::to_string(g_input);
    j += ",\"w\":" + std::to_string(rw) + ",\"h\":" + std::to_string(rh);
    j += ",\"rot\":" + std::to_string(rot);
    j += ",\"maxScore\":" + f3(maxScore);
    j += ",\"faces\":[";
    for (size_t i = 0; i < faces.size(); i++) {
        if (i) j += ",";
        const Face& f = faces[i];
        j += "{\"box\":[" + f1(f.box[0]) + "," + f1(f.box[1]) + "," + f1(f.box[2]) + "," + f1(f.box[3]) + "]";
        j += ",\"score\":" + f3(f.score) + ",\"kps\":[";
        for (int k = 0; k < 10; k++) j += (k ? "," : "") + f1(f.kps[k]);
        j += "]";
        if (f.hasPose) {
            j += ",\"pose\":[" + f1(f.ypr[0]) + "," + f1(f.ypr[1]) + "," + f1(f.ypr[2]) + "]";
            j += ",\"normal\":[[" + f1(f.normal[0][0]) + "," + f1(f.normal[0][1]) + "],["
               + f1(f.normal[1][0]) + "," + f1(f.normal[1][1]) + "]]";
        }
        if (f.hasOv) {
            j += ",\"overlay\":[";
            for (int a = 0; a < 4; a++) {
                if (a) j += ",";
                j += "[" + f1(f.ov[a][0]) + "," + f1(f.ov[a][1]) + "]";
            }
            j += "],\"back\":[";
            for (int a = 0; a < 4; a++) {
                if (a) j += ",";
                j += "[" + f1(f.ovB[a][0]) + "," + f1(f.ovB[a][1]) + "]";
            }
            j += "],\"ovC\":[" + f1(f.ovC[0]) + "," + f1(f.ovC[1]) + "],\"ovD\":" + f1(f.ovD);
        }
        j += "}";
    }
    j += "]}";
    return env->NewStringUTF(j.c_str());
}

// ═══════════════════════════════════════════════════════════════════════════
// overlay 超分（ncnn-benchmark sr_benchmark ESPCN x2 灰度，单色文字目标域训练）：
// 480² 亮度 → 960² 亮度；色度仍走 GL 双线性（luma-SR + 色度移植是文字超分标准做法）。
// 独立 g_srMutex：SR 在 GL 线程调用，不得与检测线程的 g_mutex 互相阻塞。
// 后端：Vulkan 优先（小卷积网 CPU 4 线程 ~10ms/帧偏重），失败按蓝本回退
// safe-CPU（fp32/普通卷积/多线程；真机对 pnnx 模型有 SIGSEGV 前科，见
// ensurePoseNets 注释）。
// ═══════════════════════════════════════════════════════════════════════════

static ncnn::Net* g_srNet = nullptr;
static std::mutex g_srMutex;
static const char* kSrParam = "espcn_x2.ncnn.param";
static const char* kSrBin = "espcn_x2.ncnn.bin";

extern "C" JNIEXPORT jstring JNICALL
Java_com_gscp_desktop_ArNative_nativeSrOpen(
    JNIEnv* env, jobject, jobject assets, jint threads, jboolean wantVulkan)
{
    std::lock_guard<std::mutex> lk(g_srMutex);
    if (g_srNet) { delete g_srNet; g_srNet = nullptr; }
    AAssetManager* am = assets ? AAssetManager_fromJava(env, assets) : nullptr;
    if (!am) return env->NewStringUTF("{\"ok\":false,\"err\":\"no_assets\"}");
    std::vector<unsigned char> p = readAsset(am, kSrParam);
    std::vector<unsigned char> b = readAsset(am, kSrBin);
    if (p.empty() || b.empty()) return env->NewStringUTF("{\"ok\":false,\"err\":\"assets_missing\"}");
    std::string param((const char*)p.data(), p.size());
    param.push_back('\0');

    // Vulkan 设备进程级只取一次（与 ensurePoseNets 同款；SR 独立使用时也生效）
    static std::once_flag srVkOnce;
    std::call_once(srVkOnce, [] { if (!g_vk) g_vk = ncnn::get_gpu_device(0); });

    if (wantVulkan == JNI_TRUE && g_vk) {
        ncnn::Net* net = new ncnn::Net;
        net->opt.lightmode = true;
        net->opt.num_threads = 1;
        net->opt.use_vulkan_compute = true;
        // 真机 Mali 实测：本模型 Vulkan fp16 损坏，默认 fp32（SR_VK_FP16=1 实验用）
        net->opt.use_fp16_packed = false;
        net->opt.use_fp16_storage = false;
        net->opt.use_fp16_arithmetic = false;
        if (const char* e = getenv("SR_VK_FP16"); e && e[0] == '1') {
            net->opt.use_fp16_packed = true;
            net->opt.use_fp16_storage = true;
            net->opt.use_fp16_arithmetic = true;
            LOGI("sr vulkan fp16 experiment");
        }
        net->opt.use_packing_layout = true;
        net->opt.use_shader_local_memory = false;
        ncnn::VkBlobAllocator* vb = new ncnn::VkBlobAllocator(g_vk);
        ncnn::VkStagingAllocator* vs = new ncnn::VkStagingAllocator(g_vk);
        net->opt.blob_vkallocator = vb;
        net->opt.workspace_vkallocator = vb;
        net->opt.staging_vkallocator = vs;
        net->set_vulkan_device(g_vk);
        if (net->load_param_mem(param.c_str()) == 0) {
            const unsigned char* mem = b.data();
            ncnn::DataReaderFromMemory dr(mem);
            if (net->load_model(dr) == 0) {
                g_srNet = net;
                LOGI("sr net on vulkan");
                return env->NewStringUTF("{\"ok\":true,\"backend\":\"vulkan\",\"scale\":2}");
            }
        }
        delete net;
        LOGE("sr vulkan load failed, fallback cpu");
    }

    // safe-CPU：fp32 + 普通卷积（pnnx 模型真机稳健配置），线程数可配
    ncnn::Net* net = new ncnn::Net;
    net->opt.lightmode = true;
    net->opt.num_threads = threads > 0 ? threads : 4;
    net->opt.use_fp16_storage = false;
    net->opt.use_fp16_packed = false;
    net->opt.use_fp16_arithmetic = false;
    net->opt.use_sgemm_convolution = false;
    net->opt.use_winograd_convolution = false;
    net->opt.use_vulkan_compute = false;
    if (net->load_param_mem(param.c_str()) != 0) { delete net; return env->NewStringUTF("{\"ok\":false,\"err\":\"load_param\"}"); }
    const unsigned char* mem = b.data();
    ncnn::DataReaderFromMemory dr(mem);
    if (net->load_model(dr) != 0) { delete net; return env->NewStringUTF("{\"ok\":false,\"err\":\"load_model\"}"); }
    g_srNet = net;
    LOGI("sr net on safe cpu threads=%d", net->opt.num_threads);
    return env->NewStringUTF("{\"ok\":true,\"backend\":\"cpu\",\"scale\":2}");
}

extern "C" JNIEXPORT void JNICALL
Java_com_gscp_desktop_ArNative_nativeSrClose(JNIEnv*, jobject)
{
    std::lock_guard<std::mutex> lk(g_srMutex);
    if (g_srNet) { delete g_srNet; g_srNet = nullptr; }
}

/** lumaIn = w*h 字节亮度；lumaOut = 4*w*h 字节（2x2 放大）。同线程串行调用。 */
extern "C" JNIEXPORT jstring JNICALL
Java_com_gscp_desktop_ArNative_nativeSrProcess(
    JNIEnv* env, jobject, jobject lumaIn, jint w, jint h, jobject lumaOut)
{
    std::lock_guard<std::mutex> lk(g_srMutex);
    if (!g_srNet) return env->NewStringUTF("{\"ok\":false,\"err\":\"not_open\"}");
    uint8_t* in = (uint8_t*)(lumaIn ? env->GetDirectBufferAddress(lumaIn) : nullptr);
    uint8_t* out = (uint8_t*)(lumaOut ? env->GetDirectBufferAddress(lumaOut) : nullptr);
    const jlong inCap = lumaIn ? env->GetDirectBufferCapacity(lumaIn) : 0;
    const jlong outCap = lumaOut ? env->GetDirectBufferCapacity(lumaOut) : 0;
    if (!in || !out || inCap < (jlong)w * h || outCap < (jlong)4 * w * h)
        return env->NewStringUTF("{\"ok\":false,\"err\":\"bad_buffer\"}");

    const double t0 = nowMs();
    ncnn::Mat src(w, h, 1);
    {
        float* p = (float*)src.data;
        for (int i = 0; i < w * h; i++) p[i] = in[i] * (1.0f / 255.0f);
    }
    ncnn::Extractor ex = g_srNet->create_extractor();
    if (ex.input("in0", src) != 0)
        return env->NewStringUTF("{\"ok\":false,\"err\":\"input\"}");
    ncnn::Mat dst;
    if (ex.extract("out0", dst) != 0 || dst.w != w * 2 || dst.h != h * 2)
        return env->NewStringUTF("{\"ok\":false,\"err\":\"extract\"}");
    const int ow = w * 2, oh = h * 2;
    for (int y = 0; y < oh; y++) {
        const float* row = dst.row(y);
        uint8_t* orow = out + (size_t)y * ow;
        for (int x = 0; x < ow; x++) {
            float v = row[x] * 255.0f;
            int i = (int)(v + 0.5f);
            orow[x] = (uint8_t)(i < 0 ? 0 : (i > 255 ? 255 : i));
        }
    }
    const double ms = nowMs() - t0;
    char buf[96];
    snprintf(buf, sizeof(buf), "{\"ok\":true,\"ms\":%.2f,\"w\":%d,\"h\":%d,\"ow\":%d,\"oh\":%d}",
             ms, w, h, ow, oh);
    return env->NewStringUTF(buf);
}

/** 文件路径加载变体：离线 app_process jar 测试用（无 AssetManager 上下文）。 */
extern "C" JNIEXPORT jstring JNICALL
Java_com_gscp_desktop_ArNative_nativeSrOpenPath(
    JNIEnv* env, jobject, jstring paramPath, jstring binPath, jint threads, jboolean wantVulkan)
{
    std::lock_guard<std::mutex> lk(g_srMutex);
    if (g_srNet) { delete g_srNet; g_srNet = nullptr; }
    const char* pp = env->GetStringUTFChars(paramPath, nullptr);
    const char* bp = env->GetStringUTFChars(binPath, nullptr);
    FILE* fp = fopen(pp, "rb");
    FILE* fb = fopen(bp, "rb");
    std::string param;
    std::vector<unsigned char> b;
    if (fp) {
        char chunk[4096];
        size_t n;
        while ((n = fread(chunk, 1, sizeof(chunk), fp)) > 0) param.append(chunk, n);
        param.push_back('\0');
        fclose(fp);
    }
    if (fb) {
        unsigned char chunk[4096];
        size_t n;
        while ((n = fread(chunk, 1, sizeof(chunk), fb)) > 0) b.insert(b.end(), chunk, chunk + n);
        fclose(fb);
    }
    env->ReleaseStringUTFChars(paramPath, pp);
    env->ReleaseStringUTFChars(binPath, bp);
    if (param.empty() || b.empty()) return env->NewStringUTF("{\"ok\":false,\"err\":\"files_missing\"}");

    static std::once_flag srVkOnce2;
    std::call_once(srVkOnce2, [] { if (!g_vk) g_vk = ncnn::get_gpu_device(0); });

    if (wantVulkan == JNI_TRUE && g_vk) {
        ncnn::Net* net = new ncnn::Net;
        net->opt.lightmode = true;
        net->opt.num_threads = 1;
        net->opt.use_vulkan_compute = true;
        net->opt.use_packing_layout = true;
        // 真机 Mali 实测：本模型 Vulkan fp16 输出竖条纹损坏（fp32 与 CPU 逐像素
        // 一致）——默认 fp32，SR_VK_FP16=1 仅作实验对照。
        net->opt.use_fp16_packed = false;
        net->opt.use_fp16_storage = false;
        net->opt.use_fp16_arithmetic = false;
        if (const char* e = getenv("SR_VK_FP16"); e && e[0] == '1') {
            net->opt.use_fp16_packed = true;
            net->opt.use_fp16_storage = true;
            net->opt.use_fp16_arithmetic = true;
            LOGI("sr vulkan fp16 experiment");
        }
        net->opt.use_shader_local_memory = false;
        ncnn::VkBlobAllocator* vb = new ncnn::VkBlobAllocator(g_vk);
        ncnn::VkStagingAllocator* vs = new ncnn::VkStagingAllocator(g_vk);
        net->opt.blob_vkallocator = vb;
        net->opt.workspace_vkallocator = vb;
        net->opt.staging_vkallocator = vs;
        net->set_vulkan_device(g_vk);
        if (net->load_param_mem(param.c_str()) == 0) {
            const unsigned char* mem = b.data();
            ncnn::DataReaderFromMemory dr(mem);
            if (net->load_model(dr) == 0) {
                g_srNet = net;
                LOGI("sr net(path) on vulkan");
                return env->NewStringUTF("{\"ok\":true,\"backend\":\"vulkan\",\"scale\":2}");
            }
        }
        delete net;
        LOGE("sr vulkan(path) load failed, fallback cpu");
    }

    ncnn::Net* net = new ncnn::Net;
    net->opt.lightmode = true;
    net->opt.num_threads = threads > 0 ? threads : 4;
    net->opt.use_fp16_storage = false;
    net->opt.use_fp16_packed = false;
    net->opt.use_fp16_arithmetic = false;
    net->opt.use_sgemm_convolution = false;
    net->opt.use_winograd_convolution = false;
    net->opt.use_vulkan_compute = false;
    if (net->load_param_mem(param.c_str()) != 0) { delete net; return env->NewStringUTF("{\"ok\":false,\"err\":\"load_param\"}"); }
    const unsigned char* mem = b.data();
    ncnn::DataReaderFromMemory dr(mem);
    if (net->load_model(dr) != 0) { delete net; return env->NewStringUTF("{\"ok\":false,\"err\":\"load_model\"}"); }
    g_srNet = net;
    LOGI("sr net(path) on safe cpu threads=%d", net->opt.num_threads);
    return env->NewStringUTF("{\"ok\":true,\"backend\":\"cpu\",\"scale\":2}");
}
