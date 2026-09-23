// ncnn_engine.cpp — JNI shim for NCNN(-vulkan, fp16) det_10g + 1k3d68.
#include <jni.h>
#include <android/log.h>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include "net.h"
#include "gpu.h"
#include <sys/system_properties.h>

#define TAG "gscp-ncnn"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)

static ncnn::Net* load_net(const std::string& param, const std::string& bin, bool gpu) {
    (void)ncnn::get_gpu_count();
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
    // 引擎选择：默认（未设）全 GPU fp16（det+lm，~95ms/帧，真机已验）；
    // debug.gscp.novulkan 1=强制全 CPU，2=全 GPU fp32。
    int mode = pb[0] ? pb[0] - '0' : -1;
    if (mode == 1) {
        gpu = false;               // 全 CPU
    } else {                        // 默认/-1、0、2 → 全 GPU（-1/0=fp16，2=fp32）
        gpu = true;
    }
    if (gpu && ncnn::get_gpu_count() > 0) {
        o.use_vulkan_compute = true;
        o.use_fp16_packed = (mode != 2);
        o.use_fp16_storage = (mode != 2);
        o.use_fp16_arithmetic = (mode != 2);
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
         (int)o.use_vulkan_compute, (int)(ncnn::get_gpu_count() > 0), (int)o.use_fp16_arithmetic);
    return net;
}

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_gscp_desktop_NcnnEngine_nativeInit(JNIEnv* env, jobject,
                                            jstring param, jstring bin, jboolean useGpu) {
    const char* p = env->GetStringUTFChars(param, nullptr);
    const char* b = env->GetStringUTFChars(bin, nullptr);
    ncnn::Net* net = p && b ? load_net(p, b, useGpu == JNI_TRUE) : nullptr;
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

}  // extern "C"