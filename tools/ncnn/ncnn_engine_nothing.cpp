// ncnn_engine.cpp — JNI shim for NCNN(-vulkan, fp16) det_10g + 1k3d68.
// Provides nativeInit/nativeRun/nativeRelease able to produce the SAME
// score/box/kps tensors (or fc1) the app's ORT decode expects.
#include <jni.h>
#include <android/log.h>
#include <atomic>
#include <cstring>
#include <string>
#include <vector>
#include <thread>

#include "net.h"
#include "gpu.h"

#define TAG "gscp-ncnn"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)

namespace {

static ncnn::Net* load_net(const std::string& param, const std::string& bin, bool gpu) {
    ncnn::get_gpu_count();  // touch vulkan to init driver
    ncnn::Net* net = new ncnn::Net();
    ncnn::Option o = net->opt;
    o.num_threads = (int)std::thread::hardware_concurrency();
    o.lightmode = false;
    o.use_packing_layout = true;
    if (gpu && ncnn::get_gpu_count() > 0) {
        o.use_vulkan_compute = true;
        o.use_fp16_packed = true;
        o.use_fp16_storage = true;
        o.use_fp16_arithmetic = true;
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
        delete net;
        return nullptr;
    }
    LOGI("loaded %s gpu=%d vulkanAvail=%d fp16_arith=%d", param.c_str(),
         (int)o.use_vulkan_compute, (int)(ncnn::get_gpu_count() > 0), (int)o.use_fp16_arithmetic);
    return net;
}

struct RunCtx {
    ncnn::Net* net = nullptr;
    bool owns = false;
};

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_gscp_desktop_NcnnEngine_nativeInit(JNIEnv* env, jobject /*this*/,
                                            jstring param, jstring bin, jboolean useGpu) {
    const char* p = env->GetStringUTFChars(param, nullptr);
    const char* b = env->GetStringUTFChars(bin, nullptr);
    ncnn::Net* net = load_net(p ? p : "", b ? b : "", useGpu == JNI_TRUE);
    if (p) env->ReleaseStringUTFChars(param, p);
    if (b) env->ReleaseStringUTFChars(bin, b);
    return (jlong)net;
}

JNIEXPORT void JNICALL
Java_com_gscp_desktop_NcnnEngine_nativeRelease(JNIEnv*, jobject, jlong handle) {
    if (handle == 0) return;
    delete (ncnn::Net*)handle;
}

// nativeRun(handle, inputName, data, names, dims) -> concat of each named output;
// dims[i] filled with output i element count.
JNIEXPORT jfloatArray JNICALL
Java_com_gscp_desktop_NcnnEngine_nativeRun(JNIEnv* env, jobject /*this*/, jlong handle,
                                          jstring inputName, jfloatArray data,
                                          jobjectArray names, jintArray dims) {
    ncnn::Net* net = (ncnn::Net*)handle;
    if (!net) return nullptr;
    const char* in = inputName ? env->GetStringUTFChars(inputName, nullptr) : nullptr;
    jsize ndata = env->GetArrayLength(data);
    jfloat* dptr = env->GetFloatArrayElements(data, nullptr);
    jsize nNames = names ? env->GetArrayLength(names) : 0;
    if (nNames == 0 || ndata == 0) { if (in) env->ReleaseStringUTFChars(inputName, in); if (dptr) env->ReleaseFloatArrayElements(data, dptr, JNI_ABORT); return nullptr; }

    // build ncnn::Mat input [W,H,C] planar CHW
    // ndata = 1*C*H*W ; we don't know H,W; we create (W=H=sq, C=3) — but det/lm differ.
    // Simpler: pass dims via data length? We know inW/inH at init time -> store on Net? no.
    // Use: channels=3, plane = ndata/3.
    int C = 3;
    int plane = ndata / C;
    int side = (int)std::sqrt((double)plane);
    ncnn::Mat in(side, side, C);
    size_t ps = (size_t)(side * side);
    for (int c = 0; c < C; c++) memcpy((float*)in.channel(c), dptr + (size_t)c*ps, ps*sizeof(float));

    ncnn::Extractor ex = net->create_extractor();
    if (in) {
        int r = ex.input(in, in);
        if (r != 0) { LOGW("extractor input fail %d", r);
            env->ReleaseStringUTFChars(inputName, in);
            env->ReleaseFloatArrayElements(data, dptr, JNI_ABORT);
            return nullptr; }
    }
    if (in) env->ReleaseStringUTFChars(inputName, in);
    env->ReleaseFloatArrayElements(data, dptr, JNI_ABORT);

    // run all named outputs
    std::vector<jfloat> out;
    std::vector<int> dimsOut(nNames, 0);
    for (jsize i = 0; i < nNames; i++) {
        jstring js = (jstring)env->GetObjectArrayElement(names, i);
        const char* nm = js ? env->GetStringUTFChars(js, nullptr) : nullptr;
        ncnn::Mat m;
        int rc = ex.extract(nm, m);
        if (rc == 0) {
            const float* mp = (const float*)m.data;
            size_t t = m.total();
            dimsOut[i] = (int)t;
            for (size_t k = 0; k < t; k++) out.push_back(mp[k]);
        } else {
            dimsOut[i] = 0;
            LOGW("extract %s fail rc=%d", nm ? nm : "?", rc);
        }
        if (js) env->ReleaseStringUTFChars(js, nm);
    }

    // copy dims
    jint* ds = env->GetIntArrayElements(dims, nullptr);
    for (jsize i = 0; i < nNames && i < env->GetArrayLength(dims); i++) ds[i] = dimsOut[i];
    env->ReleaseIntArrayElements(dims, ds, 0);

    jfloatArray ret = env->NewFloatArray((jsize)out.size());
    if (ret && !out.empty()) env->SetFloatArrayRegion(ret, 0, (jsize)out.size(), out.data());
    return ret;
}

}  // extern "C"