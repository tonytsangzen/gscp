#include "net.h"
#include "mat.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

static float genInput(size_t i) {
    // deterministic, same formula as the python oracle
    return (float)(( (i * 7u) % 1000u )) / 1000.0f - 0.5f;
}

static void dump(const char* name, const ncnn::Mat& m) {
    if (m.empty()) { printf("%s: EMPTY\n", name); return; }
    printf("%s: dims=%d w=%d h=%d c=%d d=%d elemsize=%d total=%zu\n",
           name, m.dims, m.w, m.h, m.c, m.d, m.elemsize, m.total());
    const float* p = (const float*)m.data;
    size_t n = m.total(); if (n > 0) {
        printf("   first: ");
        for (int k = 0; k < 6 && (size_t)k < n; k++) printf("%.5f ", p[k]);
        printf("| last: ");
        for (int k = 4; k >= 0 && (size_t)(n-1-k) < n; k--) printf("%.5f ", p[n-1-k]);
        printf("\n");
    }
}

int main(int argc, char** argv) {
    const char* param = "/tmp/ncnn-out/det_480.param";
    const char* bin   = "/tmp/ncnn-out/det_480.bin";
    const int W = 480, H = 480, C = 3;
    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.num_threads = 1;
    if (net.load_param(param) != 0 || net.load_model(bin) != 0) {
        printf("load FAILED\n"); return 1;
    }
    printf("loaded\n");

    std::vector<float> chw((size_t)C * H * W);
    for (size_t i = 0; i < chw.size(); i++) chw[i] = genInput(i);

    ncnn::Mat in(W, H, C);
    size_t plane = (size_t)H * W;
    for (int c = 0; c < C; c++)
        memcpy((float*)in.channel(c), chw.data() + c * plane, plane * sizeof(float));

    ncnn::Extractor ex = net.create_extractor();
    ex.input("input.1", in);
    const char* outs[] = {"471","474","477","494","497","500"};
    for (auto o : outs) { ncnn::Mat m; int rc = ex.extract(o, m); printf("rc(%s)=%d\n", o, rc); dump(o, m); }
    return 0;
}