// Shared output normalisation for ncnn benchmark comparisons.
//
// Header-only so the same code runs on-device (ncnn_bench.cpp) and in the host
// test (model/test_out_norm.cpp): backends hand back network outputs in
// different representations, and reading them without normalising is how a
// comparison ends up reporting inf/nan instead of a number.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "ncnn/mat.h"

namespace nbench {

// One network output, always fp32 / unpacked / unpadded. A Vulkan or CPU-FP16
// net hands back float16 with 4-wide channel packing and channel counts rounded
// up to a multiple of 4, which must not be read as fp32.
struct FlatOut {
    int w = 0, h = 0, c = 0;
    std::vector<float> v;
    // Values a backend handed back that are not a number. A NaN here is a
    // numerical failure of that backend, not a difference to average over.
    int nonFinite = 0;
    // Sum of magnitudes, so two runs can be shown to differ in aggregate even
    // where a pairwise diff is reported as zero.
    double sumAbs = 0;
};

inline FlatOut flatten(const ncnn::Mat& y)
{
    ncnn::Mat m = y;
    if (m.elempack != 1) {
        ncnn::Mat u;
        ncnn::convert_packing(m, u, 1);
        m = u;
    }
    if (m.elemsize == 2u) {
        ncnn::Mat f;
        ncnn::cast_float16_to_float32(m, f);
        m = f;
    }
    FlatOut o;
    o.w = m.w;
    o.h = m.h;
    o.c = m.c;
    const size_t plane = (size_t)m.w * m.h * (m.d > 1 ? m.d : 1);
    o.v.resize(plane * m.c);
    for (int q = 0; q < m.c; q++)
        memcpy(o.v.data() + (size_t)q * plane, m.channel(q), plane * sizeof(float));
    for (float x : o.v) {
        if (std::isfinite(x)) o.sumAbs += std::fabs(x);
        else o.nonFinite++;
    }
    return o;
}

struct OutDiff {
    double maxAbs = 0, meanAbs = 0;
    long elements = 0;
    int compared = 0, skipped = 0;
    // Differences that are not finite. They are counted, never averaged in: a
    // NaN added to a running max silently replaces it with whatever element
    // happened to come next, which reads as a plausible number but is not one.
    long badDiffs = 0;
};

inline OutDiff compareOutputs(const std::vector<FlatOut>& ref, const std::vector<FlatOut>& got)
{
    OutDiff d;
    double sum = 0;
    for (size_t i = 0; i < ref.size() && i < got.size(); i++) {
        const FlatOut& a = ref[i];
        const FlatOut& b = got[i];
        if (a.w != b.w || a.h != b.h) { d.skipped++; continue; }
        const int c = std::min(a.c, b.c);      // extra channels are packing padding
        const size_t plane = (size_t)a.w * a.h;
        for (int q = 0; q < c; q++) {
            for (size_t k = 0; k < plane; k++) {
                const double e = std::fabs((double)a.v[(size_t)q * plane + k]
                                           - b.v[(size_t)q * plane + k]);
                d.elements++;
                if (!std::isfinite(e)) { d.badDiffs++; continue; }
                if (e > d.maxAbs) d.maxAbs = e;
                sum += e;
            }
        }
        d.compared++;
    }
    const long good = d.elements - d.badDiffs;
    if (good) d.meanAbs = sum / good;
    return d;
}

}  // namespace nbench
