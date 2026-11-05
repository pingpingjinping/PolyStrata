// Decode accounting independent of CUDA; host phases overlap device work.
#pragma once
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>

namespace strata::poly {

struct HyV3ProfileTotals {
    static constexpr int kStride = 16;
    int64_t steps = 0;
    int64_t samples = 0;
    double attention_gpu_ms = 0, collect_host_ms = 0;
    double qkv_gpu_ms = 0, core_gpu_ms = 0, output_gpu_ms = 0;

    bool wants_sample() const { return steps % kStride == 0; }

    void add_step(double attention, double collect, bool sampled = true,
                  double qkv = 0, double core = 0, double output = 0) {
        for (double ms : {attention, collect, qkv, core, output})
            if (!std::isfinite(ms) || ms < 0)
                throw std::runtime_error("invalid HY-V3 profile sample");
        ++steps;
        if (sampled) {
            ++samples;
            attention_gpu_ms += attention;
            qkv_gpu_ms += qkv;
            core_gpu_ms += core;
            output_gpu_ms += output;
        }
        collect_host_ms += collect;
    }

    double per_step(double ms) const { return steps > 0 ? ms / (double) steps : 0; }
    double per_sample(double ms) const { return samples > 0 ? ms / (double) samples : 0; }
};

}  // namespace strata::poly
