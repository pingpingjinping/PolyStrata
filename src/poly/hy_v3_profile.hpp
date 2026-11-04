// Decode accounting independent of CUDA; host phases overlap device work.
#pragma once
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace strata::poly {

struct HyV3ProfileTotals {
    static constexpr int kStride = 16;
    int64_t steps = 0;
    int64_t samples = 0;
    double attention_gpu_ms = 0, collect_host_ms = 0;

    bool wants_sample() const { return steps % kStride == 0; }

    void add_step(double attention, double collect, bool sampled = true) {
        if (!std::isfinite(attention) || !std::isfinite(collect) || attention < 0 || collect < 0)
            throw std::runtime_error("invalid HY-V3 profile sample");
        ++steps;
        if (sampled) { ++samples; attention_gpu_ms += attention; }
        collect_host_ms += collect;
    }

    double per_step(double ms) const { return steps > 0 ? ms / (double) steps : 0; }
    double per_sample(double ms) const { return samples > 0 ? ms / (double) samples : 0; }
};

}  // namespace strata::poly
