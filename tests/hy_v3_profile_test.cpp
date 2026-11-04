// CPU-only: accounting is reset per decode phase, zero-step handling and invalid samples.
#include "../src/poly/hy_v3_profile.hpp"
#include <cassert>
#include <limits>

int main() {
    strata::poly::HyV3ProfileTotals profile;
    assert(profile.steps == 0 && profile.per_step(10) == 0);
    profile.add_step(48 * .05, .12);
    profile.add_step(48 * .10, .08);
    assert(profile.steps == 2);
    assert(std::abs(profile.per_step(profile.attention_gpu_ms) - 3.6) < 1e-6);
    assert(std::abs(profile.per_step(profile.collect_host_ms) - .1) < 1e-6);
    // The request's first sampled eval never inherits the previous request's sums.
    profile = {};
    assert(profile.steps == 0 && profile.attention_gpu_ms == 0);
    for (int i = 0; i < 33; ++i) {
        const bool sampled = profile.wants_sample();
        profile.add_step(sampled ? 3. : 0., 0, sampled);
    }
    assert(profile.steps == 33 && profile.samples == 3);
    assert(profile.per_sample(profile.attention_gpu_ms) == 3.);
    // Each stage uses the same GPU samples, not the number of all decode steps.
    profile = {};
    for (int i = 0; i < 33; ++i) {
        const bool sampled = profile.wants_sample();
        profile.add_step(sampled ? 6. : 0., .01, sampled,
                         sampled ? 2. : 0., sampled ? 3. : 0., sampled ? 1. : 0.);
    }
    assert(profile.steps == 33 && profile.samples == 3);
    assert(profile.per_sample(profile.qkv_gpu_ms) == 2.);
    assert(profile.per_sample(profile.core_gpu_ms) == 3.);
    assert(profile.per_sample(profile.output_gpu_ms) == 1.);
    assert(profile.qkv_gpu_ms + profile.core_gpu_ms + profile.output_gpu_ms == profile.attention_gpu_ms);
    profile = {};
    assert(profile.samples == 0 && profile.qkv_gpu_ms == 0 && profile.core_gpu_ms == 0 && profile.output_gpu_ms == 0);
    for (double bad : {-1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        bool threw = false;
        try { profile.add_step(bad, 0); }
        catch (const std::runtime_error&) { threw = true; }
        assert(threw && profile.steps == 0);
        for (int stage = 0; stage < 3; ++stage) {
            double parts[3] = {};
            parts[stage] = bad;
            threw = false;
            try { profile.add_step(0, 0, true, parts[0], parts[1], parts[2]); }
            catch (const std::runtime_error&) { threw = true; }
            // Reject an invalid interval before mutating any totals.
            assert(threw && profile.steps == 0 && profile.samples == 0 && profile.attention_gpu_ms == 0);
            assert(profile.qkv_gpu_ms == 0 && profile.core_gpu_ms == 0 && profile.output_gpu_ms == 0);
        }
    }
}
