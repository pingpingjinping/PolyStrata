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
    profile = {};
    for (double bad : {-1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        bool threw = false;
        try { profile.add_step(bad, 0); }
        catch (const std::runtime_error&) { threw = true; }
        assert(threw && profile.steps == 0);
    }
}
