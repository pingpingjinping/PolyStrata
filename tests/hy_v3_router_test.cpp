// Standalone, CUDA-free test: c++ -std=c++17 -O2 tests/hy_v3_router_test.cpp -o hy_v3_router_test
#include "../src/poly/hy_v3_router.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

using strata::poly::hyv3::route;

static bool near(float a, float b, float tolerance = 1e-6f) {
    return std::fabs(a - b) <= tolerance;
}

int main() {
    // A selection bias may change the winner but MUST NOT change the output weight.
    const auto with_bias = route({2.0f, 0.0f, -2.0f}, {-1.0f, 0.6f, 0.0f}, 1, false, 1.0f);
    assert(with_bias.experts.size() == 1 && with_bias.experts[0] == 1);
    assert(near(with_bias.weights[0], 0.5f)); // NOT sigmoid(0) + 0.6

    // Normalized selected weights sum to the configured scale, not one.
    const auto normalized = route({2.0f, 0.0f, -2.0f}, {}, 2, true, 2.5f);
    assert((normalized.experts == std::vector<int>{0, 1}));
    assert(near(normalized.weights[0] + normalized.weights[1], 2.5f));

    // Without normalization, the original sigmoid weights are preserved.
    const auto raw = route({0.0f, 0.0f}, {}, 2, false, 2.0f);
    assert((raw.experts == std::vector<int>{0, 1}));
    assert(near(raw.weights[0], 1.0f) && near(raw.weights[1], 1.0f));

    // Stable extreme values and deterministic ties.
    const auto extreme = route({-1000.0f, 1000.0f}, {}, 2, true, 1.0f);
    assert((extreme.experts == std::vector<int>{1, 0}));
    assert(near(extreme.weights[0], 1.0f));
    assert(near(extreme.weights[1], 0.0f));
    const auto ties = route({0.0f, 0.0f, 0.0f}, {}, 2, true, 1.0f);
    assert((ties.experts == std::vector<int>{0, 1}));

    bool rejected = false;
    try { (void)route({0.0f, 1.0f}, {1.0f}, 1, true, 1.0f); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    return 0;
}
