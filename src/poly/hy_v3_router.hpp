// Experimental HY-V3 routing reference. This is not yet wired into Engine::open().
// Keep expert selection bias separate from the actual expert output weights.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace strata::poly::hyv3 {

struct Routing {
    std::vector<int> experts;
    std::vector<float> weights;
};

// Mirrors the sigmoid / selection-bias / top-k / optional renormalization
// ordering used by llama.cpp's build_moe_ffn for HY-V3.
// CPU reference for testing; production GPU and batch routing must be added separately.
inline Routing route(const std::vector<float>& logits,
                     const std::vector<float>& selection_bias,
                     std::size_t top_k,
                     bool normalize,
                     float weight_scale) {
    const std::size_t n = logits.size();
    if (n == 0 || top_k == 0 || top_k > n)
        throw std::invalid_argument("invalid HY-V3 expert count/top_k");
    if (!selection_bias.empty() && selection_bias.size() != n)
        throw std::invalid_argument("HY-V3 expert selection bias size mismatch");
    if (!std::isfinite(weight_scale))
        throw std::invalid_argument("non-finite HY-V3 expert weight scale");

    std::vector<float> probs(n), chosen(n);
    for (std::size_t i = 0; i < n; ++i) {
        const float x = logits[i];
        if (!std::isfinite(x))
            throw std::invalid_argument("non-finite HY-V3 router logit");
        const float sigmoid = x >= 0.0f ? 1.0f / (1.0f + std::exp(-x))
                                         : std::exp(x) / (1.0f + std::exp(x));
        probs[i] = sigmoid;
        const float bias = selection_bias.empty() ? 0.0f : selection_bias[i];
        if (!std::isfinite(bias))
            throw std::invalid_argument("non-finite HY-V3 expert selection bias");
        chosen[i] = sigmoid + bias;
    }

    std::vector<int> sorted(n);
    std::iota(sorted.begin(), sorted.end(), 0);
    std::partial_sort(sorted.begin(), sorted.begin() + top_k, sorted.end(),
                      [&](int a, int b) {
                          return chosen[a] > chosen[b] ||
                                 (chosen[a] == chosen[b] && a < b);
                      });
    Routing out;
    out.experts.assign(sorted.begin(), sorted.begin() + top_k);
    out.weights.reserve(top_k);
    float sum = 0.0f;
    for (int id : out.experts) {
        out.weights.push_back(probs[static_cast<std::size_t>(id)]);
        sum += probs[static_cast<std::size_t>(id)];
    }
    const float factor = (normalize && sum > 0.0f ? 1.0f / sum : 1.0f) * weight_scale;
    for (float& w : out.weights)
        w *= factor;
    return out;
}

}  // namespace strata::poly::hyv3
