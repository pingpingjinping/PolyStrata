// Actual ggml quantized CPU experts: 16/32-row schedules must be bit-identical.
#include "cpu_experts.hpp"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

using namespace strata::poly;

static void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }

static std::vector<uint8_t> matrix(ggml_type type, int cols, int rows, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-.15f, .15f);
    std::vector<float> values((size_t) cols * rows);
    for (float& value : values) value = dist(rng);
    std::vector<uint8_t> bytes((size_t) rows * ggml_row_size(type, cols));
    require(ggml_quantize_chunk(type, values.data(), bytes.data(), 0, rows, cols, nullptr) == bytes.size(),
            "wrong quantized matrix size");
    return bytes;
}

static int compare(ggml_type gate_type, ggml_type up_type, ggml_type down_type, int width, int ff) {
    struct Weights { std::vector<uint8_t> gate, up, down; } weights[3];
    std::mt19937 rng(67231);
    for (auto& w : weights) {
        w.gate = matrix(gate_type, width, ff, rng);
        w.up = matrix(up_type, width, ff, rng);
        w.down = matrix(down_type, ff, width, rng);
    }
    int checks = 0;
    for (int threads : {1, 4, 6, 8}) {
        Workers workers(threads);
        for (int rows : {1, 2}) for (int used : {1, 2, 3}) for (bool gelu : {false, true}) {
            std::vector<CpuExpert> uses;
            std::vector<float> x((size_t) rows * width);
            for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin((float) i * .17f) * .2f;
            for (int row = 0; row < rows; ++row) for (int k = 0; k < used; ++k) {
                const auto& w = weights[(row + k) % 3];  // includes reused weights across different rows
                uses.push_back({w.gate.data(), w.up.data(), w.down.data(), 1.f / (k + 2), row});
            }
            const int n = (int) uses.size();
            std::vector<uint8_t> late((size_t) n);
            for (int i = 0; i < n; ++i) late[(size_t) i] = (uint8_t) (i % 2);
            std::atomic<int> old_launch{0}, new_launch{0}, old_wait{0}, new_wait{0};
            const std::function<void()> a = [&] { ++old_launch; }, b = [&] { ++new_launch; };
            const std::function<void(int)> wa = [&](int) { ++old_wait; }, wb = [&](int) { ++new_wait; };
            std::vector<float> old(x.size(), std::numeric_limits<float>::quiet_NaN()), changed = old;
            cpu_experts(workers, uses.data(), n, rows, gate_type, up_type, down_type, width, ff,
                        x.data(), 1.5f, gelu, old.data(), &a, late.data(), &wa);
            cpu_experts_down32(workers, uses.data(), n, rows, gate_type, up_type, down_type, width, ff,
                               x.data(), 1.5f, gelu, changed.data(), &b, late.data(), &wb);
            require(std::all_of(old.begin(), old.end(), [](float v) { return std::isfinite(v); }), "nonfinite baseline");
            require(std::memcmp(old.data(), changed.data(), old.size() * sizeof(float)) == 0, "16/32 CPU output differs");
            require(old_launch == 1 && new_launch == 1, "GPU launch callback must run exactly once");
            require(old_wait == new_wait, "late expert wait accounting changed");
            checks += 3;
        }
    }
    return checks;
}

int main() {
    try {
        int checks = 0;
        checks += compare(GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, 2048, 768);
        checks += compare(GGML_TYPE_Q6_K, GGML_TYPE_Q6_K, GGML_TYPE_Q6_K, 2048, 768);
        checks += compare(GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, 2048, 768);
        checks += compare(GGML_TYPE_Q4_K, GGML_TYPE_F32, GGML_TYPE_Q6_K, 512, 768);  // separate up activations
        checks += compare(GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, 49, 17);       // partial final task
        std::printf("CPU down32 parity: %d checks passed (real ggml quantization and worker threads)\n", checks);
        ggml_quantize_free();
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
