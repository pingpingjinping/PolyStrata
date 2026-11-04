// Real GPU bitwise check for grouped HY-V3 QKV projections; no model required.
#include "hy_v3_qkv.hpp"
#include "ggml.h"
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace strata;
constexpr int width = 2048, q_rows = 4096, kv_rows = 512, outputs = q_rows + 2 * kv_rows, guard = 16;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }

struct Device {
    void* p = nullptr;
    explicit Device(size_t bytes) { ck(cudaMalloc(&p, bytes), "allocate"); }
    ~Device() { if (p) cudaFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    float* f() const { return (float*) p; }
};
struct Stream {
    cudaStream_t s = nullptr;
    Stream() { ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "create stream"); }
    ~Stream() { if (s) cudaStreamDestroy(s); }
};
struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    ~Graph() {
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
    }
};

std::vector<uint8_t> weights(ggml_type type, int rows, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-.15f, .15f);
    std::vector<float> source((size_t) rows * width);
    for (float& x : source) x = dist(rng);
    std::vector<uint8_t> result((size_t) rows * ggml_row_size(type, width));
    require(ggml_quantize_chunk(type, source.data(), result.data(), 0, rows, width, nullptr) == result.size(),
            "quantized matrix size mismatch");
    return result;
}

int check(ggml_type v_type) {
    std::mt19937 rng(70913 + (int) v_type);
    const auto wq = weights(GGML_TYPE_Q4_K, q_rows, rng);
    const auto wk = weights(GGML_TYPE_Q4_K, kv_rows, rng);
    const auto wv = weights(v_type, kv_rows, rng);
    Stream stream;
    Device dq(wq.size()), dk(wk.size()), dv(wv.size()), dx(width * sizeof(float));
    Device xq(kernels::native_q8_1_bytes(width));
    Device baseline((outputs + guard) * sizeof(float)), changed((outputs + guard) * sizeof(float));
    ck(cudaMemcpyAsync(dq.p, wq.data(), wq.size(), cudaMemcpyHostToDevice, stream.s), "upload Q");
    ck(cudaMemcpyAsync(dk.p, wk.data(), wk.size(), cudaMemcpyHostToDevice, stream.s), "upload K");
    ck(cudaMemcpyAsync(dv.p, wv.data(), wv.size(), cudaMemcpyHostToDevice, stream.s), "upload V");
    auto separate = [&] {
        kernels::native_mmvq(12, dq.p, xq.p, baseline.f(), width, q_rows, 1, stream.s);
        kernels::native_mmvq(12, dk.p, xq.p, baseline.f() + q_rows, width, kv_rows, 1, stream.s);
        kernels::native_mmvq(v_type, dv.p, xq.p, baseline.f() + q_rows + kv_rows, width, kv_rows, 1, stream.s);
    };
    auto grouped = [&] {
        require(poly::hy_v3_qkv_slots(dq.p, dk.p, dv.p, v_type, xq.p, changed.f(),
                                     changed.f() + q_rows, changed.f() + q_rows + kv_rows, stream.s),
                "grouped QKV kernel unavailable");
    };
    // Resolve lazy kernel setup before capture; capture only the real grouped calls.
    ck(cudaMemsetAsync(xq.p, 0, kernels::native_q8_1_bytes(width), stream.s), "warm input");
    separate();
    grouped();
    ck(cudaStreamSynchronize(stream.s), "warm kernels");
    Graph graph;
    ck(cudaStreamBeginCapture(stream.s, cudaStreamCaptureModeGlobal), "begin capture");
    grouped();
    ck(cudaStreamEndCapture(stream.s, &graph.graph), "end capture");
    ck(cudaGraphInstantiateWithFlags(&graph.exec, graph.graph, 0), "instantiate graph");

    std::vector<float> x(width), expected(outputs + guard), actual(outputs + guard);
    int checks = 0;
    for (int input = 0; input < 4; ++input) {
        for (int i = 0; i < width; ++i) {
            const float wave = std::sin((float) (i + 1) * .013f);
            x[(size_t) i] = input == 0 ? 0.f : wave * (input == 1 ? .2f : input == 2 ? .0001f : 50.f);
        }
        ck(cudaMemcpyAsync(dx.p, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice, stream.s), "upload input");
        kernels::native_quantize_q8_1(dx.f(), xq.p, width, 1, stream.s);
        ck(cudaMemsetAsync(baseline.p, 0xff, expected.size() * sizeof(float), stream.s), "clear baseline");
        separate();
        ck(cudaMemcpyAsync(expected.data(), baseline.p, expected.size() * sizeof(float),
                           cudaMemcpyDeviceToHost, stream.s), "read baseline");
        ck(cudaStreamSynchronize(stream.s), "baseline sync");
        // Reuse the same graph with different inputs and with repeated replay.
        for (int mode = 0; mode < 3; ++mode) {
            ck(cudaMemsetAsync(changed.p, 0xff, actual.size() * sizeof(float), stream.s), "clear candidate");
            if (mode == 0) grouped();
            else ck(cudaGraphLaunch(graph.exec, stream.s), "replay graph");
            ck(cudaMemcpyAsync(actual.data(), changed.p, actual.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream.s), "read candidate");
            ck(cudaStreamSynchronize(stream.s), "candidate sync");
            for (int i = 0; i < outputs; ++i) {
                if (!std::isfinite(expected[(size_t) i]) ||
                    std::memcmp(&expected[(size_t) i], &actual[(size_t) i], sizeof(float)) != 0) {
                    std::fprintf(stderr, "Mismatch V type=%d input=%d mode=%d output=%d: stock=%.9g grouped=%.9g\n",
                                 (int) v_type, input, mode, i, expected[(size_t) i], actual[(size_t) i]);
                    throw std::runtime_error("QKV outputs are not bit-identical");
                }
            }
            require(std::memcmp(expected.data() + outputs, actual.data() + outputs, guard * sizeof(float)) == 0,
                    "QKV wrote beyond the output rows");
            ++checks;
        }
    }
    return checks;
}
}  // namespace

int main() {
    try {
        int count = 0;
        ck(cudaGetDeviceCount(&count), "CUDA devices");
        require(count > 0, "no CUDA GPU; compile-only is not a parity pass");
        ck(cudaSetDevice(0), "select GPU");
        cudaDeviceProp prop{};
        ck(cudaGetDeviceProperties(&prop, 0), "GPU properties");
        std::printf("GPU: %s; slots warp mode=%d\n", prop.name, (int) strata::kernels::native_mmvq_slots_wide());
        const int checks = check(GGML_TYPE_Q4_K) + check(GGML_TYPE_Q6_K);
        ggml_quantize_free();
        std::printf("HY-V3 QKV slots GPU parity: PASS (%d cases, %d output values bit-identical; eager + graph replay)\n",
                    checks, checks * outputs);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "HY-V3 QKV slots GPU parity: FAIL: %s\n", e.what());
        return 1;
    }
}
