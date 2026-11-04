// Opt-in single-token HY-V3 projections using the existing MMVQ slots kernel.
#pragma once
#include "strata/kernels/native_mmvq.hpp"
#include <cstddef>
#include <cstdint>

namespace strata::poly {

// Caller validates Q/K are Q4_K, V is Q4_K or Q6_K, and the fixed Hy geometry.
// Q's 4096 rows are eight contiguous 512-row views, not copies or re-quantized
// weights. Every view uses the same input and writes its original output rows.
inline bool hy_v3_qkv_slots(const void* wq, const void* wk, const void* wv, int v_type,
                            const void* xq, float* q, float* k, float* v, void* stream) {
    if (v_type != 12 && v_type != 14) return false;
    constexpr int width = 2048, rows = 512, q_parts = 8;
    const std::size_t bytes = kernels::native_mmvq_weight_bytes(12, width, rows);
    kernels::MmvqSlots slots{};
    for (int i = 0; i < q_parts; ++i) {
        slots.w[i] = (const uint8_t*) wq + (std::size_t) i * bytes;
        slots.x[i] = xq;
        slots.y[i] = q + i * rows;
    }
    slots.w[8] = wk;
    slots.x[8] = xq;
    slots.y[8] = k;
    if (v_type == 12) {
        slots.w[9] = wv;
        slots.x[9] = xq;
        slots.y[9] = v;
    }
    // The API guarantees false means nothing was launched; caller can fall back.
    if (!kernels::native_mmvq_slots(12, slots, v_type == 12 ? 10 : 9, width, rows, stream)) return false;
    if (v_type == 14) kernels::native_mmvq(14, wv, xq, v, width, rows, 1, stream);
    return true;
}

}  // namespace strata::poly
