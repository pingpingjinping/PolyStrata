// src/poly/flint.cpp - the layers of HyV3.  The reference for the math is llama.cpp's flint graph
// (src/models/flint.cpp, a37cc46104).
#include "hy_v3.hpp"

#include "runtime.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {

constexpr float kMinWeightSum = 6.103515625e-5f;   // the floor ggml's MoE graph puts under the gate sum
constexpr size_t kScoreFloats = size_t(16) << 20;  // the attention's scores of the tokens it takes at a time

int64_t of_layer(const MetaValue& v, int il) {
    if (v.type != MetaType::ARRAY) return (int64_t) v.u;
    if ((size_t) il >= v.items.size()) throw std::runtime_error("a per-layer value of the GGUF ends before its layers");
    return (int64_t) v.items[(size_t) il].u;
}

}  // namespace

void HyV3::begin_pass() { blocks_.assign(layers_.size(), Block{}); }

void HyV3::load_geometry() {
    // HY-V3 has an optional NextN/MTP block in the GGUF. The trunk is always
    // loaded without speculative decoding until we implement and parity-test
    // the HY-V3 draft block separately.
    opt_.draft = false;
    load_shape();
    auto find = [&](const char* key) { return meta_in(*gg_, key); };
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported HY-V3 model: ") + what);
    };

    // The first native target is Tencent Hy-MT2-30B-A3B, not all HY-V3 sizes.
    // The Q/K head norms and full-context RoPE are also required by this shape.
    need(n_layer_ == 48, "expected 48 trunk layers");
    need(n_embd_ == 2048 && n_expert_ == 128 && n_used_ == 8,
         "expected Hy-MT2-30B-A3B (2048 hidden, 128 experts, top-8)");
    n_head_ = shape_i("attention.head_count");
    n_kv_ = shape_i("attention.head_count_kv");
    head_dim_ = shape_i("attention.key_length");
    need(n_head_ == 32 && n_kv_ == 4 && head_dim_ == 128,
         "expected 32 query heads, 4 KV heads and 128-dimensional heads");
    need(shape_i("attention.value_length") == head_dim_,
         "keys and values have different widths");

    const MetaValue* gate = find("expert_gating_func");
    need(!gate || (int64_t) gate->u == 2, "expected sigmoid expert gating");
    const MetaValue* dense = find("leading_dense_block_count");
    n_dense_ = dense ? (int) dense->u : (tensor(0, "ffn_gate_inp.weight") ? 0 : 1);
    need(n_dense_ == 1, "expected one leading dense FFN block");
    n_ff_ = shape_i("feed_forward_length");
    need(n_ff_ == 6912 && n_ff_exp_ == 768 && n_ff_shexp_ == 768,
         "unexpected dense/routed/shared FFN widths");

    // The original Hy-MT2 config sets route_norm=true and scaling_factor=2.826.
    // GGUF metadata should preserve these fields; missing fields take the
    // known Hy-MT2 values, not Flint's legacy defaults.
    const MetaValue* norm = find("expert_weights_norm");
    w_norm_ = norm ? norm->u != 0 : true;
    const MetaValue* scale = find("expert_weights_scale");
    w_scale_ = scale ? (float) scale->num() : 2.826f;
    need(w_norm_ && std::isfinite(w_scale_) && w_scale_ > 0,
         "expected normalized expert weights with a positive scaling factor");

    theta_scale_ = std::pow(shape_f("rope.freq_base"), -2.f / (float) head_dim_);
    need(std::isfinite(theta_scale_) && theta_scale_ > 0,
         "invalid RoPE frequency base");

    // No sliding-window layers, sink logits or clamp in Hy-MT2. The MTP
    // block is excluded from n_layer_ and is not executed as a trunk layer.
    window_ = 0;
    clamp_exp_.assign((size_t) n_layer_ + 1, INFINITY);
    clamp_shexp_ = clamp_exp_;
}

void HyV3::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(il, n), blk(il, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };

    B.swa = false;
    B.rope = true;
    L.fixed = true;
    L.attn_norm = f32("attn_norm.weight", n_embd_);
    L.ffn_norm = f32("ffn_norm.weight", n_embd_);
    B.wq = mat("attn_q.weight", n_embd_, (int64_t) n_head_ * head_dim_);
    B.wk = mat("attn_k.weight", n_embd_, (int64_t) n_kv_ * head_dim_);
    B.wv = mat("attn_v.weight", n_embd_, (int64_t) n_kv_ * head_dim_);
    B.q_norm = f32("attn_q_norm.weight", head_dim_);
    B.k_norm = f32("attn_k_norm.weight", head_dim_);
    B.wo = mat("attn_output.weight", (int64_t) n_head_ * head_dim_, n_embd_);

    if (il < n_dense_) {
        B.ffn_gate = mat("ffn_gate.weight", n_embd_, n_ff_);
        B.ffn_up = mat("ffn_up.weight", n_embd_, n_ff_);
        B.ffn_down = mat("ffn_down.weight", n_ff_, n_embd_);
        return;
    }
    load_moe(il, true);
    // Older converters use "exp_probs_b.bias", newer llama.cpp checkpoints
    // may use "exp_probs_b" with no suffix. This is a selection-only bias:
    // weight calculation uses the original, unbiased sigmoid probabilities.
    const TensorInfo* b = tensor(il, "exp_probs_b");
    const char* suffix = "exp_probs_b";
    if (!b) {
        b = tensor(il, "exp_probs_b.bias");
        suffix = "exp_probs_b.bias";
    }
    if (!b)
        throw std::runtime_error("HY-V3 is missing routed-expert selection bias " + blk(il, suffix));
    if (b->type != kTypeF32 || (int64_t) b->elements() != n_expert_)
        throw std::runtime_error(blk(il, suffix) + " is " + shape_of(*b) + ", expected F32 [" +
                                 std::to_string(n_expert_) + "]");
    B.probs_b.assign((size_t) n_expert_, 0.f);
    std::memcpy(B.probs_b.data(), home(*b).data(*b), (size_t) n_expert_ * sizeof(float));
    B.probs_b_dev = upload(b, blk(il, suffix), n_expert_, 1, kF32);
}

// The caches: a full layer keeps a row a position, a window layer a ring of the rows a call can read.
void HyV3::alloc_caches() {
    ring_ = std::min(opt_.max_context, window_ + rows_);
    keep_ = std::min(window_, ring_);
    const size_t row = (size_t) n_kv_ * head_dim_;
    size_t rings = 0;   // a window layer's rows are a ring, whatever the sequence's length
    for (int il = 0; il < (int) blocks_.size(); ++il)
        if (runs(il) && blocks_[(size_t) il].swa) rings += 2 * (size_t) ring_ * row;
    uint16_t* kv = rings ? (uint16_t*) seq_alloc(rings * sizeof(uint16_t), "the attention cache") : nullptr;
    for (int il = 0; il < (int) blocks_.size(); ++il) {
        Block& B = blocks_[(size_t) il];
        if (!runs(il)) continue;
        if (B.swa) {
            B.keys = kv;
            kv += ring_ * row;
            B.vals = kv;
            kv += ring_ * row;
        } else {
            B.keys = (uint16_t*) kv_alloc(row * sizeof(uint16_t), 1);
            B.vals = (uint16_t*) kv_alloc(row * sizeof(uint16_t), 1);
        }
    }
    chunk_ = (int) std::clamp<size_t>(kScoreFloats / ((size_t) n_head_ * opt_.max_context), 1, (size_t) rows_);
}

// Per-token buffers hold rows_ tokens, row after row.
void HyV3::assign_scratch() {
    const size_t n = (size_t) rows_, heads = n * n_head_ * head_dim_, kv = n * n_kv_ * head_dim_;
    qraw_ = fbuf(heads);
    q_ = fbuf(heads);
    kraw_ = fbuf(kv);
    k_ = fbuf(kv);
    v_ = fbuf(kv);
    att_ = fbuf(heads);
    score_n_ = (size_t) chunk_ * n_head_ * opt_.max_context;
    score_ = fbuf(score_n_);
    if (rows_ >= kStageRows) awork_ = fbuf(heads + heads / 2 + 1 + 3 * n * n_head_);   // a prompt batch's attention
}

// A sequence's rows are the full layers': a window layer's are part of its state (save_layer()).
size_t HyV3::card_row_bytes() const {
    size_t layers = 0;
    for (const Block& B : blocks_)
        if (B.keys && !B.swa) ++layers;
    return layers * 2 * (size_t) n_kv_ * head_dim_ * sizeof(uint16_t);
}

void HyV3::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t), at = (size_t) first * row, bytes = (size_t) n * row;
    for (const Block& B : blocks_) {
        if (!B.keys || B.swa || bytes == 0) continue;
        for (const uint16_t* cache : {B.keys, B.vals}) {
            ck(cudaMemcpy(dst, (const uint8_t*) cache + at, bytes, cudaMemcpyDeviceToHost), "save rows");
            dst += bytes;
        }
    }
}

void HyV3::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t), at = (size_t) first * row, bytes = (size_t) n * row;
    const size_t skip = (size_t) (first - from) * row, all = (size_t) count * row;
    for (const Block& B : blocks_) {
        if (!B.keys || B.swa || bytes == 0) continue;
        for (uint16_t* cache : {B.keys, B.vals}) {
            ck(cudaMemcpy((uint8_t*) cache + at, src + skip, bytes, cudaMemcpyHostToDevice), "load rows");
            src += all;
        }
    }
}

// The last keep_ positions before `pos` as runs of a ring's rows: f(the run's place among the keep_, its first
// ring row, its rows).  A position before the sequence's start has no run.
template <class F>
void HyV3::window_runs(int64_t pos, F&& f) const {
    for (int64_t p = std::max<int64_t>(0, pos - keep_); p < pos;) {
        const int64_t at = p % ring_, m = std::min<int64_t>(pos - p, ring_ - at);
        f((size_t) (p - (pos - keep_)), (size_t) at, (size_t) m);
        p += m;
    }
}

// What a window layer has of the sequence: the rows the next token attends.
size_t HyV3::extra_bytes() const {
    size_t layers = 0;
    for (const Block& B : blocks_)
        if (B.swa) ++layers;
    return layers * 2 * (size_t) keep_ * n_kv_ * head_dim_ * sizeof(uint16_t);
}

// A window layer's rows, behind the window layers before it (there is no recurrent state).
void HyV3::save_layer(int il, int64_t pos, uint8_t* dst) {
    if (!blocks_[(size_t) il].swa) return;
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t), side = (size_t) keep_ * row;
    dst += state_n_ * sizeof(float);
    for (int j = 0; j < il; ++j)
        if (blocks_[(size_t) j].swa) dst += 2 * side;
    const Block& B = blocks_[(size_t) il];
    for (const uint16_t* cache : {B.keys, B.vals}) {
        std::memset(dst, 0, side);   // the places of positions before the sequence's start
        window_runs(pos, [&](size_t slot, size_t at, size_t m) {
            ck(cudaMemcpy(dst + slot * row, (const uint8_t*) cache + at * row, m * row, cudaMemcpyDeviceToHost),
               "save state");
        });
        dst += side;
    }
}

void HyV3::load_extra(const uint8_t* src, int64_t pos) {
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t), side = (size_t) keep_ * row;
    for (const Block& B : blocks_) {
        if (!B.swa) continue;
        for (uint16_t* cache : {B.keys, B.vals}) {
            window_runs(pos, [&](size_t slot, size_t at, size_t m) {
                ck(cudaMemcpy((uint8_t*) cache + at * row, src + slot * row, m * row, cudaMemcpyHostToDevice),
                   "load state");
            });
            src += side;
        }
    }
}

// The recordings of a full layer's head are numbered by the rows they are sized for; a window layer's never
// reads more than its window.
int HyV3::head_key(int il, int n) const { return blocks_[(size_t) il].swa ? 0 : room_key(n); }

// A side of a layer's cache: a full layer's at the positions, a window layer's at their places in the ring.
void HyV3::store(const float* x, const Block& B, int64_t pos, int n, bool value) {
    const int width = n_kv_ * head_dim_;
    uint16_t* cache = value ? B.vals : B.keys;
    if (recording_) return store_f16_at(x, cache, width, at_, n, B.swa ? ring_ : 0, st_);
    if (!B.swa) return store_f16(x, cache, width, pos, n, st_);
    const int at = (int) (pos % ring_), head = std::min(n, ring_ - at);
    store_f16(x, cache, width, at, head, st_);
    if (head < n) store_f16(x + (size_t) head * width, cache, width, 0, n - head, st_);
}

// The attention: queries and keys normed a head, and in a window layer turned by the position; the keys and
// values into the caches; each head's softmax with its sink.
void HyV3::attn(Block& B, int64_t pos, int n) {
    if (rows_of_) return attn_rows(B, n);
    const long long* at = recording_ ? at_ : nullptr;   // a recorded launch: the tokens from what at_ holds
    const int n_rot = B.rope ? head_dim_ : 0;
    lin(B.wq, xn_, qraw_, n, xq_);
    head_norm_rope(qraw_, head_dim_, fp(B.q_norm), q_, head_dim_, n_head_, n_rot, theta_scale_, pos, n, rms_eps_, st_,
                   nullptr, nullptr, 1, at ? at + 1 : nullptr);
    lin(B.wk, xn_, kraw_, n, xq_);
    head_norm_rope(kraw_, head_dim_, fp(B.k_norm), k_, head_dim_, n_kv_, n_rot, theta_scale_, pos, n, rms_eps_, st_,
                   nullptr, nullptr, 1, at ? at + 1 : nullptr);
    lin(B.wv, xn_, v_, n, xq_);
    store(k_, B, pos, n, false);
    store(v_, B, pos, n, true);
    const float scale = 1.f / std::sqrt((float) head_dim_);
    const int window = B.swa ? window_ : 0, ring = B.swa ? ring_ : 0;
    // a prompt batch with attend_batch()'s kernels; a few tokens with the ones that give a token the same bits in any
    // company
    if (n >= kStageRows)
        attend_batch(blas_, q_, B.keys, B.vals, pos, n, score_, score_n_, awork_,
                     att_, head_dim_, n_head_, n_kv_, scale, st_, window, ring, false, {}, {}, nullptr, nullptr, 0,
                     nullptr);
    else
        attend(q_, B.keys, B.vals, pos, n, score_, chunk_, att_, head_dim_, n_head_, n_kv_, scale, st_, window, ring,
               false, at, !at ? 0 : B.swa ? std::min(window_ + n - 1, opt_.max_context) : room(n), score_n_, {}, {},
               nullptr);
    lin(B.wo, att_, ao_, n);
}

// step()'s rows, each of its own sequence: the products for all of them at once, and a row at a time the turn (it
// takes one position and counts on from it), the stored keys and values and the attention, in the row's sequence.
void HyV3::attn_rows(Block& B, int n) {
    const int n_rot = B.rope ? head_dim_ : 0;
    lin(B.wq, xn_, qraw_, n, xq_);
    lin(B.wk, xn_, kraw_, n, xq_);
    lin(B.wv, xn_, v_, n, xq_);
    const float scale = 1.f / std::sqrt((float) head_dim_);
    const size_t kv = (size_t) n_kv_ * head_dim_, hq = (size_t) n_head_ * head_dim_;
    const int window = B.swa ? window_ : 0, ring = B.swa ? ring_ : 0;
    for (int i = 0; i < n; ++i) {
        const Row& r = rows_of_[i];
        const long long* ai = at_ + 2 * i;
        head_norm_rope(qraw_ + i * hq, head_dim_, fp(B.q_norm), q_ + i * hq, head_dim_, n_head_, n_rot, theta_scale_, 0,
                       1, rms_eps_, st_, nullptr, nullptr, 1, ai + 1);
        head_norm_rope(kraw_ + i * kv, head_dim_, fp(B.k_norm), k_ + i * kv, head_dim_, n_kv_, n_rot, theta_scale_, 0,
                       1, rms_eps_, st_, nullptr, nullptr, 1, ai + 1);
        uint16_t* kc = (uint16_t*) seq_buf(r.seq, B.keys);
        uint16_t* vc = (uint16_t*) seq_buf(r.seq, B.vals);
        store_f16_at(k_ + i * kv, kc, (int) kv, ai, 1, ring, st_);
        store_f16_at(v_ + i * kv, vc, (int) kv, ai, 1, ring, st_);
        const int64_t keep = pos_;
        pos_ = r.pos;
        const int sized = B.swa ? std::min(window_, opt_.max_context) : room(1);
        pos_ = keep;
        attend(q_ + i * hq, kc, vc, r.pos, 1, score_, chunk_, att_ + i * hq, head_dim_, n_head_, n_kv_, scale, st_,
               window, ring, false, ai, sized, score_n_, {}, {}, nullptr);
    }
    lin(B.wo, att_, ao_, n);
}

void HyV3::dense_ffn(Block& B, int il, int n) {
    lin(B.ffn_gate, xn_, g_, n, xq_);
    lin(B.ffn_up, xn_, u_, n, xq_);
    swiglu_clamp(g_, u_, h_, n * n_ff_, clamp_shexp_[(size_t) il], st_);
    lin(B.ffn_down, h_, fo_, n);
}

// A layer from its input to its router's copy (a dense layer: to its end).  A prompt batch's MoE (moe_rows) does
// its own routing.  A norm and its quantization are one launch, and so are the residual's sum and the next norm.
void HyV3::layer_head(Layer& L, int il, int n) {
    Block& B = blocks_[(size_t) il];
    const int threads = sum_threads(n_embd_);
    kernels::native_add_rms_norm_q8_1(r_, nullptr, fp(L.attn_norm), xn_, xq_, n_embd_, n, rms_eps_, threads, st_);
    dbg("attn_in", il, xn_, n_embd_);
    attn(B, pos_, n);
    dbg("attn", il, ao_, n_embd_);

    kernels::native_add_rms_norm_q8_1(r_, ao_, fp(L.ffn_norm), xn_, xq_, n_embd_, n, rms_eps_, threads, st_);
    dbg("ffn_in", il, xn_, n_embd_);
    if (!L.moe) {
        dense_ffn(B, il, n);
        dbg("ffn", il, fo_, n_embd_);
        axpy(fo_, 1.f, r_, n * n_embd_, true, st_);
    } else if (!batched(L, n)) {
        moe_router(L, n);
    }
}

void HyV3::layer_tail(Layer&, int, int n) { axpy(fo_, 1.f, r_, n * n_embd_, true, st_); }

void HyV3::output_rows(int first, int rows, float* h) {
    rms_norm(r_ + (size_t) first * n_embd_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

// What ggml's MoE graph does for this model: sigmoid gates, the n_used largest of gate + bias (no bias in the
// files so far), and weights from the bare gates, made to sum to one.
void HyV3::route(int il, const float* logits, int32_t, int* ids, float* w, float* probs) const {
    const std::vector<float>& bias = blocks_[(size_t) il].probs_b;
    float* pick = probs + n_expert_;
    for (int e = 0; e < n_expert_; ++e) {
        probs[e] = 1.f / (1.f + std::exp(-logits[e]));
        pick[e] = probs[e] + bias[(size_t) e];
    }
    float sum = 0.f;
    for (int k = 0; k < n_used_; ++k) {
        int best = 0;
        for (int e = 1; e < n_expert_; ++e)
            if (pick[e] > pick[best]) best = e;
        pick[best] = -INFINITY;
        ids[k] = best;
        w[k] = probs[best];
        sum += w[k];
    }
    const float denom = w_norm_ ? std::max(sum, kMinWeightSum) : 1.f;
    for (int k = 0; k < n_used_; ++k) {
        w[k] /= denom;
        if (w_scale_ != 0.f) w[k] *= w_scale_;
    }
}

bool HyV3::route_card(int il, int n, int32_t* ids, float* w) {
    const Block& B = blocks_[(size_t) il];
    route_rows(rl_, n_expert_, n_used_, n, 1, 1.f, fp(B.probs_b_dev), nullptr, w_norm_, w_scale_, ids, w, st_);
    return true;
}

// The shared expert, ungated: the routed experts' sum starts from its output.
void HyV3::moe_shared(const Layer& L, int il, int n) {
    lin(L.sh_gate, xn_, g_, n, xq_);
    lin(L.sh_up, xn_, u_, n, xq_);
    swiglu_clamp(g_, u_, h_, n * n_ff_shexp_, clamp_shexp_[(size_t) il], st_);
    lin(L.sh_down, h_, fo_, n);
}

}  // namespace strata::poly
