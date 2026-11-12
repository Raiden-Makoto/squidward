// GLM-5.x routing from bf16 router logits in one launch on gfx950, bit-identical to aiter biased_grouped_topk
// (one group, renormalized, 288 experts, top-8) followed by fused_append_shared_experts (slot 8 = expert 288).
// One wave per token, scores in registers. Lane t holds experts 4t..4t+3 and, for t < 8, 256+4t..256+4t+3, in
// aiter's per-thread scan order; each round takes the first strict maximum in that order per lane and combines
// lanes with aiter's wave_reduce DPP sequence (the receiving lane keeps its value on a tie), read from lane 63.
// Rows m >= *n_valid get routed ids 0 and zero weights, as the append kernel's padding path does.
#include <hip/hip_runtime.h>
#include <stdint.h>

// n_valid points at an int32 or int64 count; the low 32-bit word is read (little-endian, count < 2^31).
struct __attribute__((packed)) TArgs {
    const uint16_t* logits; const uint16_t* bias; int32_t* ids; float* wts; const int32_t* n_valid;
    int32_t tok, stride; float rsf, shared_w;
};
static_assert(sizeof(TArgs) == 56, "kernarg ABI drift: keep in sync with TArgs in the Python wrapper");

static constexpr int E = 288, TOPK = 8, SLOTS = TOPK + 1, NV = 8;

__device__ __forceinline__ float bf16_to_f32(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }

template <int CTRL>
__device__ __forceinline__ void reduce_step(float& v, int& k) {
    const float rv = __int_as_float(__builtin_amdgcn_update_dpp(__float_as_int(v), __float_as_int(v), CTRL, 0xf, 0xf, false));
    const int rk = __builtin_amdgcn_update_dpp(k, k, CTRL, 0xf, 0xf, false);
    if (rv > v) v = rv, k = rk;
}

template <int NW>
__device__ __forceinline__ void glm_topk_body(const TArgs& a) {
    const int lane = threadIdx.x & 63, m = blockIdx.x * NW + (threadIdx.x >> 6);
    if (m >= a.tok) return;
    const bool two = lane < 8;
    const uint16_t* lp = a.logits + (size_t)m * a.stride;

    float sc[NV], sg[NV];
#pragma unroll
    for (int j = 0; j < NV; ++j) {
        if (j >= 4 && !two) {
            sc[j] = -INFINITY, sg[j] = 0.f;
            continue;
        }
        const int e = j < 4 ? lane * 4 + j : 256 + lane * 4 + (j - 4);
        const float g = bf16_to_f32(lp[e]);
        const float s = __builtin_amdgcn_rcpf(1.0f + exp2f(-1.44269504088896340736 * g));
        sg[j] = s;
        sc[j] = s + bf16_to_f32(a.bias[e]);
    }

    float sum = 0.f, my_v = 0.f;
    int my_i = 0;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
        float v = -INFINITY;
        int idx = k;
#pragma unroll
        for (int j = 0; j < NV; ++j)
            if (sc[j] > v) v = sc[j], idx = j < 4 ? lane * 4 + j : 256 + lane * 4 + (j - 4);
        reduce_step<0xb1>(v, idx);
        reduce_step<0x4e>(v, idx);
        reduce_step<0x141>(v, idx);
        reduce_step<0x140>(v, idx);
        reduce_step<0x142>(v, idx);
        reduce_step<0x143>(v, idx);
        idx = __builtin_amdgcn_readlane(idx, 63);
        // the owner lane retires the winner and broadcasts its pre-bias sigmoid
        const int owner = idx < 256 ? idx >> 2 : (idx - 256) >> 2, slot = idx < 256 ? (idx & 3) : 4 + (idx & 3);
        float mine = 0.f;
#pragma unroll
        for (int j = 0; j < NV; ++j)
            if (lane == owner && j == slot) mine = sg[j], sc[j] = -INFINITY;
        const float s = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(mine), owner));
        sum += s;
        if (lane == k) my_v = s, my_i = idx;
    }
    const float f = sum > 0.f ? a.rsf / sum : 0.f;
    const bool pad = a.n_valid && m >= *a.n_valid;
    if (lane < TOPK) {
        a.ids[m * SLOTS + lane] = pad ? 0 : my_i;
        a.wts[m * SLOTS + lane] = pad ? 0.f : my_v * f;
    } else if (lane == TOPK) {
        a.ids[m * SLOTS + TOPK] = E;
        a.wts[m * SLOTS + TOPK] = pad ? 0.f : a.shared_w;
    }
}

extern "C" __global__ void __launch_bounds__(256) glm_topk_w4(TArgs a) { glm_topk_body<4>(a); }
extern "C" __global__ void __launch_bounds__(128) glm_topk_w2(TArgs a) { glm_topk_body<2>(a); }
extern "C" __global__ void __launch_bounds__(64) glm_topk_w1(TArgs a) { glm_topk_body<1>(a); }
