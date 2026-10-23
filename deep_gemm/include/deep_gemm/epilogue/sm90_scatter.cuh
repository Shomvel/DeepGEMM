#pragma once

#include <cuda_bf16.h>
#include <cstdint>

namespace deep_gemm {

// WGMMA m64 accumulator mapping, also shared by the scaled FP8 accumulator.
// Each route contributes once per output element. Negative routes are padding.
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t WAVE_BLOCK_M, uint32_t NUM_ACCUM>
__device__ __forceinline__ void sm90_scatter_accum(
    const float* accum, uint32_t m_block, uint32_t n_block,
    uint32_t shape_m, uint32_t shape_n, uint32_t warp, uint32_t lane,
    const int* experts, float* output, const int64_t* row_to_route,
    const float* scores, const __nv_bfloat16* bias, int64_t topk, int64_t num_tokens) {
    #pragma unroll
    for (uint32_t wave = 0; wave < BLOCK_M / WAVE_BLOCK_M; ++wave) {
        #pragma unroll
        for (uint32_t half = 0; half < 2; ++half) {
            const auto row = m_block * BLOCK_M + wave * WAVE_BLOCK_M + warp * 16 + lane / 4 + half * 8;
            if (row >= shape_m)
                continue;
            const int64_t route = row_to_route[row];
            if (route < 0 or route / topk >= num_tokens)
                continue;
            const float score = scores ? scores[route] : 1.0f;
            const int64_t output_base = (route / topk) * shape_n;
            const int64_t bias_base = bias ? static_cast<int64_t>(experts[row]) * shape_n : 0;
            #pragma unroll
            for (uint32_t i = 0; i < NUM_ACCUM / 4; ++i) {
                #pragma unroll
                for (uint32_t pair = 0; pair < 2; ++pair) {
                    const auto col = n_block * BLOCK_N + (lane % 4) * 2 + i * 8 + pair;
                    if (col < shape_n) {
                        float value = accum[wave * NUM_ACCUM + i * 4 + half * 2 + pair];
                        if (bias)
                            value += __bfloat162float(bias[bias_base + col]);
                        atomicAdd(output + output_base + col, value * score);
                    }
                }
            }
        }
    }
}

} // namespace deep_gemm
