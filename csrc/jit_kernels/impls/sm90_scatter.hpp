#pragma once

#include <cstdint>

namespace deep_gemm {

// Separate from the logical BF16 D descriptor: scatter never allocates D[M,N].
struct SM90ScatterArgs {
    void* output = nullptr;
    void* row_to_route = nullptr;
    void* scores = nullptr;
    void* bias = nullptr;
    int64_t topk = 0;
    int64_t num_tokens = 0;
};

} // namespace deep_gemm
