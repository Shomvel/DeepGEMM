# SM90 grouped training adapter

`deep_gemm.sonic_training.GroupedTrainingLayout.from_counts(counts, device)` pads
individual expert groups using the runtime alignment. Zero-token experts keep K=0.
It exposes distinct M-group labels and K-group sizes, since these API arguments
have different meanings. `GroupedTrainingGemm(layout, precision)` implements
`m_gemm([sumK,K], [E,N,K])` and `wgrad([sumK,M], [sumK,N])`.

BF16 uses the native SM90 M-grouped NT and K-grouped TN kernels. FP8 uses M-grouped
NT (1x128 activations, 128x128 weights) and K-grouped NT (1x128 for each operand).
The latter physically transposes **each expert separately**, and requantizes along
the token reduction axis. Its FP32 scale tensor is `[features,sumK/128]`. Neither
transposing token-major FP8 bytes nor reusing forward scales implements this format.

All wgrad output is FP32, zero-initialized before the required SM90 accumulation.
Every conversion is included in the adapter call. Caller must supply zero padding
for both wgrad operands. There is no activation epilogue fusion or weight cache.
This is an experimental Python/Triton adapter around existing DeepGEMM CUDA
kernels, not a change to the SM90 GEMM mainloop. FP8 training uses approximate
straight-through gradients. No numerical convergence claim follows from a short
smoke training run.

On Ubuntu 24.04, install `libdw-dev libelf-dev` before building the C++ extension
(the pinned DeepJIT revision includes `elfutils/libdwfl.h`). GCC 13+ is required
for C++20 `<format>`.

Run `python -m pytest tests/test_sonic_training.py -q` on Hopper, with PyTorch,
Triton, CUDA 12.9+ and a built DeepGEMM extension. The independent FP32 checks
include strided weights, an empty expert, and per-expert transposed FP8 scale layout.

## H100 validation, 2026-10-23

The companion [SonicMoE branch](https://github.com/Shomvel/sonic-moe/tree/deepgemm-hopper-training)
uses this adapter for all six expert-layer training GEMMs. On one H100 80GB,
adapter M-GEMM relative L2 errors against FP32 were 0.0016583 (BF16) and 0.0369523
(FP8). Weight-gradient errors were 2.07e-7 and 0.0272196 respectively. Empty-expert
weight gradients were exactly zero. Its full MoE numerical, wrapper and 20-step
SGD checks also passed.

For T/D/I/E/K=4096/4096/2048/8/2, synchronized expert forward+backward took
2.386 ms with original Sonic BF16, 5.855 ms with this BF16 adapter and 4.818 ms
with FP8. Packing and quantization are included. The adapter has no measured
end-to-end speed or memory advantage in these tests. Full results and limits are
in the companion branch's `assets/deepgemm-hopper-training.md` and
`assets/deepgemm-h100-results.json`.

The executed environment was CUDA 13.0, NVIDIA PyTorch
`2.10.0a0+b558c986e8.nv25.11`, Python 3.12.3, GCC 13.3 and Triton
`3.5.0+gitde3506d2`. This test leaves the SM90 CUDA mainloop unchanged. H800/H200
and long-run training convergence have not been executed.
