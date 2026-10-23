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
The grouped GEMM mainloops are unchanged. The optional scatter path adds a
CUDA epilogue to both SM90 BF16 and FP8 1D2D kernels. FP8 training uses approximate
straight-through gradients. No numerical convergence claim follows from a short
smoke training run.

On Ubuntu 24.04, install `libdw-dev libelf-dev` before building the C++ extension
(the pinned DeepJIT revision includes `elfutils/libdwfl.h`). GCC 13+ is required
for C++20 `<format>`.

Run `python -m pytest tests/test_sonic_training.py -q` on Hopper, with PyTorch,
Triton, CUDA 12.9+ and a built DeepGEMM extension. The independent FP32 checks
include strided weights, an empty expert, per-expert transposed FP8 scale layout,
and weighted/unweighted atomic scatter with padding and cancellation.

## Historical materialized validation, 2026-10-23

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

## FP32 token scatter epilogue

`GroupedTrainingGemm.m_gemm_scatter(a, w, row_to_route, num_tokens, topk,
scores=None, bias=None)` uses WGMMA FP32 accumulators directly. For each valid
padded row it adds optional BF16 expert bias, multiplies an optional FP32 route
score, and atomically adds into `[num_tokens,N]` FP32 storage. It returns a BF16
cast after the GEMM. There is no `[padded routes,N]` global output allocation.
The existing shared-memory/mainloop configuration is retained for this first
version; unused output shared memory is not yet reclaimed.

The low-level APIs `m_grouped_bf16_gemm_nt_scatter` and
`m_grouped_fp8_gemm_nt_scatter` **accumulate** into caller-initialized FP32 storage;
they neither zero it nor cast the output. Both use contiguous K-major operands,
SM90, N/K divisible by 128, aligned single-expert M tiles and int32 expert labels.
The FP8 API takes E4M3 pairs with FP32 scales and a fixed 1x128 / 128x128 recipe.
`row_to_route` is contiguous int64 with one entry per padded row, `-1` for padding;
valid entries are flattened token/top-k indices. Scores, when present, contain
`num_tokens*topk` FP32 values; bias is BF16 `[E,N]`. Input metadata must describe
the same padded groups as the operands, following the existing grouped API contract.
These APIs reject DeepGEMM and PyTorch deterministic modes because atomic order
can change between launches. This is an experimental inference/first-gradient
primitive; the companion Sonic custom autograd supplies its training backward.

Forward down-projection supplies both scores and bias. Backward input-gradient
projection supplies neither: its dH operand already contains routing weights.
This removes two D-wide route-expanded outputs while preserving the unweighted
router derivative and the six GEMM training structure. Raw FP32 scatter also
removes the previous per-route BF16 output rounding, so numerical results are not
bitwise equal to the materialized path.

## Scatter H100 validation

Both new CUDA epilogues passed weighted/unweighted, bias, cancellation, nonzero
padding, empty-expert and invalid-metadata checks. FP32 scatter relative L2 error
against independent FP32 GEMM/reduction was at most 1.29e-7 for BF16 and 3.67%
for FP8. Ordinary M-GEMM and K-grouped weight-gradient regressions also passed.
The companion Sonic layer passed output/all-gradient, wrapper and 20-step SGD
checks with scatter enabled, including top-k=8 signed scores and empty batches.

For the pre-grouped down-projection at T/D/I/E/K=32768/4096/256/8/8, BF16 peak
incremental allocation falls from 2304 to 768 MiB, but latency increases from
2.029 to 7.375 ms. FP8 falls from 2304 to 842 MiB and increases from 2.178 to
7.656 ms. These CUDA-event timings include zeroing/casting and FP8 conversions.
All tokens select the 8 experts in expert order. This validates removal of the
2 GiB route-expanded output; it does not establish a speedup or full-layer peak
reduction of the same size.

For whole-layer T/D/I/E/K=4096/4096/512/8/8, scatter reduces BF16 incremental peak
from 1248.75 to 992.75 MiB while increasing forward+backward from 4.457 to
5.620 ms. FP8 drops from 1290.82 to 1066.82 MiB, increasing 5.686 to 6.946 ms.
Original Sonic is 2.350 ms and 480.52 MiB. The first atomic implementation is
therefore a memory/time tradeoff, not a competitive throughput backend.

The companion branch's `assets/deepgemm-hopper-training.md` includes all four
shapes and measurement definitions. Raw results and provenance are in
`assets/deepgemm-h100-scatter-{results,adapters,provenance}.json`. The clean build
completed with exit 0 on H100; source/install hashes matched and the job was
removed. H800/H200 and long-run convergence remain untested. The build now tracks
C++ API headers as extension dependencies to prevent stale native objects.
