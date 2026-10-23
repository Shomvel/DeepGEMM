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

Wgrad defaults to FP32 and accepts `output_dtype=torch.bfloat16`. Both paths
default to `overwrite=True`: fresh output is not cleared, and the native SM90
epilogue stores every result, including zeros for empty experts. Explicit
`overwrite=False` retains zero-initialized accumulation. The native APIs still
accumulate when C is supplied. BF16 output keeps FP32 register accumulation and
rounds at writeback, removing the large FP32 gradient and subsequent cast.
On SM90, nonempty multi-expert overwrite requires M divisible by 64, and the
selected tile must divide M, so a tail store cannot overwrite the next expert.
C-present accumulation, single-expert tails and all-empty zero-fill retain their
previous contracts. The
FP8 1D1D epilogue supports this output dtype with an unswizzled BF16 shared-memory
tile; its matrix multiplication is unchanged. Caller must supply zero padding
for both wgrad operands, except that gathered operands mask padding themselves.
There is no cross-step weight cache. The grouped GEMM math mainloops are unchanged;
SM90 K-group multicast configurations must keep each physical CTA cluster within
one expert. The optional scatter path adds a
CUDA epilogue to both SM90 BF16 and FP8 1D2D kernels. FP8 training uses approximate
straight-through gradients. No numerical convergence claim follows from a short
smoke training run.

Additional adapter operations are `m_gemm_gather` (FP8 token gather fused with
quantization) and `m_gemm_columns` (reused BF16 column workspace). The latter yields
`(column_offset, tile)`; a consumer must enqueue use of the tile on the same CUDA
stream before requesting the next tile, or clone it for longer retention. FP8
activations are quantized once per projection. Scale tensors are emitted in
native MN-major TMA layout to avoid repeated transposes. Supported packed BF16
transposed weights are passed to the native kernel without copying. K-group
quantization also accepts a token-to-route gather, while still transposing and
scaling each expert group independently.

`wgrad_gather_rhs_columns(lhs, tokens, row_to_route, topk, columns=2048,
output_dtype=torch.float32)` bounds the FP8 RHS gather workspace. It quantizes
lhs once, then gathers feature chunks into reusable FP8 and FP32 scale buffers.
Each native K-group GEMM overwrites the corresponding column slice of the final
`[experts,lhs_features,token_features]` gradient. There is no BF16 chunk buffer or
copy into the output. With R padded routes and C workspace columns, the RHS
scratch is `R*C + 4*(R/128)*C` bytes instead of scaling with the full token width.
`columns` must be a positive multiple of 128; the last chunk can be shorter.
Source row strides and column offsets use 64-bit address arithmetic. Token
columns must be contiguous, but token rows need not be. The final output retains
the native 16-byte TMA row-alignment requirement.

A producer that already emits native K-group FP8 lhs values can call this method
with `lhs=None, quantized_lhs=(q_flat, scales)`. The scale tensor has shape
`[lhs_features,R/128]` and strides `(1,lhs_features)`. This avoids creating a BF16
lhs just to quantize it again; both prequantized tensors remain unchanged. Native
K-group column slices support overwrite only. Accumulation into a strided slice
is rejected, while contiguous outputs retain their existing C-present behavior.

On Ubuntu 24.04, install `libdw-dev libelf-dev` before building the C++ extension
(the pinned DeepJIT revision includes `elfutils/libdwfl.h`). GCC 13+ is required
for C++20 `<format>`.

Run `python -m pytest tests/test_sonic_training.py -q` on Hopper, with PyTorch,
Triton, CUDA 12.9+ and a built DeepGEMM extension. The independent FP32 checks
include strided weights, an empty expert, per-expert transposed FP8 scale layout,
and weighted/unweighted atomic scatter with padding and cancellation.

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

## Clean H100 validation: opt-r15

[Native acceptance data](sonic-hopper-training-validation.json) records the clean-build adapter/scatter checks, 40 multicast cases, 16 overwrite cases, 8 strided-output cases, 10 gathered-column checks and 120 M-tail combinations. Real row/K quantization and companion dH/AP/DS/gather/bias tests cross 2**31 element addresses. No-C multi-group M-tail rejection and C-present/single-group/all-empty compatibility are covered.

The companion Sonic fork passed 42 strict FP32 full-layer cases, all gradients, wrappers, 20-step FP32-master SGD and 55 whole-layer timing records. BF16 maximum relative L2 error was 0.4630%, FP8 6.6957%; the optional dH-recompute path was bitwise identical to default FP8 on its 9 full-gradient cases and final short-training outputs/parameters. These checks do not establish long-run convergence.

For T32768/D4096/I1536/E64/K8, optional recompute measured 47.414 ms / 5858.26 MiB incremental peak versus original Sonic BF16 51.531 ms / 6404.13 MiB. FP8 is approximate training, not an equal-precision substitution. Other shapes have time/memory tradeoffs. See the companion `assets/deepgemm-hopper-training.md` and optimized results/provenance JSON for full tables and exact scope.

Only H100 was executed, using CUDA 13.0 and NVIDIA Torch 2.10.0a0+b558c986e8.nv25.11, QuACK 0.6.4 and CUTLASS DSL 4.6.2. The companion package declares Torch >=2.11; the source-import prerelease test is not general 2.10 support. Source/install/native hashes matched; the job completed with exit 0 and was deleted. H800/H200 remain untested.
