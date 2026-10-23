"""SM90 training adapters: padded M groups and independently packed K groups.

These helpers deliberately leave routing and activation policy to the caller. FP8
uses E4M3 with FP32 scales, 1x128 activations and 128x128 weights for M-grouped
GEMMs; K-grouped GEMMs use 1x128 scales on each *transposed expert group*.
All conversions are part of the call. There is no stale cross-step weight cache.
"""
from dataclasses import dataclass

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _expert_labels(OFFSETS, INDICES, B: tl.constexpr):
    e = tl.program_id(0)
    start, end = tl.load(OFFSETS + e), tl.load(OFFSETS + e + 1)
    row = start + tl.program_id(1) * B + tl.arange(0, B)
    tl.store(INDICES + row, e, row < end)


@dataclass(frozen=True)
class GroupedTrainingLayout:
    ks: tuple[int, ...]
    offsets: torch.Tensor
    m_indices: torch.Tensor
    k_sizes: torch.Tensor

    @classmethod
    def from_counts(cls, counts: list[int], device: torch.device):
        from . import get_mk_alignment_for_contiguous_layout
        alignment = get_mk_alignment_for_contiguous_layout()
        if alignment % 128:
            raise ValueError("SM90 training requires group alignment divisible by 128")
        if not counts or any(k < 0 for k in counts):
            raise ValueError("counts must be a nonempty list of nonnegative integers")
        ks = tuple(tr.cdiv(k, alignment) * alignment for k in counts)
        ends = [0]
        for k in ks:
            ends.append(ends[-1] + k)
        metadata = torch.tensor([*ks, *ends], dtype=torch.int32, device=device)
        sizes, offsets = metadata[:len(ks)], metadata[len(ks):]
        indices = torch.empty((ends[-1],), dtype=torch.int32, device=device)
        if ends[-1]:
            _expert_labels[(len(ks), tr.cdiv(max(ks), 1024))](offsets, indices, 1024)
        return cls(ks, offsets, indices, sizes)

    @property
    def rows(self):
        return sum(self.ks)


@tr.jit
def _row_quant(X, Q, S, MAP, M: tl.constexpr, N: tl.constexpr, B: tl.constexpr, R: tl.constexpr,
               TOPK: tl.constexpr, GATHER: tl.constexpr, SS: tl.constexpr):
    # Padded top-16 routes can exceed 2**31 elements even at 32K tokens.
    r = (tl.program_id(0) * R + tl.arange(0, R)).to(tl.int64)
    b = tl.program_id(1)
    c = b * B + tl.arange(0, B)
    mask = (r[:, None] < M) & (c[None, :] < N)
    source = r
    valid = r < M
    if GATHER:
        route = tl.load(MAP + r, r < M, -1)
        source = route // TOPK
        valid = valid & (route >= 0)
    v = tl.load(X + source[:, None] * N + c[None, :],
                valid[:, None] & (c[None, :] < N), 0).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(v), 1), 1.e-4) / 448.
    tl.store(Q + r[:, None] * N + c[None, :], (v / s[:, None]).to(Q.dtype.element_ty), mask)
    tl.store(S + r + b * SS, s, r < M)


@tr.jit
def _weight_quant(W, Q, S, N: tl.constexpr, K: tl.constexpr,
                  SE: tl.constexpr, SN: tl.constexpr, SK: tl.constexpr):
    e, bn, bk = tl.program_id(0).to(tl.int64), tl.program_id(1), tl.program_id(2)
    n = (bn * 128 + tl.arange(0, 128)).to(tl.int64)
    k = (bk * 128 + tl.arange(0, 128)).to(tl.int64)
    v = tl.load(W + e * SE + n[:, None] * SN + k[None, :] * SK,
                (n[:, None] < N) & (k[None, :] < K), 0).to(tl.float32)
    s = tl.maximum(tl.max(tl.max(tl.abs(v), 0), 0), 1.e-4) / 448.
    tl.store(Q + e * N * K + n[:, None] * K + k[None, :],
             (v / s).to(Q.dtype.element_ty), (n[:, None] < N) & (k[None, :] < K))
    tl.store(S + (e * tl.cdiv(N, 128) + bn) * tl.cdiv(K, 128) + bk, s)


@tr.jit
def _k_quant(X, Q, S, OFFSETS, MAP, ROWS: tl.constexpr, N: tl.constexpr,
             SOURCE_STRIDE: tl.constexpr, COLUMN_START: tl.constexpr,
             BN: tl.constexpr, TOPK: tl.constexpr, GATHER: tl.constexpr):
    # A block of 128 token rows belongs to exactly one padded expert group.
    e, block, nc = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start = tl.load(OFFSETS + e).to(tl.int64)
    end = tl.load(OFFSETS + e + 1).to(tl.int64)
    r = (block * 128 + tl.arange(0, 128)).to(tl.int64)
    n = (nc * BN + tl.arange(0, BN)).to(tl.int64)
    mask = (start + r[:, None] < end) & (n[None, :] < N)
    source = start + r
    valid = source < end
    if GATHER:
        route = tl.load(MAP + source, valid, -1).to(tl.int64)
        source = route // TOPK
        valid = valid & (route >= 0)
    v = tl.load(X + source[:, None] * SOURCE_STRIDE + COLUMN_START + n[None, :],
                valid[:, None] & (n[None, :] < N), 0).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(v), axis=0), 1.e-4) / 448.
    # Physical NT format: each group's [K,N] becomes flat contiguous [N,K].
    dest = start * N + n[None, :] * (end - start) + r[:, None]
    tl.store(Q + dest, (v / s[None, :]).to(Q.dtype.element_ty), mask)
    tl.store(S + n + (start // 128 + block) * N, s,
             (start + block * 128 < end) & (n < N))


def quantize_rows(x, row_to_route=None, topk=1):
    rows = x.size(0) if row_to_route is None else row_to_route.numel()
    q = torch.empty((rows, x.size(1)), device=x.device, dtype=torch.float8_e4m3fn)
    # Produce the native MN-major scale layout, avoiding a transpose at each GEMM.
    sf = torch.empty_strided((rows, tr.cdiv(x.size(1), 128)), (1, tr.cdiv(rows, 4) * 4),
                             device=x.device, dtype=torch.float32)
    if rows:
        _row_quant[(tr.cdiv(rows, 16), tr.cdiv(x.size(1), 128))](
            x, q, sf, row_to_route, rows, x.size(1), 128, 16, topk, row_to_route is not None, sf.stride(1), num_warps=4)
    return q, sf


def quantize_weights(w):
    e, n, k = w.shape
    q = torch.empty(w.shape, device=w.device, dtype=torch.float8_e4m3fn)
    sf = torch.empty((e, tr.cdiv(n, 128), tr.cdiv(k, 128)), device=w.device, dtype=torch.float32)
    _weight_quant[(e, tr.cdiv(n, 128), tr.cdiv(k, 128))](w, q, sf, n, k, *w.stride())
    return q, sf


def quantize_k_groups(x, layout, row_to_route=None, topk=1):
    if x.ndim != 2 or x.stride(1) != 1:
        raise ValueError("K-group quantization expects a matrix with contiguous columns")
    q = torch.empty((layout.rows * x.size(1),), device=x.device, dtype=torch.float8_e4m3fn)
    sf = torch.empty_strided((x.size(1), layout.rows // 128), (1, x.size(1)),
                             device=x.device, dtype=torch.float32)
    if layout.rows:
        _k_quant[(len(layout.ks), max(layout.ks) // 128, tr.cdiv(x.size(1), 64))](
            x, q, sf, layout.offsets, row_to_route, layout.rows, x.size(1), x.stride(0), 0, 64,
            topk, row_to_route is not None, num_warps=4)
    return q, sf


def _bf16_weights(w):
    # Native SM90 accepts packed K-major or MN-major expert matrices. Preserve
    # transposed parameters without copying; other strided views still pack.
    packed_batch = w.stride(0) == w.size(1) * w.size(2)
    if packed_batch and (w.is_contiguous() or w.transpose(1, 2).is_contiguous()):
        return w
    # contiguous() can preserve a noncanonical batch stride when E=1.
    return w.clone(memory_format=torch.contiguous_format)


class GroupedTrainingGemm:
    """All six training GEMMs on SM90. BF16 dgrad; FP32 or BF16 wgrad.

    ``precision='fp8'`` is approximate training with straight-through gradients;
    it does not differentiate the quantizer or claim BF16 numerical equivalence.
    Caller must zero padding in BOTH wgrad operands.
    """
    def __init__(self, layout, precision="bf16"):
        if precision not in ("bf16", "fp8"):
            raise ValueError(f"unsupported training precision: {precision}")
        self.layout, self.precision = layout, precision

    def m_gemm(self, a, w):
        from . import m_grouped_bf16_gemm_nt_contiguous, m_grouped_fp8_gemm_nt_contiguous
        if a.ndim != 2 or w.ndim != 3 or a.size(0) != self.layout.rows or a.size(1) != w.size(2):
            raise ValueError("M GEMM expects [sum padded tokens,K] and [experts,N,K]")
        out = torch.empty((a.size(0), w.size(1)), dtype=torch.bfloat16, device=a.device)
        if self.precision == "bf16":
            m_grouped_bf16_gemm_nt_contiguous(a.contiguous(), _bf16_weights(w), out, self.layout.m_indices)
        else:
            m_grouped_fp8_gemm_nt_contiguous(
                quantize_rows(a.contiguous()), quantize_weights(w), out,
                self.layout.m_indices, recipe=(1, 128, 128), disable_ue8m0_cast=True)
        return out

    def m_gemm_gather(self, tokens, w, row_to_route, topk):
        """FP8 gather+quantize directly from tokens, without a BF16 route buffer."""
        from . import m_grouped_fp8_gemm_nt_contiguous
        if self.precision != "fp8":
            raise ValueError("fused quantized gather requires fp8 precision")
        out = torch.empty((self.layout.rows, w.size(1)), device=tokens.device, dtype=torch.bfloat16)
        m_grouped_fp8_gemm_nt_contiguous(
            quantize_rows(tokens, row_to_route, topk), quantize_weights(w), out,
            self.layout.m_indices, recipe=(1, 128, 128), disable_ue8m0_cast=True)
        return out

    def m_gemm_columns(self, a, w, columns):
        """Yield (column offset, BF16 tile) using one reusable output workspace.

        Consumers must enqueue each tile's use on the same CUDA stream before
        requesting the next tile.
        FP8 activations are quantized once per projection, not once per tile.
        """
        from . import m_grouped_bf16_gemm_nt_contiguous, m_grouped_fp8_gemm_nt_contiguous
        if columns <= 0 or columns % 128:
            raise ValueError("column workspace must be a positive multiple of 128")
        if a.ndim != 2 or w.ndim != 3 or a.size(0) != self.layout.rows or a.size(1) != w.size(2):
            raise ValueError("column GEMM expects grouped [rows,K] and [experts,N,K]")
        columns = min(columns, w.size(1))
        workspace = torch.empty((a.size(0), columns), device=a.device, dtype=torch.bfloat16)
        qa = a.contiguous() if self.precision == "bf16" else quantize_rows(a.contiguous())
        for start in range(0, w.size(1), columns):
            weights = w[:, start:start + columns]
            tile = workspace[:, :weights.size(1)]
            if self.precision == "bf16":
                m_grouped_bf16_gemm_nt_contiguous(qa, _bf16_weights(weights), tile, self.layout.m_indices)
            else:
                m_grouped_fp8_gemm_nt_contiguous(
                    qa, quantize_weights(weights), tile, self.layout.m_indices,
                    recipe=(1, 128, 128), disable_ue8m0_cast=True)
            yield start, tile

    def m_gemm_scatter(self, a, w, row_to_route, num_tokens, topk, scores=None, bias=None):
        """Accumulate FP32 WGMMA results directly into tokens, then cast to BF16.

        scores=None is the backward dX path; dH already contains routing weights.
        The atomic reduction is nondeterministic. No [padded routes,N] output exists.
        """
        from . import m_grouped_bf16_gemm_nt_scatter, m_grouped_fp8_gemm_nt_scatter
        if a.ndim != 2 or w.ndim != 3 or a.size(0) != self.layout.rows or a.size(1) != w.size(2):
            raise ValueError("scatter GEMM expects grouped [rows,K] and [experts,N,K]")
        out = torch.zeros((num_tokens, w.size(1)), dtype=torch.float32, device=a.device)
        if self.precision == "bf16":
            operands = a.contiguous(), w.contiguous()
            fn = m_grouped_bf16_gemm_nt_scatter
        else:
            operands = quantize_rows(a.contiguous()), quantize_weights(w)
            fn = m_grouped_fp8_gemm_nt_scatter
        fn(*operands, out, self.layout.m_indices, row_to_route, topk,
           None if scores is None else scores.float().contiguous(), bias)
        return out.to(torch.bfloat16)

    def wgrad(self, lhs, rhs, *, row_to_route=None, topk=1, gather_lhs=False, gather_rhs=False,
              output_dtype=torch.float32, overwrite=True):
        """Weight gradients, overwriting fresh output without first zeroing it.

        ``overwrite=False`` retains the zero-initialized accumulation path.
        """
        from . import k_grouped_bf16_gemm_tn_contiguous, k_grouped_fp8_gemm_nt_contiguous
        if (not gather_lhs and lhs.size(0) != self.layout.rows) or (not gather_rhs and rhs.size(0) != self.layout.rows):
            raise ValueError("wgrad operands must share the padded token layout")
        if (gather_lhs or gather_rhs) and (self.precision != "fp8" or row_to_route is None):
            raise ValueError("fused wgrad gather requires fp8 and a route map")
        if output_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("wgrad output dtype must be FP32 or BF16")
        allocate = torch.empty if overwrite else torch.zeros
        out = allocate((len(self.layout.ks), lhs.size(1), rhs.size(1)),
                       device=lhs.device, dtype=output_dtype)
        initial = None if overwrite else out
        if self.precision == "bf16":
            k_grouped_bf16_gemm_tn_contiguous(
                lhs.contiguous(), rhs.contiguous(), out, list(self.layout.ks), self.layout.k_sizes, initial)
        else:
            k_grouped_fp8_gemm_nt_contiguous(
                quantize_k_groups(lhs.contiguous(), self.layout, row_to_route if gather_lhs else None, topk),
                quantize_k_groups(rhs.contiguous(), self.layout, row_to_route if gather_rhs else None, topk),
                out, list(self.layout.ks), self.layout.k_sizes, initial, recipe=(1, 1, 128))
        return out

    def wgrad_gather_rhs_columns(self, lhs, tokens, row_to_route, topk, *, columns=2048,
                                  output_dtype=torch.float32, quantized_lhs=None):
        """FP8 wgrad with bounded gather workspace and direct final-gradient stores.

        Quantize lhs once, then gather/quantize token feature chunks into reusable
        FP8 and scale storage. Each GEMM overwrites its slice of the final output;
        no BF16 chunk or copy is needed. Padding is masked by ``row_to_route``.

        A producer of native K-group FP8 values may pass ``lhs=None`` and
        ``quantized_lhs=(q_flat, scales)`` to avoid materializing a BF16 lhs.
        Its scale shape is ``[lhs_features, layout.rows // 128]``.
        """
        from . import k_grouped_fp8_gemm_nt_contiguous
        layout = self.layout
        rows = layout.rows
        if self.precision != "fp8":
            raise ValueError("column-gather wgrad requires fp8 precision")
        if columns <= 0 or columns % 128:
            raise ValueError("column workspace must be a positive multiple of 128")
        if output_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("wgrad output dtype must be FP32 or BF16")
        if tokens.ndim != 2 or tokens.size(1) == 0 or tokens.stride(1) != 1:
            raise ValueError("tokens must be a matrix with nonempty contiguous columns")
        if (row_to_route.ndim != 1 or row_to_route.numel() != rows
                or row_to_route.dtype != torch.int64 or not row_to_route.is_contiguous()
                or row_to_route.device != tokens.device or layout.offsets.device != tokens.device
                or topk <= 0):
            raise ValueError("column-gather wgrad requires a matching int64 route map and positive topk")
        if quantized_lhs is None:
            if (lhs is None or lhs.ndim != 2 or lhs.size(0) != rows
                    or lhs.size(1) == 0 or lhs.device != tokens.device):
                raise ValueError("lhs must share the padded token layout and device")
            qa = quantize_k_groups(lhs.contiguous(), layout)
        else:
            if lhs is not None:
                raise ValueError("pass lhs=None when supplying quantized_lhs")
            qa = quantized_lhs
        q_lhs, s_lhs = qa
        if (q_lhs.ndim != 1 or q_lhs.dtype != torch.float8_e4m3fn or not q_lhs.is_contiguous()
                or s_lhs.ndim != 2 or s_lhs.dtype != torch.float32 or s_lhs.size(0) == 0
                or s_lhs.size(1) != rows // 128
                or q_lhs.numel() != rows * s_lhs.size(0)
                or s_lhs.stride() != (1, s_lhs.size(0))
                or q_lhs.device != tokens.device or s_lhs.device != tokens.device):
            raise ValueError("quantized lhs must use native flat K groups and MN-major FP32 scales")
        out = torch.empty((len(layout.ks), s_lhs.size(0), tokens.size(1)),
                          device=tokens.device, dtype=output_dtype)
        if not rows:
            return out.zero_()
        columns = min(columns, tokens.size(1))
        q = torch.empty((rows * columns,), device=tokens.device, dtype=torch.float8_e4m3fn)
        sf = torch.empty((rows // 128 * columns,), device=tokens.device, dtype=torch.float32)
        ks = list(layout.ks)
        for start in range(0, tokens.size(1), columns):
            width = min(columns, tokens.size(1) - start)
            qb = q[:rows * width]
            scales = sf[:rows // 128 * width].view(-1, width).T
            _k_quant[(len(ks), max(ks) // 128, tr.cdiv(width, 64))](
                tokens, qb, scales, layout.offsets, row_to_route, rows, width,
                tokens.stride(0), start, 64, topk, True, num_warps=4)
            k_grouped_fp8_gemm_nt_contiguous(
                qa, (qb, scales), out[:, :, start:start+width], ks,
                layout.k_sizes, None, recipe=(1, 1, 128))
        return out
