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
        sizes = torch.tensor(ks, dtype=torch.int32, device=device)
        indices = torch.repeat_interleave(
            torch.arange(len(ks), dtype=torch.int32, device=device), sizes,
            output_size=ends[-1],
        )
        return cls(ks, torch.tensor(ends, dtype=torch.int32, device=device), indices, sizes)

    @property
    def rows(self):
        return sum(self.ks)


@tr.jit
def _row_quant(X, Q, S, N: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0)
    b = tl.program_id(1)
    c = b * B + tl.arange(0, B)
    v = tl.load(X + r * N + c, c < N, 0).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(v), 0), 1.e-4) / 448.
    tl.store(Q + r * N + c, (v / s).to(Q.dtype.element_ty), c < N)
    tl.store(S + r * tl.cdiv(N, B) + b, s)


@tr.jit
def _weight_quant(W, Q, S, N: tl.constexpr, K: tl.constexpr,
                  SE: tl.constexpr, SN: tl.constexpr, SK: tl.constexpr):
    e, bn, bk = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n = bn * 128 + tl.arange(0, 128)
    k = bk * 128 + tl.arange(0, 128)
    v = tl.load(W + e * SE + n[:, None] * SN + k[None, :] * SK,
                (n[:, None] < N) & (k[None, :] < K), 0).to(tl.float32)
    s = tl.maximum(tl.max(tl.max(tl.abs(v), 0), 0), 1.e-4) / 448.
    tl.store(Q + e * N * K + n[:, None] * K + k[None, :],
             (v / s).to(Q.dtype.element_ty), (n[:, None] < N) & (k[None, :] < K))
    tl.store(S + (e * tl.cdiv(N, 128) + bn) * tl.cdiv(K, 128) + bk, s)


@tr.jit
def _k_quant(X, Q, S, OFFSETS, ROWS: tl.constexpr, N: tl.constexpr,
             BN: tl.constexpr):
    # A block of 128 token rows belongs to exactly one padded expert group.
    e, block, nc = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start, end = tl.load(OFFSETS + e), tl.load(OFFSETS + e + 1)
    r = block * 128 + tl.arange(0, 128)
    n = nc * BN + tl.arange(0, BN)
    mask = (start + r[:, None] < end) & (n[None, :] < N)
    v = tl.load(X + (start + r[:, None]) * N + n[None, :], mask, 0).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(v), axis=0), 1.e-4) / 448.
    # Physical NT format: each group's [K,N] becomes flat contiguous [N,K].
    dest = start * N + n[None, :] * (end - start) + r[:, None]
    tl.store(Q + dest, (v / s[None, :]).to(Q.dtype.element_ty), mask)
    tl.store(S + n * (ROWS // 128) + start // 128 + block, s,
             (start + block * 128 < end) & (n < N))


def quantize_rows(x):
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    sf = torch.empty((x.size(0), tr.cdiv(x.size(1), 128)), device=x.device, dtype=torch.float32)
    if x.numel():
        _row_quant[(x.size(0), tr.cdiv(x.size(1), 128))](x, q, sf, x.size(1), 128)
    return q, sf


def quantize_weights(w):
    e, n, k = w.shape
    q = torch.empty(w.shape, device=w.device, dtype=torch.float8_e4m3fn)
    sf = torch.empty((e, tr.cdiv(n, 128), tr.cdiv(k, 128)), device=w.device, dtype=torch.float32)
    _weight_quant[(e, tr.cdiv(n, 128), tr.cdiv(k, 128))](w, q, sf, n, k, *w.stride())
    return q, sf


def quantize_k_groups(x, layout):
    q = torch.empty((x.numel(),), device=x.device, dtype=torch.float8_e4m3fn)
    sf = torch.empty((x.size(1), x.size(0) // 128), device=x.device, dtype=torch.float32)
    if x.numel():
        _k_quant[(len(layout.ks), max(layout.ks) // 128, tr.cdiv(x.size(1), 32))](
            x, q, sf, layout.offsets, layout.rows, x.size(1), 32, num_warps=8)
    return q, sf


class GroupedTrainingGemm:
    """All six training GEMMs on SM90. Outputs: BF16 dgrad, FP32 wgrad.

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
            m_grouped_bf16_gemm_nt_contiguous(a.contiguous(), w.contiguous(), out, self.layout.m_indices)
        else:
            m_grouped_fp8_gemm_nt_contiguous(
                quantize_rows(a.contiguous()), quantize_weights(w), out,
                self.layout.m_indices, recipe=(1, 128, 128), disable_ue8m0_cast=True)
        return out

    def wgrad(self, lhs, rhs):
        from . import k_grouped_bf16_gemm_tn_contiguous, k_grouped_fp8_gemm_nt_contiguous
        if lhs.size(0) != self.layout.rows or rhs.size(0) != self.layout.rows:
            raise ValueError("wgrad operands must share the padded token layout")
        out = torch.zeros((len(self.layout.ks), lhs.size(1), rhs.size(1)),
                          device=lhs.device, dtype=torch.float32)
        if self.precision == "bf16":
            k_grouped_bf16_gemm_tn_contiguous(
                lhs.contiguous(), rhs.contiguous(), out, list(self.layout.ks), self.layout.k_sizes, out)
        else:
            k_grouped_fp8_gemm_nt_contiguous(
                quantize_k_groups(lhs.contiguous(), self.layout),
                quantize_k_groups(rhs.contiguous(), self.layout),
                out, list(self.layout.ks), self.layout.k_sizes, out, recipe=(1, 1, 128))
        return out
