"""Small SM90 adapter tests, including an empty expert and NT scale orientation."""
import torch
from deep_gemm.sonic_training import (
    GroupedTrainingLayout, GroupedTrainingGemm, quantize_k_groups,
    quantize_rows, quantize_weights,
)
from deep_gemm.utils import per_channel_cast_to_fp8, per_token_cast_to_fp8, per_block_cast_to_fp8


def validate_adapters():
    torch.manual_seed(55)
    layout=GroupedTrainingLayout.from_counts([3,0,129],torch.device('cuda'))
    n,k=256,256
    x=torch.randn(layout.rows,k,device='cuda',dtype=torch.bfloat16)
    w=torch.randn(3,n,k,device='cuda',dtype=torch.bfloat16).transpose(1,2)
    q,s=quantize_rows(x)
    rq,rs=per_token_cast_to_fp8(x,False)
    torch.testing.assert_close(s,rs,rtol=1e-5,atol=1e-8)
    assert ((q.float()-rq.float()).norm()/rq.float().norm()).item()<.005
    q,s=quantize_weights(w)
    for e in range(3):
        rq,rs=per_block_cast_to_fp8(w[e].contiguous(),False)
        torch.testing.assert_close(s[e],rs,rtol=1e-5,atol=1e-8)
        assert ((q[e].float()-rq.float()).norm()/rq.float().norm()).item()<.005
    q,s=quantize_k_groups(x,layout)
    start=0
    for size in layout.ks:
        if size:
            rq,rs=per_channel_cast_to_fp8(x[start:start+size],False)
            actual=q[start*k:(start+size)*k].view(k,size).T
            torch.testing.assert_close(s[:,start//128:(start+size)//128].T,rs,rtol=1e-5,atol=1e-8)
            assert ((actual.float()-rq.float()).norm()/rq.float().norm()).item()<.005
        start+=size
    errors={}
    for precision in ('bf16','fp8'):
        g=GroupedTrainingGemm(layout,precision)
        out=g.m_gemm(x,w)
        ref=torch.cat([a.float()@b.float().T for a,b in zip(x.split(layout.ks),w)])
        err=((out.float()-ref).norm()/ref.norm()).item()
        assert err < (.005 if precision=='bf16' else .06), (precision,'m_gemm',err)
        dw=g.wgrad(x,x)
        ref=torch.stack([v.float().T@v.float() for v in x.split(layout.ks)])
        werr=((dw-ref).norm()/ref.norm()).item()
        assert werr < (.001 if precision=='bf16' else .06), (precision,'wgrad',werr)
        assert torch.count_nonzero(dw[1])==0
        errors[precision]=dict(m_rrmse=err,wgrad_rrmse=werr)
    return errors


def test_adapters():
    validate_adapters()


def validate_scatter():
    import deep_gemm as dg
    torch.manual_seed(51)
    counts = [3, 0, 129]
    layout = GroupedTrainingLayout.from_counts(counts, torch.device("cuda"))
    t, topk, n, k = 17, 8, 384, 256
    routes = torch.full((layout.rows,), -1, device="cuda", dtype=torch.int64)
    logical = 0
    for e, count in enumerate(counts):
        start = int(layout.offsets[e])
        routes[start:start+count] = torch.arange(logical, logical+count, device="cuda") % (t*topk)
        logical += count
    # Padding is deliberately nonzero; only the route map can mask it.
    x = torch.randn(layout.rows, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(3, n, k, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(3, n, device="cuda", dtype=torch.bfloat16)
    scores = torch.randn(t, topk, device="cuda", dtype=torch.float32)
    scores[:, 0] = 0
    valid = routes >= 0
    y = torch.cat([a.float() @ b.float().T for a, b in zip(x.split(layout.ks), w)])
    errors = {}
    for precision in ("bf16", "fp8"):
        fn = getattr(dg, f"m_grouped_{precision}_gemm_nt_scatter")
        operands = (x, w) if precision == "bf16" else (quantize_rows(x), quantize_weights(w))
        for weighted in (False, True):
            out = torch.full((t, n), .125, device="cuda", dtype=torch.float32)
            values = y + bias[layout.m_indices.long()].float() if weighted else y
            if weighted:
                values = values[valid] * scores.flatten()[routes[valid], None]
            else:
                values = values[valid]
            ref = torch.full_like(out, .125).index_add_(0, routes[valid] // topk, values)
            fn(*operands, out, layout.m_indices, routes, topk,
               scores if weighted else None, bias if weighted else None)
            error = ((out-ref).norm()/ref.norm()).item()
            assert error < (1e-5 if precision == "bf16" else .07), (precision, weighted, error)
            errors[f"{precision}_{weighted}"] = error
        # Both deterministic switches must reject atomic reduction before launch.
        for library in ("torch", "deep_gemm"):
            setter = torch.use_deterministic_algorithms if library == "torch" else dg.use_deterministic_algorithms
            setter(True)
            try:
                try:
                    fn(*operands, out, layout.m_indices, routes, topk)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("scatter accepted deterministic mode")
            finally:
                setter(False)
        for invalid in (routes.int(), routes[:-1]):
            try:
                fn(*operands, out, layout.m_indices, invalid, topk)
            except RuntimeError:
                pass
            else:
                raise AssertionError("scatter accepted invalid route metadata")
    return errors


def test_scatter():
    validate_scatter()
