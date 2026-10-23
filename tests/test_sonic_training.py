"""Small SM90 adapter tests, including an empty expert and NT scale orientation."""
import torch

torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
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
    from deep_gemm import get_mn_major_tma_aligned_tensor
    assert get_mn_major_tma_aligned_tensor(s).data_ptr() == s.data_ptr()
    rq,rs=per_token_cast_to_fp8(x,False)
    torch.testing.assert_close(s,rs,rtol=1e-5,atol=1e-8)
    assert ((q.float()-rq.float()).norm()/rq.float().norm()).item()<.005
    q,s=quantize_weights(w)
    for e in range(3):
        rq,rs=per_block_cast_to_fp8(w[e].contiguous(),False)
        torch.testing.assert_close(s[e],rs,rtol=1e-5,atol=1e-8)
        assert ((q[e].float()-rq.float()).norm()/rq.float().norm()).item()<.005
    q,s=quantize_k_groups(x,layout)
    assert get_mn_major_tma_aligned_tensor(s).data_ptr() == s.data_ptr()
    start=0
    for size in layout.ks:
        if size:
            rq,rs=per_channel_cast_to_fp8(x[start:start+size],False)
            actual=q[start*k:(start+size)*k].view(k,size).T
            torch.testing.assert_close(s[:,start//128:(start+size)//128].T,rs,rtol=1e-5,atol=1e-8)
            assert ((actual.float()-rq.float()).norm()/rq.float().norm()).item()<.005
        start+=size
    # Tail row tiles must not write past Q/S or alter independent per-row scales.
    for rows in (3, 17):
        tail = torch.randn(rows, 384, device='cuda', dtype=torch.bfloat16)
        tq, ts = quantize_rows(tail)
        rq, rs = per_token_cast_to_fp8(tail, False)
        torch.testing.assert_close(ts, rs, rtol=1e-5, atol=1e-8)
        assert ((tq.float()-rq.float()).norm()/rq.float().norm()).item() < .005
    # Compare fused token gather+quantization with independently expanded inputs.
    tokens = torch.randn(17,k,device='cuda',dtype=torch.bfloat16)
    routes = torch.full((layout.rows,),-1,device='cuda',dtype=torch.int64)
    routes[:3] = torch.tensor([0,1,2],device='cuda')
    routes[128:257] = torch.arange(129,device='cuda')
    valid = routes >= 0
    expanded = torch.zeros(layout.rows,k,device='cuda',dtype=torch.bfloat16)
    expanded[valid] = tokens[routes[valid]//8]
    for explicit, fused in ((quantize_rows(expanded),quantize_rows(tokens,routes,8)),
                            (quantize_k_groups(expanded,layout),quantize_k_groups(tokens,layout,routes,8))):
        torch.testing.assert_close(explicit[0].view(torch.uint8),fused[0].view(torch.uint8),rtol=0,atol=0)
        torch.testing.assert_close(explicit[1],fused[1],rtol=0,atol=0)
    fused_g = GroupedTrainingGemm(layout,'fp8')
    torch.testing.assert_close(fused_g.m_gemm_gather(tokens,w,routes,8),fused_g.m_gemm(expanded,w),rtol=0,atol=0)
    torch.testing.assert_close(fused_g.wgrad(tokens,expanded,row_to_route=routes,topk=8,gather_lhs=True),
                               fused_g.wgrad(expanded,expanded),rtol=0,atol=0)
    torch.testing.assert_close(fused_g.wgrad(expanded,tokens,row_to_route=routes,topk=8,gather_rhs=True),
                               fused_g.wgrad(expanded,expanded),rtol=0,atol=0)
    single_layout = GroupedTrainingLayout.from_counts([17],torch.device('cuda'))
    single_x = torch.randn(single_layout.rows,k,device='cuda',dtype=torch.bfloat16)
    single_w = torch.randn(1,384,k,device='cuda',dtype=torch.bfloat16)
    for precision in ('bf16','fp8'):
        single = GroupedTrainingGemm(single_layout,precision)
        expected = single.m_gemm(single_x,single_w)
        actual = torch.cat([y.clone() for _,y in single.m_gemm_columns(single_x,single_w,256)],1)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    errors={}
    for precision in ('bf16','fp8'):
        g=GroupedTrainingGemm(layout,precision)
        out=g.m_gemm(x,w)
        ref=torch.cat([a.float()@b.float().T for a,b in zip(x.split(layout.ks),w)])
        err=((out.float()-ref).norm()/ref.norm()).item()
        assert err < (.005 if precision=='bf16' else .06), (precision,'m_gemm',err)
        tail_weights = torch.randn(3,384,k,device='cuda',dtype=torch.bfloat16)
        tail_ref = g.m_gemm(x,tail_weights)
        for columns in (128, 256):
            pieces = [tile.clone() for _,tile in g.m_gemm_columns(x,tail_weights,columns)]
            tiled = torch.cat(pieces,dim=1)
            torch.testing.assert_close(tiled,tail_ref,rtol=0,atol=0)
        dw=g.wgrad(x,x)
        ref=torch.stack([v.float().T@v.float() for v in x.split(layout.ks)])
        werr=((dw-ref).norm()/ref.norm()).item()
        assert werr < (.001 if precision=='bf16' else .06), (precision,'wgrad',werr)
        assert torch.count_nonzero(dw[1])==0
        # Direct BF16 epilogues remove the full FP32 gradient and cast workspace.
        compact = g.wgrad(x,x,output_dtype=torch.bfloat16)
        cerr = ((compact.float()-dw).norm()/dw.norm()).item()
        assert cerr < .005, (precision, 'bf16_wgrad', cerr)
        assert compact.dtype == torch.bfloat16 and torch.count_nonzero(compact[1]) == 0
        # Exercise non-square output tiles and a partial N tile with fresh output.
        rhs = torch.randn(layout.rows,384,device='cuda',dtype=torch.bfloat16)
        full = g.wgrad(x,rhs)
        compact = g.wgrad(x,rhs,output_dtype=torch.bfloat16)
        assert ((compact.float()-full).norm()/full.norm()).item() < .005
        if precision == 'fp8':
            from deep_gemm import k_grouped_fp8_gemm_nt_contiguous
            initial = torch.full_like(compact,.125)
            accumulated = torch.empty_like(compact)
            qa,qb = quantize_k_groups(x,layout),quantize_k_groups(rhs,layout)
            accumulated_zero = torch.zeros_like(compact)
            k_grouped_fp8_gemm_nt_contiguous(qa,qb,accumulated_zero,list(layout.ks),layout.k_sizes,
                                            accumulated_zero,recipe=(1,1,128))
            k_grouped_fp8_gemm_nt_contiguous(qa,qb,accumulated,list(layout.ks),layout.k_sizes,initial,
                                            recipe=(1,1,128))
            torch.testing.assert_close(accumulated,(accumulated_zero.float()+initial.float()).bfloat16(),rtol=0,atol=0)
        errors[precision]=dict(m_rrmse=err,wgrad_rrmse=werr)
    return errors


def test_adapters():
    validate_adapters()


def validate_wgrad_overwrite():
    """No-C writes every tile, while separate/aliased C keeps accumulation."""
    import deep_gemm as dg
    torch.manual_seed(983)
    reports = []
    # Leading, middle and trailing empty groups; tail N tiles; a long reduction;
    # enough expert groups to exercise multiple persistent scheduler iterations.
    for counts, m, n in (([0, 129, 0, 3, 0], 128, 384),
                          ([0, 32768, 1, 0], 256, 128),
                          ([1 if e % 7 == 0 else 0 for e in range(64)], 128, 128),
                          ([0, 0, 0], 128, 384)):
        layout = GroupedTrainingLayout.from_counts(counts, torch.device('cuda'))
        lhs = torch.randn(layout.rows, m, device='cuda', dtype=torch.bfloat16) * .125
        rhs = torch.randn(layout.rows, n, device='cuda', dtype=torch.bfloat16) * .125
        for precision in ('bf16', 'fp8'):
            gemms = GroupedTrainingGemm(layout, precision)
            if precision == 'fp8':
                operands = quantize_k_groups(lhs, layout), quantize_k_groups(rhs, layout)
                native = dg.k_grouped_fp8_gemm_nt_contiguous
            else:
                operands = lhs, rhs
                native = dg.k_grouped_bf16_gemm_tn_contiguous
            for dtype in (torch.float32, torch.bfloat16):
                baseline = gemms.wgrad(lhs, rhs, output_dtype=dtype, overwrite=False)
                direct = gemms.wgrad(lhs, rhs, output_dtype=dtype, overwrite=True)
                default = gemms.wgrad(lhs, rhs, output_dtype=dtype)
                torch.testing.assert_close(default, direct, rtol=0, atol=0)
                # Poison old D contents, proving overwrite does not read them.
                poisoned = torch.full_like(direct, float('nan'))
                native(*operands, poisoned, list(layout.ks), layout.k_sizes)
                torch.testing.assert_close(poisoned, direct, rtol=0, atol=0)
                assert torch.isfinite(direct).all()
                error = ((direct.float()-baseline.float()).norm() / baseline.float().norm().clamp_min(1e-8)).item()
                assert error < (.005 if dtype == torch.bfloat16 else .0001), (precision, counts, error)
                for e, count in enumerate(counts):
                    if count == 0:
                        assert torch.count_nonzero(direct[e]) == 0
                initial = torch.full_like(direct, .125)
                separate = torch.empty_like(direct)
                native(*operands, separate, list(layout.ks), layout.k_sizes, initial)
                aliased = initial.clone()
                native(*operands, aliased, list(layout.ks), layout.k_sizes, aliased)
                torch.testing.assert_close(separate, aliased, rtol=0, atol=0)
                torch.testing.assert_close(initial, torch.full_like(initial, .125), rtol=0, atol=0)
                torch.testing.assert_close(separate, (baseline.float()+initial.float()).to(dtype), rtol=0, atol=0)
                reports.append(dict(precision=precision, dtype=str(dtype), counts=counts, m=m, n=n, rrmse=error))
    return reports


def test_wgrad_overwrite():
    validate_wgrad_overwrite()


def validate_k_group_overwrite_m_boundaries():
    """A no-C tile must never overwrite the next expert through flattened M.

    Multi-expert SM90 overwrite rejects non-64-aligned M, except when an empty
    reduction takes the safe zero-fill return. Aligned M still needs a tile
    size that divides M: M=64/192 must not use BM=128/256. Single-expert tails
    remain safe because the TMA descriptor ends at that expert's last row.
    """
    import deep_gemm as dg
    torch.manual_seed(1753)
    reports = []
    for m in (64, 128, 136, 192, 256, 2176):
        for counts in ([128, 128], [128, 0, 256], [0, 128, 0], [0, 0], [128]):
            layout = GroupedTrainingLayout.from_counts(counts, torch.device('cuda'))
            lhs = torch.randn(layout.rows, m, device='cuda', dtype=torch.bfloat16) * .125
            rhs = torch.randn(layout.rows, 128, device='cuda', dtype=torch.bfloat16) * .125
            reference = torch.stack([a.float().T @ b.float()
                                     for a, b in zip(lhs.split(layout.ks), rhs.split(layout.ks))])
            rejected = len(counts) > 1 and layout.rows > 0 and m % 64 != 0
            for precision in ('fp8', 'bf16'):
                if precision == 'fp8':
                    operands = quantize_k_groups(lhs, layout), quantize_k_groups(rhs, layout)
                    native = dg.k_grouped_fp8_gemm_nt_contiguous
                else:
                    operands, native = (lhs, rhs), dg.k_grouped_bf16_gemm_tn_contiguous
                for dtype in (torch.float32, torch.bfloat16):
                    stage = (m, counts, precision, str(dtype))
                    print('M_BOUNDARY_BEGIN', *stage, flush=True)
                    # C-present is unchanged, including M=136. Nonzero C also
                    # checks that an empty expert preserves its initial values.
                    initial = torch.full((len(counts), m, 128), .125, device='cuda', dtype=dtype)
                    accumulated = torch.empty_like(initial)
                    native(*operands, accumulated, list(layout.ks), layout.k_sizes, initial)
                    reference_with_c = reference + initial.float()
                    error = ((accumulated.float()-reference_with_c).norm()
                             / reference_with_c.norm().clamp_min(1e-8)).item()
                    assert error < (.06 if precision == 'fp8' else .005), (stage, error)
                    torch.testing.assert_close(initial, torch.full_like(initial, .125), rtol=0, atol=0)
                    for expert, count in enumerate(counts):
                        if not count:
                            torch.testing.assert_close(accumulated[expert], initial[expert], rtol=0, atol=0)

                    poisoned = torch.full_like(initial, float('nan'))
                    # FP8 also admits regular column slices. Padding columns
                    # must survive, including when the no-C M contract rejects.
                    backing = torch.full((len(counts), m, 384), float('nan'), device='cuda', dtype=dtype)
                    outputs = [poisoned]
                    if precision == 'fp8':
                        outputs.append(backing[:, :, 128:256])
                    for output in outputs:
                        if rejected:
                            try:
                                native(*operands, output, list(layout.ks), layout.k_sizes, None)
                            except RuntimeError as exc:
                                assert 'SM90 multi-group overwrite requires M divisible by 64' in str(exc), str(exc)
                            else:
                                raise AssertionError(('unsafe M overwrite accepted', stage, output.stride()))
                            assert torch.isnan(output).all()
                        else:
                            native(*operands, output, list(layout.ks), layout.k_sizes, None)
                            torch.cuda.synchronize()
                            assert torch.isfinite(output).all(), stage
                            for expert, count in enumerate(counts):
                                if not count:
                                    assert torch.count_nonzero(output[expert]) == 0, stage
                                else:
                                    expert_error = ((output[expert].float()-reference[expert]).norm()
                                                    / reference[expert].norm().clamp_min(1e-8)).item()
                                    assert expert_error < (.06 if precision == 'fp8' else .005), (stage, expert, expert_error)
                    if precision == 'fp8':
                        assert torch.isnan(backing[:, :, :128]).all()
                        assert torch.isnan(backing[:, :, 256:]).all()
                        if not rejected:
                            torch.testing.assert_close(outputs[1], poisoned, rtol=0, atol=0)
                    reports.append(dict(m=m, counts=counts, precision=precision, dtype=str(dtype),
                                        overwrite_rejected=rejected, accumulation_rrmse=error))
                    print('M_BOUNDARY_DONE', *stage, flush=True)
    return reports


def test_k_group_overwrite_m_boundaries():
    validate_k_group_overwrite_m_boundaries()


def validate_wgrad_gather_rhs_columns():
    """Bounded gathers match full wgrad, including native prequantized lhs."""
    torch.manual_seed(1637)
    reports = []
    cases = (([0, 129, 0, 3, 0], 128, 2176, (1024, 2048)),
             ([1 if e % 7 == 0 else 0 for e in range(64)], 256, 384, (256,)),
             ([129, 0, 3], 128, 264, (256,)),
             ([0, 0, 0], 128, 2176, (2048,)))
    for case, (counts, m, n, chunks) in enumerate(cases):
        layout = GroupedTrainingLayout.from_counts(counts, torch.device('cuda'))
        gemms = GroupedTrainingGemm(layout, 'fp8')
        token_count, topk = (17 if layout.rows else 0), 8
        # Noncontiguous rows expose source-stride mistakes in gathered chunks.
        tokens = torch.randn(token_count * 2, n + 16, device='cuda', dtype=torch.bfloat16)[::2, :n]
        lhs = torch.randn(layout.rows, m, device='cuda', dtype=torch.bfloat16) * .125
        rowmap = torch.full((layout.rows,), -1, device='cuda', dtype=torch.int64)
        offset = 0
        for count, padded in zip(counts, layout.ks):
            if count:
                rowmap[offset:offset+count] = torch.arange(count, device='cuda') % (token_count * topk)
            offset += padded
        valid = rowmap >= 0
        expanded = torch.zeros(layout.rows, n, device='cuda', dtype=torch.bfloat16)
        expanded[valid] = tokens[rowmap[valid] // topk]
        lhs[~valid] = 0
        reference = torch.stack([a.float().T @ b.float()
                                 for a, b in zip(lhs.split(layout.ks), expanded.split(layout.ks))])
        qa = quantize_k_groups(lhs, layout)
        q_before, s_before = qa[0].view(torch.uint8).clone(), qa[1].clone()
        for dtype in (torch.float32, torch.bfloat16):
            full = gemms.wgrad(lhs, expanded, output_dtype=dtype)
            for columns in chunks:
                print('GATHER_COLUMNS_BEGIN', case, str(dtype), columns, flush=True)
                actual = gemms.wgrad_gather_rhs_columns(
                    lhs, tokens, rowmap, topk, columns=columns, output_dtype=dtype)
                prequantized = gemms.wgrad_gather_rhs_columns(
                    None, tokens, rowmap, topk, columns=columns, output_dtype=dtype, quantized_lhs=qa)
                torch.cuda.synchronize()
                assert torch.isfinite(actual).all()
                torch.testing.assert_close(actual, prequantized, rtol=0, atol=0)
                error = ((actual.float()-full.float()).norm() / full.float().norm().clamp_min(1e-8)).item()
                assert error < (.005 if dtype == torch.bfloat16 else .0001), (case, dtype, columns, error)
                reference_error = ((actual.float()-reference).norm() / reference.norm().clamp_min(1e-8)).item()
                assert reference_error < .06, (case, dtype, columns, reference_error)
                for expert, count in enumerate(counts):
                    if not count:
                        assert torch.count_nonzero(actual[expert]) == 0
                if columns == 2048:
                    default = gemms.wgrad_gather_rhs_columns(lhs, tokens, rowmap, topk, output_dtype=dtype)
                    torch.testing.assert_close(default, actual, rtol=0, atol=0)
                reports.append(dict(case=case, counts=counts, m=m, n=n, columns=columns,
                                    dtype=str(dtype), full_rrmse=error, fp32_rrmse=reference_error))
                print('GATHER_COLUMNS_DONE', case, str(dtype), columns, flush=True)
        torch.testing.assert_close(qa[0].view(torch.uint8), q_before, rtol=0, atol=0)
        torch.testing.assert_close(qa[1], s_before, rtol=0, atol=0)
    return reports


def test_wgrad_gather_rhs_columns():
    validate_wgrad_gather_rhs_columns()


def validate_wgrad_column_output():
    """No-C writes regular column slices without touching their sibling columns."""
    import deep_gemm as dg
    torch.manual_seed(1481)
    native = dg.k_grouped_fp8_gemm_nt_contiguous
    reports = []
    cases = (([0, 129, 0, 3, 0], 128, 384, 128, 1024),
             ([129, 0, 3, 0], 256, 128, 2048, 2176),
             ([1 if e % 7 == 0 else 0 for e in range(64)], 128, 136, 8, 512),
             ([0, 0, 0], 128, 384, 128, 1024))
    for case, (counts, m, n, start, full_n) in enumerate(cases):
        layout = GroupedTrainingLayout.from_counts(counts, torch.device('cuda'))
        lhs = torch.randn(layout.rows, m, device='cuda', dtype=torch.bfloat16) * .125
        rhs = torch.randn(layout.rows, n, device='cuda', dtype=torch.bfloat16) * .125
        operands = quantize_k_groups(lhs, layout), quantize_k_groups(rhs, layout)
        for dtype in (torch.bfloat16, torch.float32):
            print('COLUMN_OUTPUT_BEGIN', case, str(dtype), flush=True)
            expected = torch.empty((len(counts), m, n), device='cuda', dtype=dtype)
            native(*operands, expected, list(layout.ks), layout.k_sizes, None)
            backing = torch.full((len(counts), m, full_n), float('nan'), device='cuda', dtype=dtype)
            view = backing[:, :, start:start+n]
            native(*operands, view, list(layout.ks), layout.k_sizes, None)
            torch.cuda.synchronize()
            torch.testing.assert_close(view, expected, rtol=0, atol=0)
            assert torch.isfinite(view).all()
            assert torch.isnan(backing[:, :, :start]).all()
            assert torch.isnan(backing[:, :, start+n:]).all()
            for expert, count in enumerate(counts):
                if not count:
                    assert torch.count_nonzero(view[expert]) == 0
            # The unchanged contiguous C-present contract still accumulates.
            accumulated_zero = torch.zeros_like(expected)
            native(*operands, accumulated_zero, list(layout.ks), layout.k_sizes, accumulated_zero)
            initial = torch.full_like(expected, .125)
            accumulated = torch.empty_like(expected)
            native(*operands, accumulated, list(layout.ks), layout.k_sizes, initial)
            torch.testing.assert_close(accumulated, (accumulated_zero.float()+initial.float()).to(dtype),
                                       rtol=0, atol=0)
            torch.testing.assert_close(initial, torch.full_like(initial, .125), rtol=0, atol=0)
            # A strided D with either separate or aliased C is not admitted.
            for c in (initial, view):
                try:
                    native(*operands, view, list(layout.ks), layout.k_sizes, c)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError('strided K-group accumulation was accepted')
            torch.testing.assert_close(view, expected, rtol=0, atol=0)
            reports.append(dict(case=case, counts=counts, m=m, n=n, start=start, full_n=full_n,
                                dtype=str(dtype), strided_exact=True, siblings_unchanged=True))
            print('COLUMN_OUTPUT_DONE', case, str(dtype), flush=True)
    # Reject layouts that cannot flatten groups/M into one regular TMA stride.
    layout = GroupedTrainingLayout.from_counts([128, 128], torch.device('cuda'))
    operands = (quantize_k_groups(torch.randn(256, 128, device='cuda', dtype=torch.bfloat16), layout),
                quantize_k_groups(torch.randn(256, 384, device='cuda', dtype=torch.bfloat16), layout))
    for dtype in (torch.bfloat16, torch.float32):
        backing = torch.zeros((2, 128, 1024), device='cuda', dtype=dtype)
        gaps = torch.zeros((2, 129, 1024), device='cuda', dtype=dtype)
        unaligned_rows = torch.zeros((2, 128, 513), device='cuda', dtype=dtype)
        invalid = (backing[:, :, 1:385], backing[:, :, :768:2], gaps[:, :128, :384],
                   unaligned_rows[:, :, :384], backing.as_strided((2, 128, 384), (128*376, 376, 1)))
        for index, output in enumerate(invalid):
            try:
                native(*operands, output, list(layout.ks), layout.k_sizes, None)
            except RuntimeError:
                pass
            else:
                raise AssertionError(('invalid column output accepted', dtype, index, output.stride()))
    return reports


def test_wgrad_column_output():
    validate_wgrad_column_output()


def validate_k_group_multicast_boundaries():
    """Exercise odd expert tile counts that must not split multicast clusters.

    The first FP8/BF16-output shape previously chose BLOCK_M=128, BLOCK_N=48,
    cluster_n=2: 17*3=51 tiles per expert crossed a physical two-CTA cluster.
    Print and synchronize every native stage so a stalled launch is identifiable.
    """
    import deep_gemm as dg
    torch.manual_seed(1137)
    reports = []
    cases = (
        ([128, 128, 128, 128], 2176, 128),  # Original hanging dW2 shape.
        ([128, 256, 384, 512], 128, 2176),  # Transpose, differing K lengths.
        ([129, 0, 3, 0], 2176, 128),       # Middle/trailing empty groups.
        ([0, 129, 0, 3], 128, 2176),       # Leading/middle empty groups.
        ([128, 128, 128, 128], 256, 2176),  # Even tile counts retain multicast candidates.
    )
    for case, (counts, m, n) in enumerate(cases):
        print('MULTICAST_BEGIN', case, 'inputs', counts, m, n, flush=True)
        layout = GroupedTrainingLayout.from_counts(counts, torch.device('cuda'))
        lhs = torch.randn(layout.rows, m, device='cuda', dtype=torch.bfloat16) * .125
        rhs = torch.randn(layout.rows, n, device='cuda', dtype=torch.bfloat16) * .125
        reference = torch.stack([a.float().T @ b.float()
                                 for a, b in zip(lhs.split(layout.ks), rhs.split(layout.ks))])
        torch.cuda.synchronize()
        print('MULTICAST_DONE', case, 'inputs', flush=True)
        # Test the observed hanging precision first.
        for precision in ('fp8', 'bf16'):
            print('MULTICAST_BEGIN', case, precision, 'operands', flush=True)
            if precision == 'fp8':
                operands = quantize_k_groups(lhs, layout), quantize_k_groups(rhs, layout)
                native = dg.k_grouped_fp8_gemm_nt_contiguous
            else:
                operands, native = (lhs, rhs), dg.k_grouped_bf16_gemm_tn_contiguous
            torch.cuda.synchronize()
            print('MULTICAST_DONE', case, precision, 'operands', flush=True)
            for dtype in (torch.bfloat16, torch.float32):
                for overwrite in (False, True):
                    output = torch.full((len(counts), m, n), float('nan') if overwrite else 0.,
                                        device='cuda', dtype=dtype)
                    stage = (case, precision, str(dtype), 'overwrite' if overwrite else 'accumulate')
                    print('MULTICAST_BEGIN', *stage, flush=True)
                    native(*operands, output, list(layout.ks), layout.k_sizes,
                           None if overwrite else output)
                    torch.cuda.synchronize()
                    print('MULTICAST_DONE', *stage, flush=True)
                    assert torch.isfinite(output).all(), stage
                    error = ((output.float() - reference).norm() / reference.norm().clamp_min(1e-8)).item()
                    assert error < (.06 if precision == 'fp8' else .005), (stage, error)
                    for expert, count in enumerate(counts):
                        if not count:
                            assert torch.count_nonzero(output[expert]) == 0, (stage, expert)
                    reports.append(dict(case=case, counts=counts, m=m, n=n, precision=precision,
                                        dtype=str(dtype), overwrite=overwrite, rrmse=error))
    return reports


def test_k_group_multicast_boundaries():
    validate_k_group_multicast_boundaries()


def validate_large_quantization_offsets():
    """Cross signed-int32 element offsets without allocating expanded BF16 X."""
    width, routes, topk = 4096, 2**19, 16
    layout = GroupedTrainingLayout.from_counts([routes, 0, 1], torch.device('cuda'))
    tokens = torch.linspace(-2, 2, 2*width, device='cuda').reshape(2, width).bfloat16()
    rowmap = torch.zeros(layout.rows, device='cuda', dtype=torch.int64)
    rowmap[routes] = topk
    rowmap[routes + 1:] = -1
    expected_q, expected_s = quantize_rows(tokens)
    q, s = quantize_rows(tokens, rowmap, topk)
    rows = torch.tensor([0, routes-1, routes], device='cuda')
    source = torch.tensor([0, 0, 1], device='cuda')
    torch.testing.assert_close(q.view(torch.uint8)[rows], expected_q.view(torch.uint8)[source], rtol=0, atol=0)
    torch.testing.assert_close(s[rows], expected_s[source], rtol=0, atol=0)
    assert torch.count_nonzero(q[routes+1:].float()) == 0
    torch.cuda.synchronize()
    del q, s
    q, s = quantize_k_groups(tokens, layout, rowmap, topk)
    # The last expert starts at exactly 2**31 FP8 elements in the packed tensor.
    tail = q[routes*width:].view(width, layout.ks[-1])
    scale = tokens[1].float().abs().clamp_min(1.e-4) / 448.
    expected = (tokens[1].float() / scale).to(torch.float8_e4m3fn).float()
    torch.testing.assert_close(tail[:, 0].float(), expected, rtol=0, atol=0)
    torch.testing.assert_close(s[:, routes//128], scale, rtol=1e-6, atol=1e-9)
    assert torch.count_nonzero(tail[:, 1:].float()) == 0
    torch.cuda.synchronize()
    return {'rows': layout.rows, 'width': width, 'packed_tail_start': routes*width}


def test_large_quantization_offsets():
    validate_large_quantization_offsets()


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
