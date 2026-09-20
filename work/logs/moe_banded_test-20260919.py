"""Standalone equivalence + timing: fused exl3_moe path vs the per-expert loop.

Uses real expert tensors (layer 3, experts 0..7) from the GLM-5.3-Flash EXL3
pack. Both paths compute the same MoE forward; compare on random routings.
"""
import json
import os
import struct
import sys
import time

import torch

P = '/home/r0b0tdgx/models/glm-5.3-flash/exl3-2.25hq-tapK3'
LAYER = 3
E = 8
H, I = 4096, 2048

sys.path.insert(0, '/home/r0b0tdgx/vllm-exl3-sm121/src')
import vllm_exl3_sm121_ext as ext

idx = json.load(open(os.path.join(P, 'model.safetensors.index.json')))['weight_map']


def read(name):
    sh = idx[name]
    path = os.path.join(P, sh)
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n))
        hdr.pop('__metadata__', None)
        off = hdr[name]['data_offsets']
        dt = hdr[name]['dtype']
        shape = hdr[name]['shape']
        f.seek(8 + n + off[0])
        raw = f.read(off[1] - off[0])
    t = {'I16': torch.int16, 'F16': torch.float16, 'I32': torch.int32}[dt]
    return torch.frombuffer(bytearray(raw), dtype=t).reshape(shape).to('cuda')


base = f'model.language_model.layers.{LAYER}.mlp.experts'
t13 = torch.empty((E, 2, 256, 128, 48), dtype=torch.int16, device='cuda')
s13 = torch.empty((E, 2, H), dtype=torch.float16, device='cuda')
v13 = torch.empty((E, 2, I), dtype=torch.float16, device='cuda')
t2 = torch.empty((E, 128, 256, 48), dtype=torch.int16, device='cuda')
s2 = torch.empty((E, I), dtype=torch.float16, device='cuda')
v2 = torch.empty((E, H), dtype=torch.float16, device='cuda')
for e in range(E):
    t13[e, 0] = read(f'{base}.{e}.gate_proj.trellis')
    t13[e, 1] = read(f'{base}.{e}.up_proj.trellis')
    s13[e, 0] = read(f'{base}.{e}.gate_proj.suh')
    s13[e, 1] = read(f'{base}.{e}.up_proj.suh')
    v13[e, 0] = read(f'{base}.{e}.gate_proj.svh')
    v13[e, 1] = read(f'{base}.{e}.up_proj.svh')
    t2[e] = read(f'{base}.{e}.down_proj.trellis')
    s2[e] = read(f'{base}.{e}.down_proj.suh')
    v2[e] = read(f'{base}.{e}.down_proj.svh')
torch.cuda.synchronize()
print('tensors loaded', flush=True)

K13, K2 = 3, 3
mcg, mul1 = False, True


def loop_apply(x, w, ids):
    out = torch.zeros_like(x)
    for e in ids.unique().tolist():
        rows = (ids == e).nonzero()[:, 0]
        tok = x[rows].contiguous()
        ww = w[rows, (ids[rows] == e).nonzero()[:, 1]].to(out.dtype)
        g = ext.exl3_gemm(tok, t13[e, 0], s13[e, 0], v13[e, 0], K13, mcg, mul1, I)
        u = ext.exl3_gemm(tok, t13[e, 1], s13[e, 1], v13[e, 1], K13, mcg, mul1, I)
        h = torch.nn.functional.silu(g) * u
        d = ext.exl3_gemm(h, t2[e], s2[e], v2[e], K2, mcg, mul1, H)
        out[rows] += d * ww.unsqueeze(-1)
    return out


def half_ptrs(t):
    es = t.element_size()
    b = t.data_ptr()
    s0, s1 = t.stride(0), t.stride(1)
    g = [b + i * s0 * es for i in range(E)]
    u = [b + (i * s0 + s1) * es for i in range(E)]
    return (torch.tensor(g, dtype=torch.int64, device='cuda'),
            torch.tensor(u, dtype=torch.int64, device='cuda'))


def flat_ptrs(t):
    es = t.element_size()
    b = t.data_ptr()
    s0 = t.stride(0)
    return torch.tensor([b + i * s0 * es for i in range(E)],
                        dtype=torch.int64, device='cuda')


gt, ut = half_ptrs(t13)
gs, us = half_ptrs(s13)
gv, uv = half_ptrs(v13)
dt_, ds, dv = flat_ptrs(t2), flat_ptrs(s2), flat_ptrs(v2)
ROWS = 256
C = ext.exl3_moe_max_concurrency(0)
print('max_concurrency:', C, flush=True)
tg = torch.empty((C, ROWS, H), dtype=torch.float16, device='cuda')
tu = torch.empty_like(tg)
tig = torch.empty((C, ROWS, I), dtype=torch.float16, device='cuda')
tiu = torch.empty_like(tig)


def fused_apply_banded(x, w, ids, counts):
    """Mirror of the plugin's banded path: one launch per row-tile band."""
    num_tokens, top_k = ids.shape
    a = num_tokens * top_k
    x16 = x.to(torch.float16).contiguous()
    flat_expert = ids.reshape(-1).to(torch.int64)
    flat_weight = w.reshape(-1).to(torch.float16)
    flat_token = torch.arange(num_tokens, dtype=torch.int64,
                              device='cuda').repeat_interleave(top_k)
    order = flat_expert.argsort()
    token_sorted = flat_token[order].contiguous()
    weight_sorted = flat_weight[order].contiguous()
    expert_count = torch.zeros(E + 1, dtype=torch.int64, device='cuda')
    expert_count.scatter_add_(0, flat_expert, torch.ones_like(flat_expert))
    inv_order = torch.empty_like(order)
    inv_order.scatter_(0, order, torch.arange(a, dtype=torch.int64, device='cuda'))
    expert_start = torch.cumsum(expert_count, 0) - expert_count
    tables = torch.stack([expert_start, expert_start,
                          (expert_count > 0).to(torch.int64)])
    scratch = torch.empty((max(a, 1), H), dtype=torch.float32, device='cuda')
    out = torch.zeros((num_tokens, H), dtype=torch.float32, device='cuda')
    t0 = sum(1 for c in counts if 0 < c <= 16)
    t1 = sum(1 for c in counts if 16 < c <= 32)
    t2 = sum(1 for c in counts if 32 < c <= ROWS)
    launches = []
    if t2: launches.append((t2, 33, ROWS, 64))
    if t1: launches.append((t1, 17, 32, 32))
    if t0: launches.append((t0, 1, 16, 16))
    for na, lo, hi, mt in launches:
        ext.exl3_moe(x16, out, expert_count, token_sorted, weight_sorted,
                     tg, tu, tig, tiu, 0, K13, K13, K2,
                     gt, gs, gv, ut, us, uv, dt_, ds, dv,
                     mcg, mul1, mcg, mul1, mcg, mul1, 0.0, na,
                     scratch, tables[0], lo, hi, mt)
    ext.exl3_moe_gather(out, scratch, flat_expert, inv_order,
                        tables[1, :E], tables[0, :E], tables[2, :E],
                        weight_sorted)
    return out.to(x.dtype)


def fused_apply(x, w, ids):
    num_tokens, top_k = ids.shape
    a = num_tokens * top_k
    x16 = x.to(torch.float16).contiguous()
    flat_expert = ids.reshape(-1).to(torch.int64)
    flat_weight = w.reshape(-1).to(torch.float16)
    flat_token = torch.arange(num_tokens, dtype=torch.int64,
                              device='cuda').repeat_interleave(top_k)
    order = flat_expert.argsort()
    token_sorted = flat_token[order].contiguous()
    weight_sorted = flat_weight[order].contiguous()
    expert_count = torch.bincount(flat_expert, minlength=E + 1)
    inv_order = torch.empty_like(order)
    inv_order.scatter_(0, order, torch.arange(a, dtype=torch.int64, device='cuda'))
    expert_start = torch.cumsum(expert_count, 0) - expert_count
    tables = torch.stack([expert_start, expert_start,
                          (expert_count > 0).to(torch.int64)])
    scratch = torch.empty((max(a, 1), H), dtype=torch.float32, device='cuda')
    out = torch.zeros((num_tokens, H), dtype=torch.float32, device='cuda')
    ext.exl3_moe(x16, out, expert_count, token_sorted, weight_sorted,
                 tg, tu, tig, tiu, 0, K13, K13, K2,
                 gt, gs, gv, ut, us, uv, dt_, ds, dv,
                 mcg, mul1, mcg, mul1, mcg, mul1, 0.0, -1,
                 scratch, tables[0], 1, ROWS, 16)
    ext.exl3_moe_gather(out, scratch, flat_expert, inv_order,
                        tables[1, :E], tables[0, :E], tables[2, :E],
                        weight_sorted)
    return out.to(x.dtype)


torch.manual_seed(0)
for num_tokens, top_k in ((1, 8), (4, 8), (16, 8), (64, 8), (128, 8)):
    x = torch.randn(num_tokens, H, dtype=torch.float16, device='cuda') * 0.1
    ids = torch.stack([torch.randperm(E, device='cuda')[:top_k]
                       for _ in range(num_tokens)])
    w = torch.rand(num_tokens, top_k, device='cuda', dtype=torch.float32)
    w = (w / w.sum(-1, keepdim=True))
    o1 = loop_apply(x, w, ids)
    if num_tokens * top_k <= ROWS:
        o2 = fused_apply(x, w, ids)
    else:
        c = torch.zeros(E, dtype=torch.int64, device='cuda')
        c.scatter_add_(0, ids.reshape(-1).to(torch.int64),
                       torch.ones(ids.numel(), dtype=torch.int64, device='cuda'))
        counts = c.tolist()
        assert max(counts) <= ROWS, counts
        o2 = fused_apply_banded(x, w, ids, counts)
    d = (o1.float() - o2.float()).abs()
    rel = d.max().item() / max(o1.float().abs().max().item(), 1e-6)
    print(f'tokens={num_tokens} topk={top_k}: max_abs={d.max().item():.6f} '
          f'rel={rel:.6f}', flush=True)

# timing at decode shape (1 token, 8 experts)
x = torch.randn(1, H, dtype=torch.float16, device='cuda') * 0.1
ids = torch.stack([torch.randperm(E, device='cuda')[:8]])
w = torch.full((1, 8), 1 / 8, device='cuda')
for name, fn in (('loop', loop_apply), ('fused', fused_apply)):
    for _ in range(3):
        fn(x, w, ids)
    torch.cuda.synchronize()
    t0 = time.time()
    n = 50
    for _ in range(n):
        fn(x, w, ids)
    torch.cuda.synchronize()
    print(f'{name}: {(time.time() - t0) / n * 1000:.3f} ms per layer call '
          f'(1 token, 8 experts)', flush=True)
print('DONE')
