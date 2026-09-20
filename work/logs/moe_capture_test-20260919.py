"""Standalone CUDA-graph capture test for the fused MoE path.

Builds the same tensors as moe_equiv_test.py, then captures fused_apply in a
CUDA graph (the op that fails is printed). Also warms the ext first to
separate 'first-call init' failures from genuinely uncapturable ops.
"""
import json
import os
import struct
import sys

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
tg = torch.empty((C, ROWS, H), dtype=torch.float16, device='cuda')
tu = torch.empty_like(tg)
tig = torch.empty((C, ROWS, I), dtype=torch.float16, device='cuda')
tiu = torch.empty_like(tig)


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
    expert_count = torch.zeros(E + 1, dtype=torch.int64, device='cuda')
    expert_count.scatter_add_(0, flat_expert, torch.ones_like(flat_expert))
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


# ---- warm the ext OUTSIDE capture first (mirrors process_weights_after_loading)
x = torch.zeros((1, H), dtype=torch.float16, device='cuda')
ids = torch.zeros((1, 1), dtype=torch.int32, device='cuda')
w = torch.ones((1, 1), dtype=torch.float32, device='cuda')
fused_apply(x, w, ids)
torch.cuda.synchronize()
print('warmup call OK', flush=True)

# ---- static capture
sx = torch.randn(1, H, dtype=torch.float16, device='cuda') * 0.1
sids = torch.zeros((1, 8), dtype=torch.int32, device='cuda')
sw = torch.full((1, 8), 1 / 8, device='cuda')
for _ in range(2):
    fused_apply(sx, sw, sids)
torch.cuda.synchronize()

g = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(g):
        fused_apply(sx, sw, sids)
    print('CAPTURE OK', flush=True)
    g.replay()
    torch.cuda.synchronize()
    print('REPLAY OK', flush=True)
except Exception as exc:
    print('CAPTURE FAILED:', type(exc).__name__, str(exc)[:400], flush=True)
print('DONE')
