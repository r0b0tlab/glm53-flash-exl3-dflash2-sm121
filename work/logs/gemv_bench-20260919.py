"""Dense-path microbench: EXL3_GEMV modes (0=off, 1=heuristic, 2=force) at m=1.

Real pack tensors, m=1 decode shape. Also checks output agreement between the
GEMV kernel and the regular GEMM kernel.
"""
import json
import os
import struct
import sys
import time

import torch

P = '/home/r0b0tdgx/models/glm-5.3-flash/exl3-2.25hq-tapK3'
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


L0 = 'model.language_model.layers.0'
CASES = [
    ('attn qkv_proj (K4, 4096->24576)', f'{L0}.self_attn.qkv_proj'),
    ('attn o_proj  (K4, 8192->4096)', f'{L0}.self_attn.o_proj'),
    ('mlp gate_proj(K3, 4096->12288)', f'{L0}.mlp.gate_proj'),
    ('mlp down_proj(K3, 12288->4096)', f'{L0}.mlp.down_proj'),
]

loaded = []
for label, base in CASES:
    t = read(base + '.trellis')
    s = read(base + '.suh')
    v = read(base + '.svh')
    k = t.shape[-1] // 16
    n_out = v.shape[0]
    in_dim = s.shape[0]
    loaded.append((label, t, s, v, k, n_out, in_dim))

print('loaded', len(loaded), 'dense linears', flush=True)

results = {}
ref_out = {}
for mode in (0, 1, 2):
    os.environ['EXL3_GEMV'] = str(mode)
    for label, t, s, v, k, n_out, in_dim in loaded:
        x = (torch.randn(1, in_dim, dtype=torch.float16, device='cuda') * 0.1)
        for _ in range(3):
            ext.exl3_gemm(x, t, s, v, k, False, True, n_out)
        torch.cuda.synchronize()
        t0 = time.time()
        n = 200
        for _ in range(n):
            y = ext.exl3_gemm(x, t, s, v, k, False, True, n_out)
        torch.cuda.synchronize()
        ms = (time.time() - t0) / n * 1000
        results[(mode, label)] = ms
        if mode == 0:
            ref_out[label] = y.clone()
        elif mode == 2:
            d = (y.float() - ref_out[label].float()).abs().max().item()
            print(f'  [{label}] mode2 vs mode0 max_abs={d:.5f}', flush=True)

print()
hdr = f'{"case":34} {"mode0":>8} {"mode1":>8} {"mode2":>8}'
print(hdr)
for label, *_ in loaded:
    print(f'{label:34} {results[(0, label)]:8.3f} {results[(1, label)]:8.3f} '
          f'{results[(2, label)]:8.3f}')
print('(ms per m=1 call; mode1 = default heuristic)')
print('DONE')
