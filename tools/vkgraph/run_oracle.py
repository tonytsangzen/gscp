#!/usr/bin/env python3
"""run_oracle.py — decompose an ONNX model into canonical ops (BN folded into
Conv) and run a numpy oracle, comparing requested outputs against onnxruntime
(CPU) to fp32 tolerance.  The CONV executor is the im2col+GEMM that the Vulkan
runtime will mirror, so this oracle is the numerical contract for the GPU.
"""
import sys, numpy as np
import onnx
from onnx import shape_inference, numpy_helper

def attr(n, k, d=None):
    for a in n.attribute:
        if a.name == k:
            t = a.type
            if t == onnx.AttributeProto.INT: return a.i
            if t == onnx.AttributeProto.INTS: return tuple(a.ints)
            if t == onnx.AttributeProto.FLOAT: return a.f
            if t == onnx.AttributeProto.STRING: return a.s.decode('utf-8')
            if t == onnx.AttributeProto.TENSOR: return onnx.numpy_helper.to_array(a.t)
    return d

def pad4(p):
    return (p[0], p[1], p[2], p[3]) if len(p) == 4 else (p[0], p[0], p[1], p[1])

def conv_exec(x, W, b, st, pad, dl):
    x = x.astype(np.float32); W = W.astype(np.float32)
    IC, H, Wd = x.shape[1], x.shape[2], x.shape[3]
    OC, _, kh, kw = W.shape
    sh, sw = st; pt, pl, pb, pr = pad; dh, dw = dl
    oh = 1 + (H + pt + pb - (dh * (kh - 1) + 1)) // sh
    ow = 1 + (Wd + pl + pr - (dw * (kw - 1) + 1)) // sw
    cols = np.zeros((oh * ow, IC * kh * kw), dtype=np.float32)
    t = 0
    for oy in range(oh):
        for ox in range(ow):
            v = cols[t]; i = 0
            for c in range(IC):
                for ki in range(kh):
                    yy = oy * sh + ki * dh - pt
                    for kj in range(kw):
                        xx = ox * sw + kj * dw - pl
                        v[i] = x[0, c, yy, xx] if (0 <= yy < H and 0 <= xx < Wd) else 0.0
                        i += 1
            t += 1
    Wm = W.reshape(OC, IC * kh * kw)
    out = cols.astype(np.float32) @ Wm.T
    if b is not None: out += b.reshape(1, OC).astype(np.float32)
    return out.reshape(oh, ow, OC).transpose(2, 0, 1)[None].astype(np.float32)

def resize_nearest_exec(x, oh, ow):
    _, _, H, W = x.shape
    sh, sw = H / oh, W / ow
    ry = np.clip(np.floor(np.arange(oh) * sh).astype(int), 0, H - 1)
    rx = np.clip(np.floor(np.arange(ow) * sw).astype(int), 0, W - 1)
    return x[:, :, ry[:, None], rx[None, :]]

def oracle(model_path, in_name, x, wanted, all_vals=False):
    m = shape_inference.infer_shapes(onnx.load(model_path))
    g = m.graph
    initv = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    vals = {in_name: x}
    for n in g.node:
        ot = n.op_type; o = n.output[0]
        if ot == 'Identity': vals[o] = vals[n.input[0]]; continue
        if ot == 'BatchNormalization':
            gamma = initv[n.input[1]]; beta = initv[n.input[2]]
            mean = initv[n.input[3]]; var = initv[n.input[4]]
            eps = float(attr(n, 'epsilon', 1e-5))
            sc = gamma.astype(np.float32) / np.sqrt(var.astype(np.float32) + eps)
            sh = beta.astype(np.float32) - mean.astype(np.float32) * sc
            vals[o] = vals[n.input[0]].astype(np.float32) * sc.reshape(1, -1, 1, 1) + sh.reshape(1, -1, 1, 1)
            continue
        if ot == 'Conv':
            W = initv[n.input[1]]
            b = initv[n.input[2]] if len(n.input) > 2 and n.input[2] in initv else np.zeros(W.shape[0], np.float32)
            st = tuple(attr(n, 'strides', [1, 1])); pad = pad4(attr(n, 'pads', [0, 0, 0, 0])); dl = tuple(attr(n, 'dilations', [1, 1]))
            xv = vals[n.input[0]]
            vals[o] = conv_exec(xv, W, b, st, pad, dl)
        elif ot == 'Relu': vals[o] = np.maximum(vals[n.input[0]], 0)
        elif ot == 'Sigmoid': vals[o] = 1.0 / (1.0 + np.exp(-vals[n.input[0]].astype(np.float32)))
        elif ot == 'Add': vals[o] = vals[n.input[0]].astype(np.float32) + vals[n.input[1]].astype(np.float32)
        elif ot == 'Mul':
            a = vals[n.input[0]]; c = vals[n.input[1]] if n.input[1] in vals else initv[n.input[1]]
            vals[o] = a.astype(np.float32) * c.astype(np.float32)
        elif ot == 'Transpose': vals[o] = vals[n.input[0]].transpose(tuple(attr(n, 'perm')))
        elif ot == 'Reshape':
            sh = initv[n.input[1]] if len(n.input) > 1 and n.input[1] in initv else vals[n.input[1]]
            vals[o] = vals[n.input[0]].reshape(tuple(int(s) for s in sh))
        elif ot == 'MaxPool':
            k = tuple(attr(n, 'kernel_shape')); st = tuple(attr(n, 'strides', [1, 1])); pad = pad4(attr(n, 'pads', [0, 0, 0, 0])); ceil = bool(attr(n, 'ceil_mode', 0))
            xv, C, H, Wv = vals[n.input[0]], 0, 0, 0; C, H, Wv = vals[n.input[0]].shape[1], vals[n.input[0]].shape[2], vals[n.input[0]].shape[3]
            kh, kw = k; sh, sw = st; pt, pl, pb, pr = pad
            oh = ((H + pt + pb - kh + sh - 1) // sh + 1) if ceil else ((H + pt + pb - kh) // sh + 1)
            ow = ((Wv + pl + pr - kw + sw - 1) // sw + 1) if ceil else ((Wv + pl + pr - kw) // sw + 1)
            Xp = np.pad(vals[n.input[0]].astype(np.float32), ((0, 0), (0, 0), (pt, pb), (pl, pr)))
            oA = np.zeros((1, C, oh, ow), np.float32)
            for oy in range(oh):
                for ox in range(ow):
                    oA[0, :, oy, ox] = Xp[0, :, oy * sh:oy * sh + kh, ox * sw:ox * sw + kw].max((1, 2))
            vals[o] = oA
        elif ot == 'AveragePool':
            k = tuple(attr(n, 'kernel_shape')); st = tuple(attr(n, 'strides', [1, 1])); pad = pad4(attr(n, 'pads', [0, 0, 0, 0]))
            C, H, Wv = vals[n.input[0]].shape[1], vals[n.input[0]].shape[2], vals[n.input[0]].shape[3]
            kh, kw = k; sh, sw = st; pt, pl, pb, pr = pad
            oh = (H + pt + pb - kh) // sh + 1; ow = (Wv + pl + pr - kw) // sw + 1
            Xp = np.pad(vals[n.input[0]].astype(np.float32), ((0, 0), (0, 0), (pt, pb), (pl, pr)))
            oA = np.zeros((1, C, oh, ow), np.float32)
            for oy in range(oh):
                for ox in range(ow):
                    oA[0, :, oy, ox] = Xp[0, :, oy * sh:oy * sh + kh, ox * sw:ox * sw + kw].sum((1, 2)) / (kh * kw)
            vals[o] = oA
        elif ot == 'GlobalAveragePool': vals[o] = vals[n.input[0]].mean(axis=(2, 3), keepdims=True)
        elif ot == 'Flatten':
            ax = int(attr(n, 'axis', 1)); vals[o] = vals[n.input[0]].reshape(vals[n.input[0]].shape[:ax] + (-1,))
        elif ot == 'Gemm':
            B = initv[n.input[1]].astype(np.float32)
            b = initv[n.input[2]].reshape(-1) if len(n.input) > 2 and n.input[2] in initv else None
            a = vals[n.input[0]].astype(np.float32)
            if a.ndim == 1: a = a.reshape(1, -1)
            A2 = a.T if bool(attr(n, 'transA', 0)) else a
            B2 = B.T if bool(attr(n, 'transB', 0)) else B
            vals[o] = A2 @ B2
            if b is not None: vals[o] = vals[o] + b
        elif ot == 'Resize':
            xv = vals[n.input[0]]; H, Wv = xv.shape[2], xv.shape[3]
            sizes = initv[n.input[3]] if len(n.input) > 3 and n.input[3] in initv else None
            if sizes is None:
                sc = initv[n.input[2]] if len(n.input) > 2 and n.input[2] in initv else None
                oh, ow = int(H * sc[2]), int(Wv * sc[3])
            else: oh, ow = int(sizes[2]), int(sizes[3])
            vals[o] = resize_nearest_exec(xv, oh, ow)
        else:
            raise RuntimeError('unsupported op %s (%s)' % (ot, n.name))
    return vals if all_vals else vals

def main():
    import onnxruntime as ort
    import json
    model = sys.argv[1]; wanted = sys.argv[2].split(',')
    inp = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    iname = inp.get_inputs()[0].name
    # deterministic input
    rng = np.random.RandomState(1)
    shape = [int(d) if d > 0 else 1 for d in inp.get_inputs()[0].shape[2:]]
    H, W = shape[0], shape[1]
    x = rng.standard_normal((1, 3, H, W)).astype(np.float32) * 0.16
    ref = {nm: r for nm, r in zip(wanted, inp.run(wanted, {iname: x}))}
    got = oracle(model, iname, x, wanted)
    print('=== oracle vs onnxruntime; max abs diff per wanted output ===')
    okall = True
    for nm in wanted:
        a = got[nm].astype(np.float64); b = ref[nm].astype(np.float64)
        d = np.abs(a - b).max()
        rel = d / (np.abs(b).max() + 1e-9)
        print('%s shape %s maxdiff %.3e  (rel %.3e)' % (nm, b.shape, d, rel))
        if d > max(0.02, 0.02 * np.abs(b).max()): okall = False
    print('PASS' if okall else 'WARN(diff>2%% scale)')


if __name__ == '__main__':
    main()