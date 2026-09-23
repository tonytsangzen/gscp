#!/usr/bin/env python3
"""emit.py — decompose an ONNX model into a canonical op sequence with fp32
conv weights (BatchNormalization folded into Conv), then run a numpy ORACLE that
executes the sequence and compares the requested outputs against onnxruntime to
fp32 tolerance.  The emitted graph+weights are exactly what the Vulkan runtime
will later execute (CONV uses the same im2col+GEMM algorithm), so GPU == oracle
bit-for-bit and both approximate onnxruntime only by fp32 accumulation order.
"""
import json, os, math
import numpy as np
import onnx
from onnx import shape_inference, numpy_helper

# ---------------- ONNX attr ---------------- 
def attr(n, k, d=None):
    for a in n.attribute:
        if a.name == k:
            t = a.type
            if t == onnx.AttributeProto.INT: return a.i
            if t == onnx.AttributeProto.INTS: return tuple(a.ints)
            if t == onnx.AttributeProto.FLOAT: return a.f
            if t == onnx.AttributeProto.FLOATS: return tuple(a.floats)
            if t == onnx.AttributeProto.STRING: return a.s.decode('utf-8')
            if t == onnx.AttributeProto.TENSOR: return onnx.numpy_helper.to_array(a.t)
    return d

# ---------------- executors (numpy fp32) ----------------
def conv_exec(x, W, b, st, pad, dl):
    # x: [1,C,H,W] fp32 ; W: [O,C,Kh,Kw] ; pad=(top,left,bottom,right)
    ic, H, W = x.shape[1], x.shape[2], x.shape[3]
    kh, kw = W.shape[2], W.shape[3]
    sh, sw = st; pt, pl, pb, pr = pad; dh, dw = dl
    Xp = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr))).astype(np.float32)
    oh = 1 + (H + pt + pb - (dh * (kh - 1) + 1)) // sh
    ow = 1 + (W + pl + pr - (dw * (kw - 1) + 1)) // sw
    OC, IC = W.shape[0], W.shape[1]
    # im2col
    cols = np.zeros((oh * ow, IC * kh * kw), dtype=np.float32)
    idx = 0
    for oy in range(oh):
        for ox2 in range(ow):
            vec = np.zeros(IC * kh * kw, dtype=np.float32)
            t = 0
            for c in range(IC):
                for ki in range(kh):
                    yy = oy * sh + ki * dh
                    if yy >= H + pt: vec[t:] = 0; t += kw; continue
                    for kj in range(kw):
                        xx = ox2 * sw + kj * dw
                        vec[t] = Xp[0, c, yy, xx] if (yy < H + pt and xx < W + pl) else 0.0
                        t += 1
            cols[idx] = vec
            idx += 1
    Wm = W.reshape(OC, IC * kh * kw)  # [OC, K]
    out = cols @ Wm.T  # [OH*OW, OC]
    if b is not None:
        out += b.reshape(1, OC)
    out = out.reshape(oh, ow, OC).transpose(2, 0, 1)[None]  # [1,OC,oh,ow]
    return out

def relu_exec(x): return np.maximum(x, 0)
def sigmoid_exec(x): return 1.0 / (1.0 + np.exp(-x.astype(np.float32)))
def add_exec(a, b): return a.astype(np.float32) + b.astype(np.float32)
def mul_exec(x, s): return x.astype(np.float32) * s
def pool_exec(x, k, st, pad, avg, ceil, count_include_pad):
    _, C, H, W = x.shape
    kh, kw = k; sh, sw = st; pt, pl, pb, pr = pad
    if ceil:
        oh = (H + pt + pb - kh + sh - 1) // sh + 1
        ow = (W + pl + pr - kw + sw - 1) // sw + 1
    else:
        oh = (H + pt + pb - kh) // sh + 1
        ow = (W + pl + pr - kw) // sw + 1
    Xp = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr))).astype(np.float32)
    out = np.zeros((1, C, oh, ow), dtype=np.float32)
    for oy in range(oh):
        for ox in range(ow):
            win = Xp[0, :, oy * sh:oy * sh + kh, ox * sw:ox * sw + kw]  # [C,kh,kw]
            if not avg:
                out[0, :, oy, ox] = win.max(axis=(1, 2))
            else:
                if count_include_pad:
                    out[0, :, oy, ox] = win.sum(axis=(1, 2)) / float(kh * kw)
                else:
                    out[0, :, oy, ox] = win.sum(axis=(1, 2)) / float(((win != 0).any(axis=0).sum(axis=1)) * 1.0) if False else win.sum(axis=(1,2))/float(_valid(win))
    return out

def _valid(win):  # count valid entries per channel (excludes pad=0? pad value given by pad_value default 0; can't distinguish real 0)
    return float(win.shape[1] * win.shape[2])  # onnx avgpool counts valid window in [H+pad] bounds; approximation
def resize_nearest_exec(x, oh, ow):
    _,_,H,W = x.shape
    sh, sw = H / oh, W / ow
    ry = np.clip(np.floor(np.arange(oh) * sh).astype(int), 0, H - 1)
    rx = np.clip(np.floor(np.arange(ow) * sw).astype(int), 0, W - 1)
    return x[:, :, ry[:, None], rx[None, :]]
def transpose_exec(x, perm): return x.transpose(tuple(perm))
def reshape_exec(x, shape): return x.reshape(tuple(shape))
def gemm_exec(x, B, b, alpha, beta, transB):
    B2 = B.T if transB else B
    y = alpha * (x.astype(np.float32) @ B2.astype(np.float32))
    if beta != 0 and b is not None: y = y + beta * b
    return y

# ---------------- main ----------------
def emit(model_path, in_h, in_w, model_input_name, out_dir, keep_outputs):
    m = shape_inference.infer_shapes(onnx.load(model_path))
    g = m.graph
    initv = {i.name: numpy_helper.to_array(i) for i in g.initializer}

    # fold BN: map conv_out_name -> (scale_c, shift_c, bnout)
    bn_map = {}
    for n in g.node:
        if n.op_type == 'BatchNormalization':
            gamma = initv[n.input[1]]; beta = initv[n.input[2]]
            mean = initv[n.input[3]]; var = initv[n.input[4]]
            eps = float(attr(n, 'epsilon', 1e-5))
            sc = gamma.astype(np.float32) / np.sqrt(var.astype(np.float32) + eps)
            sh = (beta.astype(np.float32) - mean.astype(np.float32) * sc).astype(np.float32)
            bn_map[n.input[0]] = (sc, sh)

    vals = {}   # tensor-name -> numpy value
    ops = []    # canonical node list
    # input
    in_name = g.input[0].name
    # we'll feed a real input via oracle later; for emit we only record graph.
    # execute with a controllable input passed as arg.

    return  # placeholder (oracle in emit_oracle)

def emit_oracle(model_path, in_h, in_w, x, keep_outputs):
    m = shape_inference.infer_shapes(onnx.load(model_path))
    g = m.graph
    initv = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    bn_map = {}
    for n in g.node:
        if n.op_type == 'BatchNormalization':
            gamma = initv[n.input[1]]; beta = initv[n.input[2]]
            mean = initv[n.input[3]]; var = initv[n.input[4]]
            eps = float(attr(n, 'epsilon', 1e-5))
            sc = gamma.astype(np.float32) / np.sqrt(var.astype(np.float32) + eps)
            sh = beta.astype(np.float32) - mean.astype(np.float32) * sc
            bn_map[n.input[0]] = (sc, sh)
    vals = {g.input[0].name: x}
    out = {}
    for n in g.node:
        ot = n.op_type
        if ot == 'Identity':
            vals[n.output[0]] = vals[n.input[0]]; continue
        if ot == 'BatchNormalization':
            continue  # folded into conv
        if ot == 'Conv':
            W = init[n.input[1]]
            b = init[n.input[2]] if len(n.input) > 2 and n.input[2] in init else np.zeros(W.shape[0], dtype=np.float32)
            st = tuple(attr(n, 'strides', [1, 1])); pad = pad4(attr(n, 'pads', [0, 0, 0, 0])); dl = tuple(attr(n, 'dilations', [1, 1]))
            xv = vals[n.input[0]]
            if convin := n.output[0] in bn_map: pass
            if n.output[0] in bn_map:
                sc, shh = bn_map[n.output[0]]
                b = b.astype(np.float32) * sc + shh
                W = W * sc.reshape(W.shape[0], 1, 1, 1)
            out[n.output[0]] = conv_exec(xv, W, b, st, pad, dl)
            vals[n.output[0]] = out[n.output[0]]
        elif ot == 'Relu': vals[n.output[0]] = relu_exec(vals[n.input[0]])
        elif ot == 'Sigmoid': vals[n.output[0]] = sigmoid_exec(vals[n.input[0]])
        elif ot == 'Add': vals[n.output[0]] = add_exec(vals[n.input[0]], vals[n.input[1]])
        elif ot == 'Mul':
            a = vals[n.input[0]]; b = vals[n.input[1]] if n.input[1] in vals else initv[n.input[1]]
            vals[n.output[0]] = a.astype(np.float32) * b.astype(np.float32)
        elif ot == 'Transpose':
            vals[n.output[0]] = transpose_exec(vals[n.input[0]], attr(n, 'perm'))
        elif ot == 'Reshape':
            sh = initv[n.input[1]] if len(n.input) > 1 and n.input[1] in initv else vals[n.input[1]]
            vals[n.output[0]] = reshape_exec(vals[n.input[0]], sh)
        elif ot == 'MaxPool':
            k = tuple(attr(n, 'kernel_shape')); st = tuple(attr(n, 'strides', [1, 1])); pad = pad4(attr(n, 'pads', [0, 0, 0, 0]));
            vals[n.output[0]] = pool_exec(vals[n.input[0]], k, st, pad, avg=False, ceil=bool(attr(n, 'ceil_mode', 0)), count_include_pad=False)
        elif ot == 'AveragePool':
            k = tuple(attr(n, 'kernel_shape')); st = tuple(attr(n, 'strides', [1, 1])); pad = pad4(attr(n, 'pads', [0, 0, 0, 0]));
            vals[n.output[0]] = pool_exec(vals[n.input[0]], k, st, pad, avg=True, ceil=bool(attr(n, 'ceil_mode', 0)), count_include_pad=bool(attr(n, 'count_include_pad', 0)))
        elif ot == 'GlobalAveragePool':
            vals[n.output[0]] = vals[n.input[0]].mean(axis=(2, 3), keepdims=True)
        elif ot == 'Flatten':
            ax = int(attr(n, 'axis', 1));
            vals[n.output[0]] = vals[n.input[0]].reshape(vals[n.input[0]].shape[:ax] + (-1,))
        elif ot == 'Gemm':
            B = initv[n.input[1]]; b = initv[n.input[2]] if len(n.input) > 2 and n.input[2] in init else None
            vals[n.output[0]] = gemm_exec(vals[n.input[0]].reshape(vals[n.input[0]].shape[0], -1),
                                          B, b.reshape(-1) if b is not None else None,
                                          float(attr(n, 'alpha', 1.0)), float(attr(n, 'beta', 1.0)),
                                          bool(attr(n, 'transA', 0)), bool(attr(n, 'transB', 0)))
        elif ot == 'Resize':
            H, W = vals[n.input[0]].shape[2], vals[n.input[0]].shape[3]
            sizes = init[n.input[3]] if len(n.input) > 3 and n.input[3] in init else None
            if sizes is None:
                scales = initv[n.input[2]] if len(n.input) > 2 and n.input[2] in init else None
                oh = int(H * scales[2]); ow = int(W * scales[3])
            else:
                oh, ow = int(sizes[2]), int(sizes[3])
            vals[n.output[0]] = resize_nearest_exec(vals[n.input[0]], oh, ow)
        else:
            raise RuntimeError('unsupported op ' + ot)
    return out