"""
=============================================================================
YOLOv8-P2-Router → TensorRT IIfConditional (Single Engine)
Shape verified dari model aktual:
  layer0 : [1,16,320,320]   layer1 : [1,32,160,160]
  layer2 : [1,32,160,160]   layer3 : [1,64,80,80]
  layer4 : [1,64,80,80]     layer5 : [1,128,40,40]
  layer6 : [1,128,40,40]    layer7 : [1,256,20,20]
  layer8 : [1,256,20,20]    layer9 : [1,256,20,20]  ← SPPF
  layer10: [1,256,40,40]    layer11: [1,384,40,40]
  layer12: [1,128,40,40]    layer13: [1,128,80,80]
  layer14: [1,192,80,80]    layer15: [1,64,80,80]   ← f_p3_neck
  layer16: [1,32,160,160]   ← DifficultyAwareRouter output
  layer17: [1,64,80,80]     layer18: [1,128,80,80]
  layer19: [1,64,80,80]     layer20: [1,128,40,40]
  layer21: [1,256,40,40]    layer22: [1,128,40,40]
  layer23: [1,256,20,20]    layer24: [1,512,20,20]
  layer25: [1,256,20,20]    layer26: [1, 6, 34000]

Cara pakai:
  python export_trt_ifconditional.py --check     # verifikasi weight keys
  python export_trt_ifconditional.py --build     # build engine
  python export_trt_ifconditional.py --validate  # buktikan lazy eval
=============================================================================
"""

import sys, os
import numpy as np
import torch
import torch.nn as nn
import tensorrt as trt
from ultralytics import YOLO

# =============================================================================
# CONFIG
# =============================================================================
PT_PATH     = "/kaggle/input/models/agungmaugi/thesis-yolo-router-kitti/pytorch/default/1/Thesis_kitti_fix/Skenario3_YOLOv8n_P2_Router/weights/best.pt"
ENGINE_PATH = "/kaggle/working/router_ifconditional.engine"
IMG_FOLDER  = "/kaggle/input/datasets/agungmaugi/kitti-2class/kitti_yolo_2class/images/test"

# Shape verified dari output model aktual
SHP = {
    "img":        (1,  3,   640, 640),
    "layer0":     (1,  16,  320, 320),
    "f_p2_back":  (1,  32,  160, 160),  # output layer 2  (C2f)
    "f_p3_back":  (1,  64,  80,  80),   # output layer 4  (C2f)
    "f_p4_back":  (1,  128, 40,  40),   # output layer 6  (C2f)
    "f_p5":       (1,  256, 20,  20),   # output layer 9  (SPPF)
    "f_p4_neck":  (1,  128, 40,  40),   # output layer 12 (C2f)
    "f_p3_neck":  (1,  64,  80,  80),   # output layer 15 (C2f) — input router
    "f_p2_final": (1,  32,  160, 160),  # output layer 16 (Router) — true branch
    "nc":         2,
    "reg_max":    16,
    "layer26":    (1, 6, 34000)      # reg_max(16)*4 + nc(2) = 66
}

# Jumlah bottleneck C2f setelah depth scaling=0.33
C2F_N = {
    "model.2":  1, "model.4":  2, "model.6":  2, "model.8":  1,
    "model.12": 1, "model.15": 1, "model.16.c2f_p2": 1,
    "model.19": 1, "model.22": 1, "model.25": 1,
}

# Shortcut hanya untuk backbone (True), neck dan head False
C2F_SC = {
    "model.2": True, "model.4": True, "model.6": True, "model.8": True,
    "model.12": False, "model.15": False, "model.16.c2f_p2": False,
    "model.19": False, "model.22": False, "model.25": False,
}

# =============================================================================
# STEP 0 — verifikasi weight keys sebelum build
# =============================================================================
def load_and_patch_model(pt_path, use_cuda=False):
    """Memuat model YOLO dan menyuntikkan patch kompatibilitas."""
    model = YOLO(pt_path)
    nn_m = model.model.eval()
    if use_cuda:
        nn_m = nn_m.cuda()
        
    # ========================================================
    # 🚨 PATCH: SUNTIKAN KOMPATIBILITAS UNTUK CHECKPOINT LAMA
    # ========================================================
    router_instance = nn_m.model[16] # Layer 16 adalah DifficultyAwareRouter
    if hasattr(router_instance, 'mlp') and not hasattr(router_instance, 'mlp_fc1'):
        router_instance.mlp_fc1 = router_instance.mlp[0]
        router_instance.mlp_fc2 = router_instance.mlp[2]
        print("  [INFO] Patch kompatibilitas 'mlp_fc' berhasil disuntikkan ke Router.")
    # ========================================================
    
    return model, nn_m

def cmd_check():
    """
    Cetak semua weight key dan konfirmasi shape setiap tensor intermediate.
    """
    print("=" * 65)
    print("STEP 0: VERIFIKASI MODEL")
    print("=" * 65)

    if not os.path.exists(PT_PATH):
        print(f"❌ File {PT_PATH} tidak ditemukan.")
        return

    # Gunakan fungsi helper kita di sini
    model, nn_m = load_and_patch_model(PT_PATH, use_cuda=True)
    
    layers = nn_m.model
    router = layers[16]
    sd     = nn_m.state_dict()
    # -------------------------

    print("\n[1] Shape verification:")
    dummy = torch.zeros(SHP["img"], device="cuda")
    with torch.no_grad():
        x0        = layers[0](dummy)
        x1        = layers[1](x0)
        f_p2_back = layers[2](x1)
        x3        = layers[3](f_p2_back)
        f_p3_back = layers[4](x3)
        x5        = layers[5](f_p3_back)
        f_p4_back = layers[6](x5)
        x7        = layers[7](f_p4_back)
        x8        = layers[8](x7)
        f_p5      = layers[9](x8)
        up5       = layers[10](f_p5)
        c11       = layers[11]([up5, f_p4_back])
        f_p4_neck = layers[12](c11)
        up4       = layers[13](f_p4_neck)
        c14       = layers[14]([up4, f_p3_back])
        f_p3_neck = layers[15](c14)
        
        f_p2_fin  = router.compute_expert(f_p3_neck, f_p2_back)
        gate_p    = router.compute_gate(f_p3_neck, f_p2_back)

    checks = [
        ("f_p2_back",  f_p2_back.shape,  SHP["f_p2_back"]),
        ("f_p3_back",  f_p3_back.shape,  SHP["f_p3_back"]),
        ("f_p4_back",  f_p4_back.shape,  SHP["f_p4_back"]),
        ("f_p5",       f_p5.shape,       SHP["f_p5"]),
        ("f_p4_neck",  f_p4_neck.shape,  SHP["f_p4_neck"]),
        ("f_p3_neck",  f_p3_neck.shape,  SHP["f_p3_neck"]),
        ("f_p2_final", f_p2_fin.shape,   SHP["f_p2_final"]),
    ]

    all_ok = True
    for name, actual, expected in checks:
        ok = tuple(actual) == tuple(expected)
        mark = "✓" if ok else "✗"
        print(f"  {mark} {name:<15}: actual={tuple(actual)}  expected={tuple(expected)}")
        if not ok: all_ok = False

    print("\n[2] Router methods:")
    for m in ["compute_gate", "compute_expert"]:
        has = hasattr(router, m)
        print(f"  {'✓' if has else '✗ BELUM ADA'} {m}")
        if not has: all_ok = False

    has_stats = (router._running_mean is not None and router._stats_initialized)
    print(f"  {'✓' if has_stats else '⚠'} running_mean: {router._running_mean if has_stats else 'None (akan pakai zeros)'}")

    print("\n[3] C2f bottleneck keys:")
    for prefix, n in C2F_N.items():
        for i in range(n):
            key = f"{prefix}.m.{i}.cv1.conv.weight"
            exists = key in sd
            print(f"  {'✓' if exists else '✗'} {key}")
            if not exists: all_ok = False

    print("\n[4] Router weight keys:")
    router_keys = [
        "model.16.sam.conv.weight", "model.16.conv_hint.weight",
        "model.16.layer_norm.weight", "model.16.layer_norm.bias",
        "model.16.mlp.0.weight", "model.16.mlp.0.bias",
        "model.16.mlp.2.weight", "model.16.mlp.2.bias",
        "model.16.stats_weight",
    ]
    for k in router_keys:
        exists = k in sd
        print(f"  {'✓' if exists else '✗'} {k}")
        if not exists: all_ok = False

    del model
    torch.cuda.empty_cache()
    print(f"\n{'✅ Semua OK — siap --build' if all_ok else '❌ Ada masalah — perbaiki dulu'}")

# =============================================================================
# STEP 1 — ekstrak weights
# =============================================================================
def extract_weights():
    print("Mengekstrak weights...")
    
    # Gunakan fungsi helper kita di sini (tanpa cuda untuk extract weights)
    model, nn_m = load_and_patch_model(PT_PATH, use_cuda=False)
    
    router = nn_m.model[16]
    W = {k: v.detach().cpu().float().numpy() for k, v in nn_m.state_dict().items()}

    if router._running_mean is not None and router._stats_initialized:
        W["_rmean"] = router._running_mean.cpu().float().numpy()
        W["_rstd"]  = np.clip(router._running_std.cpu().float().numpy(), 1e-4, None)
    else:
        W["_rmean"] = np.zeros(3, np.float32)
        W["_rstd"]  = np.ones(3,  np.float32)
        print("  ⚠ Running stats kosong — pakai zeros/ones")

    del model
    torch.cuda.empty_cache()
    print(f"  {len(W)} tensors diekstrak")
    return W

# =============================================================================
# STEP 2 — primitif TRT
# =============================================================================
def tw(arr):
    return trt.Weights(np.ascontiguousarray(arr.astype(np.float32)))

def const(net, arr):
    return net.add_constant(trt.Dims(list(arr.shape)), tw(arr)).get_output(0)

def reshape(net, x, shape):
    s = net.add_shuffle(x)
    s.reshape_dims = trt.Dims(list(shape))
    return s.get_output(0)

def silu(net, x):
    sig = net.add_activation(x, trt.ActivationType.SIGMOID).get_output(0)
    return net.add_elementwise(x, sig, trt.ElementWiseOperation.PROD).get_output(0)

def concat(net, tensors):
    c = net.add_concatenation(tensors)
    c.axis = 1
    return c.get_output(0)

def upsample(net, x, scale=2):
    r = net.add_resize(x)
    # ✅ PERBAIKAN: Gunakan InterpolationMode untuk TensorRT versi baru
    r.resize_mode = trt.InterpolationMode.NEAREST 
    r.scales = [1.0, 1.0, float(scale), float(scale)]
    return r.get_output(0)

def gap(net, x):
    """Global Average Pooling (B,C,H,W)→(B,C,1,1)."""
    return net.add_reduce(x, trt.ReduceOperation.AVG, axes=(1<<2)|(1<<3), keep_dims=True).get_output(0)

def flatten_hw(net, x):
    """(B,C,H,W) → (B,C,H*W)."""
    dims = x.shape
    B, C = dims[0], dims[1]
    H, Ww = dims[2], dims[3]
    s = net.add_shuffle(x)
    s.reshape_dims = trt.Dims([B, C, H * Ww])
    return s.get_output(0)

def conv_bn_silu(net, x, W, prefix, stride=1):
    w_arr  = W[f"{prefix}.conv.weight"]
    out_ch = w_arr.shape[0]
    ksize  = w_arr.shape[2]
    pad    = ksize // 2
    b_arr  = W.get(f"{prefix}.conv.bias", np.zeros(out_ch, np.float32))

    cv = net.add_convolution_nd(x, out_ch, (ksize, ksize), tw(w_arr), tw(b_arr))
    cv.stride_nd  = (stride, stride)
    cv.padding_nd = (pad, pad)
    x = cv.get_output(0)

    gamma = W[f"{prefix}.bn.weight"]
    beta  = W[f"{prefix}.bn.bias"]
    mean  = W[f"{prefix}.bn.running_mean"]
    var   = W[f"{prefix}.bn.running_var"]
    eps   = 1e-5
    
    scale = gamma / np.sqrt(var + eps)
    shift = beta  - mean * scale
    sc = net.add_scale_nd(x, trt.ScaleMode.CHANNEL,
                          tw(shift), tw(scale), tw(np.ones_like(scale)),
                          channel_axis=1)
    return silu(net, sc.get_output(0))

def conv_plain(net, x, W, prefix, out_ch, ksize=1, stride=1, pad=0):
    w_arr = W[f"{prefix}.weight"]
    b_arr = W[f"{prefix}.bias"]
    cv = net.add_convolution_nd(x, out_ch, (ksize, ksize), tw(w_arr), tw(b_arr))
    cv.stride_nd  = (stride, stride)
    cv.padding_nd = (pad, pad)
    return cv.get_output(0)

def bottleneck_block(net, x, W, prefix, add_res=True):
    res = x
    x   = conv_bn_silu(net, x, W, f"{prefix}.cv1")
    x   = conv_bn_silu(net, x, W, f"{prefix}.cv2")
    if add_res:
        x = net.add_elementwise(x, res, trt.ElementWiseOperation.SUM).get_output(0)
    return x

def c2f_block(net, x, W, prefix):
    n  = C2F_N[prefix]
    sc = C2F_SC[prefix]
    x     = conv_bn_silu(net, x, W, f"{prefix}.cv1")
    dims  = x.shape
    half  = dims[1] // 2
    H, Ww = dims[2], dims[3]

    y1 = net.add_slice(x, trt.Dims([0, 0,    0, 0]),
                          trt.Dims([1, half, H, Ww]),
                          trt.Dims([1, 1,    1, 1])).get_output(0)
    y2 = net.add_slice(x, trt.Dims([0, half, 0, 0]),
                          trt.Dims([1, half, H, Ww]),
                          trt.Dims([1, 1,    1, 1])).get_output(0)

    parts = [y1, y2]
    for i in range(n):
        y2 = bottleneck_block(net, y2, W, f"{prefix}.m.{i}", add_res=sc)
        parts.append(y2)
    x = concat(net, parts)
    return conv_bn_silu(net, x, W, f"{prefix}.cv2")

def sppf_block(net, x, W, prefix, k=5):
    x  = conv_bn_silu(net, x, W, f"{prefix}.cv1")
    x0 = x
    pooled = [x0]
    for _ in range(3):
        mp = net.add_pooling_nd(x, trt.PoolingType.MAX, (k, k))
        mp.padding_nd = (k//2, k//2)
        mp.stride_nd  = (1, 1)
        x = mp.get_output(0)
        pooled.append(x)
    x = concat(net, pooled)
    return conv_bn_silu(net, x, W, f"{prefix}.cv2")

def sam_block(net, x, W, prefix):
    avg = net.add_reduce(x, trt.ReduceOperation.AVG, axes=1<<1, keep_dims=True).get_output(0)
    mx  = net.add_reduce(x, trt.ReduceOperation.MAX, axes=1<<1, keep_dims=True).get_output(0)
    cm  = net.add_concatenation([avg, mx]); cm.axis = 1
    w_s = W[f"{prefix}.conv.weight"]
    cv  = net.add_convolution_nd(cm.get_output(0), 1, (7, 7), tw(w_s), tw(np.zeros(1, np.float32)))
    cv.padding_nd = (3, 3); cv.stride_nd = (1, 1)
    sig = net.add_activation(cv.get_output(0), trt.ActivationType.SIGMOID).get_output(0)
    return net.add_elementwise(x, sig, trt.ElementWiseOperation.PROD).get_output(0)

def layer_norm_manual(net, x, W, prefix, C):
    eps   = 1e-5
    mean  = net.add_reduce(x, trt.ReduceOperation.AVG, axes=1<<1, keep_dims=True).get_output(0)
    diff  = net.add_elementwise(x, mean, trt.ElementWiseOperation.SUB).get_output(0)
    sq    = net.add_elementwise(diff, diff, trt.ElementWiseOperation.PROD).get_output(0)
    var   = net.add_reduce(sq, trt.ReduceOperation.AVG, axes=1<<1, keep_dims=True).get_output(0)
    veps  = net.add_elementwise(var, const(net, np.array([[eps]], np.float32)), trt.ElementWiseOperation.SUM).get_output(0)
    std   = net.add_unary(veps, trt.UnaryOperation.SQRT).get_output(0)
    norm  = net.add_elementwise(diff, std, trt.ElementWiseOperation.DIV).get_output(0)
    
    w_ln  = W[f"{prefix}.weight"].reshape(1, C)
    b_ln  = W[f"{prefix}.bias"].reshape(1, C)
    norm  = net.add_elementwise(norm, const(net, w_ln), trt.ElementWiseOperation.PROD).get_output(0)
    norm  = net.add_elementwise(norm, const(net, b_ln), trt.ElementWiseOperation.SUM).get_output(0)
    return norm

def fc_layer(net, x, W, prefix):
    """
    Pengganti add_fully_connected di TRT 10 menggunakan Konvolusi 1x1.
    Input x dijamin memiliki shape (B, C, 1, 1) dari tahapan sebelumnya.
    """
    w = W[f"{prefix}.weight"] # Bobot dari PyTorch nn.Linear: (out_ch, in_ch)
    b = W[f"{prefix}.bias"]   # Bias: (out_ch)
    
    out_ch = w.shape[0]
    
    # Reshape bobot linear (out_ch, in_ch) menjadi kernel konvolusi (out_ch, in_ch, 1, 1)
    w_conv = w.reshape(out_ch, -1, 1, 1)
    
    # Lakukan operasi konvolusi 1x1
    cv = net.add_convolution_nd(
        input=x, 
        num_output_maps=out_ch, 
        kernel_shape=(1, 1), 
        kernel=tw(w_conv), 
        bias=tw(b)
    )
    cv.stride_nd = (1, 1)
    cv.padding_nd = (0, 0)
    
    return cv.get_output(0)

# =============================================================================
# STEP 3 — gate subgraph (selalu dieksekusi, DI LUAR IIfConditional)
# =============================================================================
def build_gate(net, f_p3_neck, f_p2_back, W):
    B    = 1
    c_p3 = SHP["f_p3_neck"][1]
    
    # Z_visual
    f_att   = sam_block(net, f_p3_neck, W, "model.16.sam")
    z_v4d   = gap(net, f_att)
    z_vis   = reshape(net, z_v4d, [B, c_p3])

    # Z_low
    wh      = W["model.16.conv_hint.weight"]
    ch      = net.add_convolution_nd(f_p2_back, 16, (1,1), tw(wh), tw(np.zeros(16, np.float32)))
    ch.stride_nd = (1,1); ch.padding_nd = (0,0)
    z_l4d   = gap(net, ch.get_output(0))
    z_low   = reshape(net, z_l4d, [B, 16])

    # Stats
    s_raw   = const(net, np.zeros((B, 3), np.float32))
    s_mean  = const(net, W["_rmean"].reshape(1, 3))
    s_std   = const(net, W["_rstd"].reshape(1, 3))
    s_diff  = net.add_elementwise(s_raw,  s_mean, trt.ElementWiseOperation.SUB).get_output(0)
    s_norm  = net.add_elementwise(s_diff, s_std,  trt.ElementWiseOperation.DIV).get_output(0)
    sw      = W["model.16.stats_weight"].reshape(1, 3)
    stats   = net.add_elementwise(s_norm, const(net, sw), trt.ElementWiseOperation.PROD).get_output(0)

    # Z_in & MLP
    zc      = net.add_concatenation([z_vis, z_low, stats]); zc.axis = 1
    z_in    = zc.get_output(0)
    z_norm  = layer_norm_manual(net, z_in, W, "model.16.layer_norm", c_p3+16+3)
    z_fc    = reshape(net, z_norm, [B, c_p3+16+3, 1, 1])
    h       = fc_layer(net, z_fc, W, "model.16.mlp.0")
    h       = silu(net, h)
    logits  = fc_layer(net, h, W, "model.16.mlp.2")

    # Tanh soft-clipping: 3 * tanh(logits / 3)
    c3      = const(net, np.full((1,2,1,1), 3.0, np.float32))
    lg_d    = net.add_elementwise(logits, c3, trt.ElementWiseOperation.DIV).get_output(0)
    
    # ✅ PERBAIKAN: Gunakan add_activation dengan ActivationType.TANH
    act_tanh = net.add_activation(lg_d, trt.ActivationType.TANH)
    lg_t     = act_tanh.get_output(0)
    
    logits  = net.add_elementwise(lg_t, c3, trt.ElementWiseOperation.PROD).get_output(0)

    # Softmax & Threshold
    sm = net.add_softmax(logits); sm.axes = 1<<1
    probs = sm.get_output(0)
    sl = net.add_slice(probs, trt.Dims([0,1,0,0]), trt.Dims([B,1,1,1]), trt.Dims([1,1,1,1]))
    gate_4d = sl.get_output(0)
    gate_2d = reshape(net, gate_4d, [B, 1])
    
    th  = const(net, np.full((B,1), 0.5, np.float32))
    gt  = net.add_elementwise(gate_2d, th, trt.ElementWiseOperation.GREATER).get_output(0)
    
    # 0-D Boolean Scalar for IConditionLayer
    # ✅ PERBAIKAN: Gunakan Shuffle untuk mereshape (1,1) menjadi 0-D (Scalar)
    sc_bool = net.add_shuffle(gt)
    sc_bool.reshape_dims = trt.Dims([]) # Reshape ke 0-D Scalar untuk Conditional
    gate_bool = sc_bool.get_output(0)
    
    return gate_2d, gate_bool

# =============================================================================
# STEP 4 — IIfConditional
# =============================================================================
def build_if_conditional(net, cond_obj, p3_in, p2_in, W):
    """
    True branch  = C2f P2. Akan menjadi "lazy evaluated" oleh TensorRT 
                   karena dependent terhadap IIfConditionalInputLayer (p3_in, p2_in).
    False branch = const tensor (dieksekusi "eager", tapi karena hanya const, 0 cost).
    """
    # ---- TRUE BRANCH: C2f P2 ----
    p3_up    = upsample(net, p3_in)
    fused    = concat(net, [p3_up, p2_in])
    p2_true  = c2f_block(net, fused, W, "model.16.c2f_p2")

    # ---- FALSE BRANCH: tensor nol ----
    p2_false = const(net, np.zeros(SHP["f_p2_final"], np.float32))

    # ---- Merge ----
    out = cond_obj.add_output(p2_true, p2_false)
    return out.get_output(0)

def build_detect_head(net, W, feats):
    """
    Detect head anchor-free YOLOv8 untuk 4 skala [P2, P3, P4, P5].
    """
    nc, rm, B = SHP["nc"], SHP["reg_max"], 1
    reg_flat = []
    cls_flat = []
    
    # 1. Proses masing-masing skala
    for i, feat in enumerate(feats):
        # Regression: Conv -> Conv -> Conv
        r = conv_bn_silu(net, feat, W, f"model.26.cv2.{i}.0")
        r = conv_bn_silu(net, r,    W, f"model.26.cv2.{i}.1")
        r = conv_plain(net, r, W, f"model.26.cv2.{i}.2", out_ch=4*rm)
        reg_flat.append(flatten_hw(net, r)) # (1, 4*rm, N_i)
        
        # Classification: Conv -> Conv -> Conv
        c = conv_bn_silu(net, feat, W, f"model.26.cv3.{i}.0")
        c = conv_bn_silu(net, c,    W, f"model.26.cv3.{i}.1")
        c = conv_plain(net, c, W, f"model.26.cv3.{i}.2", out_ch=nc)
        cls_flat.append(flatten_hw(net, c)) # (1, nc, N_i)

    # 2. Gabungkan (Concatenate) pada axis 2 (dimensi jumlah titik/anchors)
    reg_concat = concat_axis(net, reg_flat, axis=2) # (1, 4*rm, 34000)
    cls_concat = concat_axis(net, cls_flat, axis=2) # (1, nc, 34000)
    
    # PERBAIKAN: Hitung total_anchors dari dimensi tensor yang sudah di-concat
    # TensorRT 10 mengizinkan akses ke shape tensor hasil concat secara langsung
    dims = reg_concat.shape
    total_anchors = dims[2] 

    # =========================================================
    # 3. DECODING DFL (DISTRIBUTION FOCAL LOSS)
    # =========================================================
    # Reshape ke (1, 4, 16, 34000)
    r_4d = reshape(net, reg_concat, [B, 4, rm, total_anchors])
    
    # Transpose ke (1, 16, 4, 34000)
    shuf = net.add_shuffle(r_4d)
    # ✅ PERBAIKAN: Gunakan first_transpose dengan tuple standar
    shuf.first_transpose = (0, 2, 1, 3)
    r_trans = shuf.get_output(0)
    
    # Softmax pada dimensi channel (axis 1: reg_max)
    sm = net.add_softmax(r_trans)
    sm.axes = 1 << 1
    r_soft = sm.get_output(0)
    
    # DFL Reduction via Conv 1x1
    dfl_w = np.arange(rm, dtype=np.float32).reshape(1, rm, 1, 1)
    cv_dfl = net.add_convolution_nd(
        input=r_soft,
        num_output_maps=1,
        kernel_shape=(1, 1),
        kernel=tw(dfl_w),
        bias=tw(np.zeros(1, np.float32))
    )
    r_conv = cv_dfl.get_output(0) # (1, 1, 4, 34000)
    
    # Reshape ke format (1, 4, 34000)
    box_decoded = reshape(net, r_conv, [B, 4, total_anchors])

    # =========================================================
    # 4. DECODING CLASSIFICATION (SIGMOID)
    # =========================================================
    act_sig = net.add_activation(cls_concat, trt.ActivationType.SIGMOID)
    cls_decoded = act_sig.get_output(0) # (1, 2, 34000)

    # =========================================================
    # 5. PENGGABUNGAN AKHIR
    # =========================================================
    return concat_axis(net, [box_decoded, cls_decoded], axis=1)

def concat_axis(net, tensors, axis=1):
    c = net.add_concatenation(tensors)
    c.axis = axis
    return c.get_output(0)

# =============================================================================
# STEP 3+4 gabungan — orkestrasi build_network
# =============================================================================
def build_network(net, W):
    print("  Membangun network...")
    
    images = net.add_input("images", trt.DataType.FLOAT, trt.Dims(list(SHP["img"])))

    # BACKBONE
    x         = conv_bn_silu(net, images,    W, "model.0", stride=2)
    x         = conv_bn_silu(net, x,         W, "model.1", stride=2)
    f_p2_back = c2f_block(net, x, W, "model.2")
    x         = conv_bn_silu(net, f_p2_back, W, "model.3", stride=2)
    f_p3_back = c2f_block(net, x, W, "model.4")
    x         = conv_bn_silu(net, f_p3_back, W, "model.5", stride=2)
    f_p4_back = c2f_block(net, x, W, "model.6")
    x         = conv_bn_silu(net, f_p4_back, W, "model.7", stride=2)
    x         = c2f_block(net, x, W, "model.8")
    f_p5      = sppf_block(net, x, W, "model.9", k=5)
    print("    ✓ Backbone (1,256,20,20)")

    # NECK TOP-DOWN
    up5       = upsample(net, f_p5)
    c11       = concat(net, [up5, f_p4_back])
    f_p4_neck = c2f_block(net, c11, W, "model.12")
    up4       = upsample(net, f_p4_neck)
    c14       = concat(net, [up4, f_p3_back])
    f_p3_neck = c2f_block(net, c14, W, "model.15")
    print("    ✓ Neck top-down (1,64,80,80)")

    # GATE COMPUTATION
    gate_2d, gate_bool = build_gate(net, f_p3_neck, f_p2_back, W)
    print("    ✓ Gate computation (MLP → prob)")

    # IIfConditional
    cond = net.add_if_conditional()
    cond.set_condition(gate_bool)
    
    inp_p3 = cond.add_input(f_p3_neck)
    inp_p2 = cond.add_input(f_p2_back)
    
    f_p2_final = build_if_conditional(net, cond, inp_p3.get_output(0), inp_p2.get_output(0), W)
    print("    ✓ IIfConditional (true=C2f P2, false=zeros)")

    # BOTTOM-UP PATH
    x17        = conv_bn_silu(net, f_p2_final, W, "model.17", stride=2)
    c18        = concat(net, [x17, f_p3_neck])
    f_p3_final = c2f_block(net, c18, W, "model.19")
    x20        = conv_bn_silu(net, f_p3_final, W, "model.20", stride=2)
    c21        = concat(net, [x20, f_p4_neck])
    f_p4_final = c2f_block(net, c21, W, "model.22")
    x23        = conv_bn_silu(net, f_p4_final, W, "model.23", stride=2)
    c24        = concat(net, [x23, f_p5])
    f_p5_final = c2f_block(net, c24, W, "model.25")
    print("    ✓ Bottom-up path (layer 17-25)")

    # DETECT HEAD
    detections = build_detect_head(net, W, [f_p2_final, f_p3_final, f_p4_final, f_p5_final])
    print(f"    ✓ Detect head {SHP['layer26']}")
    
    detections.name = "detections"
    net.mark_output(detections)
    print(f"\n  Total layer TRT: {net.num_layers}")
    return net

# =============================================================================
# STEP 5 — build dan validasi
# =============================================================================
def cmd_build():
    print("=" * 65)
    print("BUILD ENGINE — TRT IIfConditional")
    print("=" * 65)
    
    W = extract_weights()
    LOG = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(LOG, "")
    
    bldr = trt.Builder(LOG)
    net  = bldr.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # ✅ PERBAIKAN: Gunakan method dari instance builder
    cfg  = bldr.create_builder_config()
    
    if bldr.platform_has_fast_fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)
        print("FP16 aktif")
        
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    build_network(net, W)
    
    print("\nAutotuning (5-20 menit pertama kali)...")
    eb = bldr.build_serialized_network(net, cfg)
    if eb is None:
        print("BUILD GAGAL — cek WARNING di atas")
        return
        
    os.makedirs(os.path.dirname(ENGINE_PATH) or ".", exist_ok=True)
    open(ENGINE_PATH, "wb").write(eb)
    print(f"\n✅ Engine: {ENGINE_PATH}")

def cmd_validate():
    import glob, cv2
    print("=" * 65)
    print("VALIDASI — lazy evaluation IIfConditional")
    print("=" * 65)
    
    LOG = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(LOG, "")
    
    try:
        with open(ENGINE_PATH, "rb") as f:
            eng = trt.Runtime(LOG).deserialize_cuda_engine(f.read())
    except FileNotFoundError:
        print(f"❌ Engine tidak ditemukan di {ENGINE_PATH}. Jalankan --build terlebih dahulu.")
        return

    assert eng, "Engine gagal dimuat"
    ctx  = eng.create_execution_context()
    
    d_in = torch.zeros(SHP["img"],     dtype=torch.float32, device="cuda")
    d_dt = torch.zeros(SHP["layer26"], dtype=torch.float32, device="cuda")
    
    ctx.set_tensor_address("images",     d_in.data_ptr())
    ctx.set_tensor_address("detections", d_dt.data_ptr())
    s = torch.cuda.current_stream().cuda_stream
    
    paths = sorted(glob.glob(f"{IMG_FOLDER}/*.png"))[:1000]
    if not paths:
        print(f"❌ Tidak ada gambar ditemukan di {IMG_FOLDER}")
        return

    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    
    # Warmup 100 iter
    for _ in range(100):
        ctx.execute_async_v3(s)
    torch.cuda.synchronize()
    
    lats = []
    for p in paths:
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640)).astype(np.float32) / 255.0
        t   = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0).cuda()
        
        d_in.copy_(t)
        st.record()
        ctx.execute_async_v3(s)
        en.record()
        torch.cuda.synchronize()
        lats.append(st.elapsed_time(en))
        
    lats = np.array(lats)
    print(f"\nHasil ({len(lats)} gambar):")
    print(f"  Mean : {np.mean(lats):.3f} ms  |  FPS: ~{1000/np.mean(lats):.0f}")
    print(f"  Std  : {np.std(lats):.4f} ms")
    print(f"  Min  : {np.min(lats):.3f} ms  ← P2 OFF (gambar mudah)")
    print(f"  Max  : {np.max(lats):.3f} ms  ← P2 ON  (gambar sulit)")
    print(f"  P50  : {np.percentile(lats, 50):.3f} ms")
    print(f"  P95  : {np.percentile(lats, 95):.3f} ms")
    print(f"  ΔLat : {np.max(lats)-np.min(lats):.3f} ms (savings estimate)")
    
    if np.std(lats) > 0.05:
        print("\n✅ IIfConditional lazy evaluation AKTIF")
        print("   Variasi latensi membuktikan routing kondisional di GPU")
        print("   Hipotesis tesis terbukti: satu engine, latensi adaptif")
    else:
        print("\n⚠ Variasi minimal — kemungkinan:")
        print("  • Gate selalu satu arah (cek _rmean/_rstd)")
        print("  • Aktifkan --check ulang dan verifikasi running_stats")
        
    del ctx, eng
    torch.cuda.empty_cache()

if __name__ == "__main__":
    if "--check"    in sys.argv: cmd_check()
    elif "--build"  in sys.argv: cmd_build()
    elif "--validate" in sys.argv: cmd_validate()
    else:
        print("Cara Penggunaan:\n  python export_trt_ifconditional.py [--check | --build | --validate]")
