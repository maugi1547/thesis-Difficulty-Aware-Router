"""
Latensi L2 (analisis sensitivitas pasca-hasil, di luar kriteria terkunci): "dari citra ter-decode sampai kotak di host".

Beda dengan L1 (e2e_pass suite, dasar H6 utama):
  L1  timer = ambang B2 (host) -> salin citra FP32 640×640 -> engine -> salin keluaran mentah (1×6×N) ke host
  L2  timer = ambang B2 (host) -> salin citra uint8 ukuran asli -> letterbox + normalisasi di GPU -> engine
              -> [router: baca keputusan & skor] -> dekode + NMS di GPU (torchvision.ops.batched_nms) -> salin kotak ke host
Di luar timer, sama untuk semua varian: decode berkas (cv2.imread) dan penyalinan ke buffer pinned.

Preprocessing meniru YOLODataset val Ultralytics 8.3.240: resize sisi panjang ke imgsz (cv2.INTER_LINEAR ->
bilinear align_corners=False lalu dibulatkan ke bilangan bulat 0–255), LetterBox(scaleup=False, center) padding 114,
BGR -> RGB, /255. Tidak identik bit per bit dengan cv2, sehingga skor gate dapat sedikit berbeda: kesepakatan keputusan
router L2 vs L1 (letterbox CPU) dilaporkan, dan angka akurasi tetap dari pass akurasi suite.

NMS deployment: conf 0,25, satu label per kotak (bawaan predict Ultralytics), IoU 0,7, max_det 300. conf 0,001 diukur
dengan multi-label (protokol val Ultralytics = rfinal.data.nms_one, jalur akurasi suite) sebagai batas atas biaya NMS.
Kotak diskalakan balik per sumbu (gh, gw) seperti rfinal.data._scale_xyxy.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .budget import B2Controller


class GpuLetterbox:
    """Citra BGR uint8 (H0, W0, 3) -> tensor (1, 3, S, S) FP32 [0, 1] di GPU, plus meta untuk skala balik kotak."""

    def __init__(self, imgsz, dev, pin=True):
        self.S, self.dev, self.pin = int(imgsz), dev, pin
        self.cap = 0
        self.host = self.raw = None
        self.meta = None

    def _ensure(self, n):
        if n > self.cap:
            self.cap = int(n * 1.25)
            self.host = torch.empty(self.cap, dtype=torch.uint8)
            if self.pin:
                self.host = self.host.pin_memory()
            self.raw = torch.empty(self.cap, dtype=torch.uint8, device=self.dev)

    def stage(self, im):
        """Di luar timer: salin citra ter-decode ke buffer pinned, hitung geometri letterbox (sama dengan Ultralytics)."""
        h0, w0 = im.shape[:2]
        n = h0 * w0 * 3
        self._ensure(n)
        self.host[:n].copy_(torch.from_numpy(np.ascontiguousarray(im).reshape(-1)))
        r = self.S / max(h0, w0)
        w, h = (min(math.ceil(w0 * r), self.S), min(math.ceil(h0 * r), self.S)) if r != 1 else (w0, h0)
        dw, dh = (self.S - w) / 2, (self.S - h) / 2
        self.meta = dict(h0=h0, w0=w0, h=h, w=w, n=n, top=round(dh - 0.1), left=round(dw - 0.1), gh=h / h0, gw=w / w0)

    def run(self, dst):
        """Di dalam timer (stream aktif): H2D uint8 + resize + padding + normalisasi, ditulis ke `dst`."""
        m = self.meta
        raw = self.raw[:m["n"]]
        raw.copy_(self.host[:m["n"]], non_blocking=True)
        x = raw.view(m["h0"], m["w0"], 3).permute(2, 0, 1).flip(0).unsqueeze(0).float()      # BGR -> RGB
        if (m["h"], m["w"]) != (m["h0"], m["w0"]):
            x = F.interpolate(x, size=(m["h"], m["w"]), mode="bilinear", align_corners=False).round_().clamp_(0, 255)
        dst.fill_(114.0 / 255.0)
        dst[:, :, m["top"]:m["top"] + m["h"], m["left"]:m["left"] + m["w"]] = x / 255.0


def nms_gpu(y, n, meta, conf=0.25, iou=0.7, max_det=300, max_nms=30000, multi_label=False):
    """y (1, 4 + nc, N) xywh piksel letterbox + skor kelas -> (k, 6) [x1, y1, x2, y2, skor, kelas] koordinat asli (GPU).
    multi_label=False: kelas skor maksimum per anchor (predict); True: setiap (anchor, kelas) > conf (val)."""
    import torchvision
    p = y[0, :, :n].float()
    if multi_label:
        i, j = torch.where(p[4:].T > conf)
        b, s, c = p[:4, i].T, p[4 + j, i], j
    else:
        sc, cls = p[4:].max(0)
        keep = sc > conf
        b, s, c = p[:4, keep].T, sc[keep], cls[keep]
    if s.numel() > max_nms:
        top = s.topk(max_nms).indices
        b, s, c = b[top], s[top], c[top]
    xyxy = torch.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
    k = torchvision.ops.batched_nms(xyxy, s, c, iou)[:max_det]
    xyxy = xyxy[k]
    xyxy[:, [0, 2]] = ((xyxy[:, [0, 2]] - meta["left"]) / meta["gw"]).clamp_(0, meta["w0"])
    xyxy[:, [1, 3]] = ((xyxy[:, [1, 3]] - meta["top"]) / meta["gh"]).clamp_(0, meta["h0"])
    return torch.cat([xyxy, s[k, None], c[k, None].float()], 1)


def make_l2_ops(eng, pre, n_a, n_b, dev, b2_args, nms_kw):
    """eng: vanilla, p2static, a3g. op(mark, s) memanggil mark() 4 kali: awal, setelah pre, setelah exec, setelah post."""
    modes = {mv: torch.full((1,), mv, dtype=torch.float32, device=dev) for mv in (-1.0, 0.0, 1.0)}
    ops, state = {}, {"b2": None, "use": []}

    def static(k, n):
        e = eng[k]

        def f(mark, s):
            mark(); pre.run(e.buf["image"])
            mark(); e.run(s)
            mark(); nms_gpu(e.buf[e.out[0]], n, pre.meta, **nms_kw).cpu()
            mark()
        return f
    ops["Vanilla"] = static("vanilla", 10 ** 9)
    ops["P2-Static"] = static("p2static", 10 ** 9)

    e = eng["a3g"]
    pin = (lambda t: t.pin_memory()) if torch.cuda.is_available() else (lambda t: t)
    hu, hs, hthr = pin(torch.zeros(1)), pin(torch.zeros(1)), pin(torch.zeros(1))

    def forced(mv, n):
        def f(mark, s):
            e.bind("mode", modes[mv])
            mark(); pre.run(e.buf["image"])
            mark(); e.run(s)
            mark(); nms_gpu(e.buf["y"], n, pre.meta, **nms_kw).cpu()
            mark()
        return f
    ops["A3 rute A"] = forced(1.0, n_a)
    ops["A3 rute B (potong 8.400)"] = forced(0.0, n_b)
    ops["A3 rute B (tanpa potong)"] = forced(0.0, n_a)

    PRIMARY, SYNC = "Router B2 efisien (L2)", "Router B2 (L2, sync keputusan)"

    def router_sync(mark, s):
        """Varian naif: host menunggu keputusan (sync) lalu memilih jumlah anchor NMS (n_a / n_b)."""
        b2 = state["b2"][SYNC]
        hthr.fill_(float(b2.threshold()))                          # ambang B2 di host (di dalam timer dinding)
        e.bind("mode", modes[-1.0])
        mark(); pre.run(e.buf["image"]); e.buf["thr"].copy_(hthr, non_blocking=True)
        mark(); e.run(s)
        mark()
        hu.copy_(e.buf["use_a"], non_blocking=True); hs.copy_(e.buf["score"], non_blocking=True)
        if s is not None:
            s.synchronize()                                        # keputusan & skor dibutuhkan host
        use = bool(float(hu[0]) > 0.5)
        b2.push(float(hs[0]))
        state["use"][SYNC].append(use)
        nms_gpu(e.buf["y"], n_a if use else n_b, pre.meta, **nms_kw).cpu()
        mark()

    def router(mark, s):
        """Deployment: tanpa sync di tengah. NMS selalu atas n_a anchor (rute B: anchor > n_b = padding nol, skor 0 <
        conf -> kotak identik dengan NMS atas n_b; tests/test_latency_l2.py), keputusan & skor ikut salinan kotak."""
        b2 = state["b2"][PRIMARY]
        hthr.fill_(float(b2.threshold()))
        e.bind("mode", modes[-1.0])
        mark(); pre.run(e.buf["image"]); e.buf["thr"].copy_(hthr, non_blocking=True)
        mark(); e.run(s)
        mark()
        det = nms_gpu(e.buf["y"], n_a, pre.meta, **nms_kw)
        hu.copy_(e.buf["use_a"], non_blocking=True); hs.copy_(e.buf["score"], non_blocking=True)
        det.cpu()                                                  # satu sinkronisasi, sama dengan varian statis
        mark()
        state["use"][PRIMARY].append(bool(float(hu[0]) > 0.5))
        b2.push(float(hs[0]))
    ops[PRIMARY] = router
    ops[SYNC] = router_sync

    def reset():
        state["b2"] = {k: B2Controller(**b2_args, efficient=True) for k in (PRIMARY, SYNC)}
        state["use"] = {k: [] for k in (PRIMARY, SYNC)}
    return ops, state, reset


def l2_pass(ops, state, reset, pre, files, timer, n_warmup=30, n_reps=3, log=print):
    """Seluruh aliran test dalam urutan berkas; riwayat B2 di-reset tiap pengulangan (keputusan setara pass akurasi)."""
    import cv2
    names = list(ops)
    reset()
    for f in files[:n_warmup]:
        pre.stage(cv2.imread(str(f)))
        for k in names:
            timer(ops[k])
    rec = {k: {s: [] for s in ("h2d", "exec", "d2h", "host", "gpu_total", "wall")} for k in names}
    use0 = None
    t0 = time.perf_counter()
    for rep in range(n_reps):
        reset()
        for i, f in enumerate(files):
            pre.stage(cv2.imread(str(f)))                         # decode + salin ke pinned: di luar timer
            for k in (names if (rep + i) % 2 == 0 else names[::-1]):
                r = timer(ops[k])
                for sgm, v in r.items():
                    rec[k][sgm].append(v)
        if rep == 0:
            use0 = list(state["use"]["Router B2 efisien (L2)"])
            assert use0 == state["use"]["Router B2 (L2, sync keputusan)"], "keputusan dua varian router berbeda"
        log(f"  L2 ulangan {rep + 1}/{n_reps} selesai ({time.perf_counter() - t0:.0f} s)")
    # segmen Timer: h2d = pre (salin uint8 + letterbox GPU), d2h = post (keputusan + NMS + salin kotak)
    for k in rec:
        rec[k]["pre"], rec[k]["post"] = rec[k].pop("h2d"), rec[k].pop("d2h")
    return rec, use0


def l1_decisions(eng_a3, loader, b2_args, s, pin):
    """Keputusan router dengan letterbox CPU (dataset Ultralytics), urutan berkas sama, B2 efisien — acuan kesepakatan."""
    b2 = B2Controller(**b2_args, efficient=True)
    dev = eng_a3.buf["image"].device
    mode = torch.full((1,), -1.0, dtype=torch.float32, device=dev)
    eng_a3.bind("mode", mode)
    hu, hs, hthr, himg = (pin(torch.zeros(1)), pin(torch.zeros(1)), pin(torch.zeros(1)), pin(eng_a3.buf["image"]))
    use, files = [], []
    for b in loader:
        himg.copy_(b["img"].float() / 255.0)
        hthr.fill_(float(b2.threshold()))
        ctx = torch.cuda.stream(s) if s is not None else _null()
        with ctx:
            eng_a3.buf["image"].copy_(himg, non_blocking=True); eng_a3.buf["thr"].copy_(hthr, non_blocking=True)
            eng_a3.run(s)
            hu.copy_(eng_a3.buf["use_a"], non_blocking=True); hs.copy_(eng_a3.buf["score"], non_blocking=True)
        if s is not None:
            s.synchronize()
        use.append(bool(float(hu[0]) > 0.5)); b2.push(float(hs[0]))
        files.append(b["im_file"][0])
    return use, files


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
