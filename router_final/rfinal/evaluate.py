"""
Evaluasi sistem final (router_final): Branch A paksa, Branch B paksa, sistem ter-route (gate + B2), dan routing
acak pada aktivasi yang sama. Metrik = pycocotools pada koordinat asli (lihat rfinal/data.py).

Backend:
  pytorch : detektor FP32, skor gate float64 numpy (acuan; = pemeriksaan silang FP32 Tahap 4)
  ort     : engine A3 FP32 via ONNX Runtime CPU (uji lokal)
  trt     : engine A3 TensorRT (sistem yang di-deploy; default FP16)
Pada ort/trt, rute ter-route diambil dari engine dengan mode −1 dan threshold B2 sebagai input — sama seperti
deployment. Rute paksa A/B memakai mode 1/0 pada engine yang sama.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .budget import B2Controller
from .data import build_coco_gt, coco_eval, image_id_of, jdict, to_dets
from .gate import score_np
from .model import BranchA, BranchB, pool4


class _OrtA3:
    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def infer(self, img, thr, mode=-1.0):
        y, use_a, score = self.sess.run(None, {"image": img.cpu().numpy().astype(np.float32),
                                               "mode": np.array([mode], np.float32),
                                               "thr": np.array([thr], np.float32)})
        return torch.from_numpy(y), bool(use_a[0] > 0.5), float(score[0])


@torch.no_grad()
def run_split(model, gate, budget_cfg, thr0, loader, device, nms, backend="pytorch", engine_path=None, log=print):
    """Returns dict per gambar: file, image_id, score, thr, use_a, detA, detB, detR (rute B2)."""
    ctl = B2Controller(budget_cfg["rho"], budget_cfg["window"], budget_cfg["min_hist"], thr0)
    if backend == "pytorch":
        a, b_ = BranchA(model).eval(), BranchB(model).eval()
    elif backend == "ort":
        eng = _OrtA3(engine_path)
    elif backend == "trt":
        from .trt import A3Engine
        eng = A3Engine(engine_path, device=device)
    else:
        raise ValueError(backend)
    R = {k: [] for k in ("file", "image_id", "score", "thr", "use_a", "detA", "detB", "detR")}
    for i, bt in enumerate(loader):
        im = bt["img"].to(device if backend != "ort" else "cpu").float() / 255.0
        rp, osh = bt["ratio_pad"][0], bt["ori_shape"][0]
        thr = ctl.threshold()
        if backend == "pytorch":
            p3, p2, p4, p5 = model.trunk(im)
            s = float(score_np(pool4(p3, p2, p4, p5).cpu().numpy(), gate)[0])   # float64 (= score_g32 Tahap 4)
            yA, yB = a(p3, p2, p4, p5), b_(p3, p4, p5)
            use = bool(np.float32(s) > thr)
            yR = yA if use else yB
        else:
            yA, _, sA = eng.infer(im, thr, 1.0); yA = yA.clone()
            yB, _, _ = eng.infer(im, thr, 0.0); yB = yB.clone()
            yR, use, s = eng.infer(im, thr, -1.0)
            assert use == bool(np.float32(s) > thr), "keputusan engine != perbandingan host"
            assert abs(s - sA) <= 1e-6 * (1 + abs(s)), "skor gate berbeda antar-mode"
        ctl.push(s)
        R["detA"].append(to_dets(yA, rp, osh, **nms)); R["detB"].append(to_dets(yB, rp, osh, **nms))
        R["detR"].append(to_dets(yR, rp, osh, **nms))
        R["file"].append(Path(bt["im_file"][0]).name); R["image_id"].append(image_id_of(bt["im_file"][0]))
        R["score"].append(s); R["thr"].append(float(thr)); R["use_a"].append(use)
        if (i + 1) % 250 == 0:
            log(f"  {i + 1} gambar | aktivasi sejauh ini {np.mean(R['use_a']):.3f}")
    return R


def metrics(R, dataset, names, gt_path, n_random=20, seed=0, max_dets=(1, 10, 100)):
    gt = build_coco_gt(dataset, names, gt_path)
    ids = R["image_id"]
    use = np.asarray(R["use_a"], bool)
    st = lambda dets: coco_eval(gt, jdict(dets, ids), max_dets)[0]
    mix = lambda mask: [R["detA"][i] if mask[i] else R["detB"][i] for i in range(len(mask))]
    out = {"Branch A (paksa)": st(R["detA"]), "Branch B (paksa)": st(R["detB"]), "Router (gate + B2)": st(R["detR"])}
    assert out["Router (gate + B2)"] == st(mix(use)), "deteksi ter-route != campuran A/B per keputusan"
    rng = np.random.default_rng(seed)
    k = int(use.sum())
    rnd = []
    for _ in range(n_random):
        m = np.zeros(len(use), bool); m[rng.choice(len(use), k, replace=False)] = True
        rnd.append(st(mix(m)))
    out[f"Acak (aktivasi sama, rata-rata {n_random})"] = {key: float(np.mean([r[key] for r in rnd])) for key in rnd[0]}
    table = pd.DataFrame(out).T
    table["aktivasi_A"] = [1.0, 0.0, use.mean(), use.mean()]
    return table


def per_image_table(R):
    return pd.DataFrame({k: R[k] for k in ("file", "image_id", "score", "thr", "use_a")})
