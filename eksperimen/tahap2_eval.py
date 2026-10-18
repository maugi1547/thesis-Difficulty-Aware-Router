"""
tahap2_eval.py — Tahap 2: kalibrasi & protokol evaluasi Router-P2 Dual-Branch (satu checkpoint per run).

Isi (dipanggil dari notebook tahap2_eval.ipynb):
  - Muat checkpoint (state_dict -> model baru, kalibrasi persisten), stage 1/2A/2B (sama dgn deployment)
  - cache_split(): SATU forward per gambar -> logit gate mentah z, output Branch A & B, loss per-sample,
    info GT. Semua mode evaluasi (A paksa, B paksa, ter-route pada threshold apa pun, acak, oracle)
    dirakit dari cache yang sama -> NMS/validator identik, biaya murah.
  - Sweep titik operasi (router vs acak) di val -> r*; kalibrasi persentil (T=1, b=-q).
  - COCO (pycocotools) + bootstrap berpasangan per gambar (AP, AP50, AP_small) dari evalImgs
    tanpa menjalankan ulang COCOeval.evaluate().
  - Perilaku gate (RQ3) dgn z (peringkat), koreksi Holm; diagnostik BatchNorm gate (close_mosaic).
"""
import json
import sys
import time
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from router_train_utils import find_router, _raw_gate_logit, _auc_with_ci


# =============================================================================
# MODEL & STAGE
# =============================================================================
def model_fingerprint(m, skip_calib=True):
    """Hash state_dict. Buffer kalibrasi dikecualikan (berubah sah saat kalibrasi)."""
    import hashlib
    h = hashlib.sha256()
    for k, v in sorted(m.state_dict().items()):
        if skip_calib and "_calib_" in k:
            continue
        h.update(k.encode()); h.update(v.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def set_calibration(router, T, b):
    router.calib_T = float(T); router.calib_b = float(b)


def get_calibration(router):
    return float(router.calib_T), float(router.calib_b)


def load_dualbranch(path, model_yaml, nc, names, device):
    """
    Muat checkpoint Ultralytics (last.pt ter-strip: 'model' = EMA) ke DualBranchDetectionModel baru
    lewat state_dict (strict=False hanya untuk buffer kalibrasi). Returns (model eval, info).
    """
    from ultralytics.model_yolov8_router import DualBranchDetectionModel
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG
    ck = torch.load(path, map_location="cpu", weights_only=False)
    key = "ema" if ck.get("ema") is not None else "model"
    src = ck[key]
    sd = {k: (v.float() if v.is_floating_point() else v) for k, v in src.state_dict().items()}
    m = DualBranchDetectionModel(cfg=model_yaml, ch=3, nc=nc, verbose=False)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    bad = [k for k in missing if "calib" not in k]
    assert not bad and not unexpected, f"State dict tidak cocok: missing={bad[:5]} unexpected={list(unexpected)[:5]}"
    hyp = ck.get("train_args") or {}
    m.args = get_cfg(DEFAULT_CFG, overrides={k: hyp[k] for k in ("box", "cls", "dfl") if k in hyp})
    m.names = names
    m = m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    info = dict(ckpt=str(path), weights_key=key, epoch=ck.get("epoch"), train_args_name=hyp.get("name"),
                train_args_seed=hyp.get("seed"), calib_in_ckpt=get_calibration(find_router(m)))
    return m, info


class Stage1_BackboneNeckGate(nn.Module):
    """Layer 0-15 + gate (compute_gate_only: kalibrasi + threshold)."""
    def __init__(self, full_model):
        super().__init__()
        ml = list(full_model.model)
        self.pre_router_layers = nn.ModuleList(ml[0:16])
        self.router = ml[16]
        assert "DifficultyAwareRouter" in type(self.router).__name__

    def forward(self, x):
        y = []
        for m in self.pre_router_layers:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x)
        p3_neck, p2_backbone, p4_neck, p5_sppf = x, y[2], y[12], y[9]
        gate = self.router.compute_gate_only(p3_neck, p2_backbone)
        return gate, p3_neck, p2_backbone, p4_neck, p5_sppf


class Stage2A_WithP2(nn.Module):
    """Layer 16 (expert) + 17-25 + Detect_A -> y (B, 4+nc, N_A)."""
    def __init__(self, full_model):
        super().__init__()
        ml = list(full_model.model)
        self.router, self.branch_a_layers, self.detect_A = ml[16], nn.ModuleList(ml[17:26]), ml[32]

    def forward(self, p3_neck, p2_backbone, p4_neck, p5_sppf):
        p2_out = self.router.compute_expert_only(p3_neck, p2_backbone)
        y = {15: p3_neck, 12: p4_neck, 9: p5_sppf, 16: p2_out}
        x = p2_out
        for i, m in zip(range(17, 26), self.branch_a_layers):
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y[i] = x
        out = self.detect_A([p2_out, y[19], y[22], y[25]])
        return out[0] if isinstance(out, tuple) else out


class Stage2B_NoP2(nn.Module):
    """Layer 26-31 + Detect_B -> y (B, 4+nc, N_B)."""
    def __init__(self, full_model):
        super().__init__()
        ml = list(full_model.model)
        self.branch_b_layers, self.detect_B = nn.ModuleList(ml[26:32]), ml[33]

    def forward(self, p3_neck, p4_neck, p5_sppf):
        y = {15: p3_neck, 12: p4_neck, 9: p5_sppf}
        x = p3_neck
        for i, m in zip(range(26, 32), self.branch_b_layers):
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y[i] = x
        out = self.detect_B([p3_neck, y[28], y[31]])
        return out[0] if isinstance(out, tuple) else out


def make_stages(model):
    s1, s2a, s2b = Stage1_BackboneNeckGate(model).eval(), Stage2A_WithP2(model).eval(), Stage2B_NoP2(model).eval()
    return s1, s2a, s2b


@torch.no_grad()
def check_stage_equivalence(model, stages, loader, device, n=2, tol=1e-5):
    s1, s2a, s2b = stages
    for k, batch in enumerate(loader):
        if k >= n:
            break
        im = batch["img"].to(device).float() / 255.0
        _, p3, p2, p4, p5 = s1(im)
        yA_ref, yB_ref = (o[0] for o in model._predict_once_dual(im))
        dA = ((s2a(p3, p2, p4, p5) - yA_ref).abs().max() / yA_ref.abs().max()).item()
        dB = ((s2b(p3, p4, p5) - yB_ref).abs().max() / yB_ref.abs().max()).item()
        assert dA < tol and dB < tol, f"Stage tidak setara dgn model utuh: {dA:.2e} / {dB:.2e}"
    return True


# =============================================================================
# DATA & VALIDATOR
# =============================================================================
def make_loader(data, split, batch, imgsz=640, workers=2, pin_memory=None):
    """
    Letterbox persegi imgsz, tanpa augmentasi, urutan tetap (sama dengan engine TensorRT fixed-shape).

    SENGAJA DataLoader biasa, BUKAN build_dataloader Ultralytics: InfiniteDataLoader memakai satu iterator
    persisten, sehingga loop yang di-`break` lebih awal (uji kesetaraan, verifikasi ONNX, next(iter(...)))
    menggeser urutan SEMUA iterasi berikutnya (rotasi). Akurasi tetap benar (GT ikut batch), tetapi apa pun
    yang menyejajarkan hasil per gambar dengan dataset.labels / run lain menjadi tidak sejajar.
    """
    from torch.utils.data import DataLoader
    from ultralytics.data import YOLODataset
    ds = YOLODataset(img_path=data[split], data=data, task="detect", imgsz=imgsz,
                     augment=False, rect=False, cache=False, batch_size=batch)
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, collate_fn=ds.collate_fn,
                      pin_memory=torch.cuda.is_available() if pin_memory is None else pin_memory, drop_last=False)


def make_validator_cls():
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils.nms import non_max_suppression

    class ListPredValidator(DetectionValidator):
        """Validator resmi Ultralytics yang menerima daftar prediksi per gambar (NMS & threshold identik)."""
        def postprocess(self, preds):
            out = []
            for pred in preds:
                x = non_max_suppression(pred, self.args.conf, self.args.iou, nc=0, multi_label=True,
                                        agnostic=False, max_det=self.args.max_det)[0]
                out.append({"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]})
            return out

        def scale_preds(self, predn, pbatch):
            """
            Skala letterbox -> resolusi asli PER SUMBU (hanya untuk JSON/pycocotools; metrik Ultralytics dihitung
            di ruang letterbox dan tidak terpengaruh). ops.scale_boxes bawaan memakai satu gain = rasio TINGGI untuk
            kedua sumbu; pada KITTI (1242x375 -> 640x194, tinggi dibulatkan dari 193.2) rasio tinggi 0.51733 vs
            lebar 0.51530, sehingga koordinat x menyusut 0.39% (±4 px di tepi kanan) -> AP COCO turun, terutama
            objek kecil & IoU tinggi.
            """
            (gh, gw), (px, py) = pbatch["ratio_pad"]
            b = predn["bboxes"].clone()
            b[:, [0, 2]] = (b[:, [0, 2]] - px) / gw
            b[:, [1, 3]] = (b[:, [1, 3]] - py) / gh
            h0, w0 = pbatch["ori_shape"]
            b[:, [0, 2]] = b[:, [0, 2]].clamp(0, w0); b[:, [1, 3]] = b[:, [1, 3]].clamp(0, h0)
            return {**predn, "bboxes": b}
    return ListPredValidator


def run_eval(name, predict_fn, loader, data, cfg, device, save_json=False, verbose=True):
    """predict_fn(img_float_0_1) -> (list[(1,4+nc,N)], list[dict info per gambar])."""
    V = make_validator_cls()
    v = V(args=dict(data=cfg["data_yaml"], conf=cfg["conf"], iou=cfg["iou"], max_det=cfg["max_det"],
                    split=cfg.get("report_split", "test"), imgsz=cfg["imgsz"], batch=1, half=False, plots=False,
                    save_json=save_json, task="detect", single_cls=False, agnostic_nms=False))
    v.data, v.device = data, device
    v.init_metrics(SimpleNamespace(names=data["names"], end2end=False))
    records = []
    with torch.no_grad():
        for batch in loader:
            batch = v.preprocess(batch)
            preds, infos = predict_fn(batch["img"])
            v.update_metrics(v.postprocess(preds), batch)
            for f, info in zip(batch["im_file"], infos):
                records.append({"file": Path(f).name, **info})
    stats = v.get_stats()
    if verbose:
        print(f"--- {name} ---"); v.print_results()
    assert v.seen == len(loader.dataset), f"{name}: {v.seen} gambar dievaluasi, dataset {len(loader.dataset)}"
    return dict(name=name, stats=stats, records=records, jdict=list(v.jdict) if save_json else None, seen=v.seen)


def summarize_acc(r):
    s = r["stats"]
    return {"mAP50": s["metrics/mAP50(B)"], "mAP50-95": s["metrics/mAP50-95(B)"],
            "P": s["metrics/precision(B)"], "R": s["metrics/recall(B)"], "n_images": r["seen"]}


# =============================================================================
# CACHE PER GAMBAR (satu forward; semua mode evaluasi dirakit dari sini)
# =============================================================================
@torch.no_grad()
def cache_split(model, stages, loader, device, keep_outputs=True):
    """
    Per gambar: z (logit gate mentah), output Stage2A & Stage2B (CPU), loss per-sample A/B (total & small),
    jumlah objek kecil ter-assign, info GT (resolusi asli, definisi COCO). Loader harus batch=1, shuffle=False.
    """
    s1, s2a, s2b = stages
    rt = find_router(model)
    crit = model.init_criterion()
    labels = loader.dataset.labels
    out = dict(file=[], z=[], yA=[], yB=[], la=[], lb=[], las=[], lbs=[], nsa=[], nsb=[],
               n_obj=[], n_small=[], small_ratio=[], mean_area=[])
    for i, batch in enumerate(loader):
        assert batch["img"].shape[0] == 1, "cache_split butuh loader batch=1"
        f = batch["im_file"][0]
        assert Path(f).name == Path(labels[i]["im_file"]).name, "Urutan loader != dataset.labels"
        im = batch["img"].to(device).float() / 255.0
        _, p3, p2, p4, p5 = s1(im)
        out["z"].append(float(_raw_gate_logit(rt, p3, p2)[0]))
        if keep_outputs:
            out["yA"].append(s2a(p3, p2, p4, p5).float().cpu())
            out["yB"].append(s2b(p3, p4, p5).float().cpu())
        b = {"img": im, **{k: batch[k].to(device) for k in ("cls", "bboxes", "batch_idx")}}
        det_A, det_B = model._predict_once_dual(im)
        crit.loss_A(det_A, b); crit.loss_B(det_B, b)
        out["la"].append(float(crit.loss_A._last_per_sample_loss[0]))
        out["lb"].append(float(crit.loss_B._last_per_sample_loss[0]))
        out["las"].append(float(crit.loss_A._last_per_sample_loss_small[0]))
        out["lbs"].append(float(crit.loss_B._last_per_sample_loss_small[0]))
        out["nsa"].append(float(crit.loss_A._last_small_obj_count[0]))
        out["nsb"].append(float(crit.loss_B._last_small_obj_count[0]))
        lab = labels[i]
        h, w = lab["shape"]
        bx = lab["bboxes"].reshape(-1, 4)
        area = bx[:, 2] * w * bx[:, 3] * h
        out["file"].append(Path(f).name)
        out["n_obj"].append(int(len(bx)))
        out["n_small"].append(int((area < 32 ** 2).sum()))
        out["small_ratio"].append(float((area < 32 ** 2).mean()) if len(bx) else np.nan)
        out["mean_area"].append(float(area.mean()) if len(bx) else np.nan)
    for k in ("z", "la", "lb", "las", "lbs", "nsa", "nsb", "n_obj", "n_small", "small_ratio", "mean_area"):
        out[k] = np.asarray(out[k], dtype=np.float64)
    out["oracle_A"] = out["la"] < out["lb"]                       # 1 = Branch A lebih baik (sinyal training 'total')
    out["valid_small"] = (out["nsa"] > 0) & (out["nsb"] > 0)
    out["oracle_A_small"] = out["las"] < out["lbs"]
    return out


def predict_from_cache(cache, use_a, device):
    """predict_fn untuk run_eval dari cache; loader harus urutan sama (dicek lewat records)."""
    pos = {"k": 0}

    def fn(im):
        preds, infos = [], []
        for _ in range(im.shape[0]):
            k = pos["k"]; pos["k"] += 1
            a = bool(use_a[k])
            # clone WAJIB: non_max_suppression Ultralytics menulis xywh->xyxy IN-PLACE pada tensor input;
            # tanpa salinan, cache rusak untuk evaluasi berikutnya (di CPU .to() tidak menyalin).
            preds.append((cache["yA"][k] if a else cache["yB"][k]).to(device, copy=True))
            infos.append({"branch": "A" if a else "B", "z": float(cache["z"][k])})
        return preds, infos
    return fn


def eval_assignment(name, cache, use_a, loader, data, cfg, device, save_json=False, verbose=False):
    r = run_eval(name, predict_from_cache(cache, use_a, device), loader, data, cfg, device,
                 save_json=save_json, verbose=verbose)
    assert [x["file"] for x in r["records"]] == cache["file"], "Urutan loader != cache"
    r["activation"] = float(np.mean(use_a))
    r["use_a"] = np.asarray(use_a, bool)
    return r


def random_assignment(n, k, seed):
    rng = np.random.default_rng(seed)
    ua = np.zeros(n, bool)
    ua[rng.choice(n, k, replace=False)] = True
    return ua


# =============================================================================
# TITIK OPERASI & KALIBRASI PERSENTIL
# =============================================================================
def fit_percentile_threshold(raw_logits, target_activation):
    """T = 1, b = -q dgn q = kuantil (1 - rho) logit val -> aktivasi val = rho (selain ties)."""
    assert 0.0 < target_activation < 1.0
    z = torch.as_tensor(raw_logits, dtype=torch.float64)
    q = torch.quantile(z, 1.0 - target_activation).item()
    return dict(calib_T=1.0, calib_b=-q, threshold=q, target_activation=float(target_activation),
                activation_on_calib_split=float((z > q).double().mean()),
                n_unique_logits=int(z.unique().numel()), n=int(z.numel()))


def sweep(cache, loader, data, cfg, device, thr_source, rates, n_random=3, seed=0, verbose=False):
    """
    Router (threshold = kuantil logit thr_source) vs acak (aktivasi sama, n_random undian) per rate.
    thr_source = logit VAL, juga saat mengevaluasi test (threshold tidak pernah dari test).
    """
    z = cache["z"]; n = len(z); rows = []
    for r in rates:
        thr = -np.inf if r >= 1 else (np.inf if r <= 0 else float(np.quantile(thr_source, 1 - r)))
        use_a = z > thr
        k = int(use_a.sum())
        acc = summarize_acc(eval_assignment(f"router r={r:.2f}", cache, use_a, loader, data, cfg, device, verbose=verbose))
        rnd = []
        for j in range(n_random if 0 < k < n else 0):
            ua = random_assignment(n, k, seed + 1000 * j + int(r * 100))
            rnd.append(summarize_acc(eval_assignment(f"acak r={k / n:.2f}", cache, ua, loader, data, cfg, device))["mAP50"])
        rows.append(dict(rate_target=r, rate_realized=k / n, threshold=thr if np.isfinite(thr) else None,
                         mAP50=acc["mAP50"], mAP50_95=acc["mAP50-95"],
                         acak_mAP50=float(np.mean(rnd)) if rnd else acc["mAP50"],
                         acak_sd=float(np.std(rnd)) if rnd else 0.0, n_acak=len(rnd)))
    df = pd.DataFrame(rows)
    df["gain_vs_acak"] = df.mAP50 - df.acak_mAP50
    return df


def choose_r_star(df_val):
    """Aturan terkunci: r* = argmax (mAP50 router - mAP50 acak) pada 0 < rho < 1, di VAL."""
    inner = df_val[(df_val.rate_target > 0) & (df_val.rate_target < 1)]
    return float(inner.loc[inner.gain_vs_acak.idxmax(), "rate_target"])


# =============================================================================
# COCO + BOOTSTRAP BERPASANGAN PER GAMBAR
# =============================================================================
def build_coco_gt(dataset, names, path):
    """GT COCO resolusi asli; category_id = cls+1, image_id = int(stem) (= pred_to_json Ultralytics)."""
    from pycocotools.coco import COCO
    images, anns, aid = [], [], 1
    for lab in dataset.labels:
        stem = Path(lab["im_file"]).stem
        iid = int(stem) if stem.isnumeric() else stem
        h, w = lab["shape"]
        images.append({"id": iid, "file_name": Path(lab["im_file"]).name, "height": int(h), "width": int(w)})
        for c, (cx, cy, bw, bh) in zip(lab["cls"].reshape(-1), lab["bboxes"].reshape(-1, 4)):
            bw_px, bh_px = float(bw * w), float(bh * h)
            anns.append({"id": aid, "image_id": iid, "category_id": int(c) + 1,
                         "bbox": [float(cx * w - bw_px / 2), float(cy * h - bh_px / 2), bw_px, bh_px],
                         "area": bw_px * bh_px, "iscrowd": 0})
            aid += 1
    gt = {"images": images, "annotations": anns, "categories": [{"id": i + 1, "name": str(n)} for i, n in names.items()]}
    Path(path).write_text(json.dumps(gt))
    return COCO(str(path))


COCO_KEYS = ["mAP50-95", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large",
             "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]


# maxDets COCOeval. Default COCO [1, 10, 100] (KITTI ±5 objek/gambar). Untuk dataset padat (VisDrone ±59 objek/gambar,
# sebagian >100) set mis. [1, 10, 500] SEBELUM evaluasi; elemen terakhir dipakai untuk AP di summarize()
# (index 2) maupun di CocoBootstrap, jadi keduanya tetap konsisten. Kolom "AR100" lalu berarti AR@maxDets[2].
COCO_MAX_DETS = [1, 10, 100]


def coco_eval(coco_gt, jdict, quiet=True):
    """COCOeval lengkap (maxDets = COCO_MAX_DETS). Returns (stats dict, objek COCOeval — evalImgs dipakai bootstrap)."""
    assert len(COCO_MAX_DETS) == 3, "summarize() pycocotools mengasumsikan tepat 3 nilai maxDets"
    import contextlib, io
    from pycocotools.cocoeval import COCOeval
    ids = set(coco_gt.getImgIds())
    assert {d["image_id"] for d in jdict} <= ids, "image_id prediksi tidak ada di GT"
    buf = io.StringIO()
    with (contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext()):
        if len(jdict):
            dt = coco_gt.loadRes(jdict)
        else:                                               # tanpa deteksi sama sekali -> semua AP = 0
            from pycocotools.coco import COCO
            dt = COCO(); dt.dataset = {"images": coco_gt.dataset["images"], "categories": coco_gt.dataset["categories"],
                                      "annotations": []}
            dt.createIndex()
        E = COCOeval(coco_gt, dt, "bbox")
        E.params.imgIds = sorted(ids)
        E.params.maxDets = list(COCO_MAX_DETS)
        E.evaluate(); E.accumulate(); E.summarize()
    # pycocotools 2.0.x: stats[0] (mAP50-95) memakai maxDets=100 TERTANAM (bukan maxDets[2]) -> -1 bila 100 tidak
    # ada di daftar. Hitung ulang dari precision pada maxDets[-1], area 'all' (identik untuk default [1,10,100]).
    prec = E.eval["precision"][:, :, :, 0, -1]
    E.stats[0] = float(prec[prec > -1].mean()) if (prec > -1).any() else -1.0
    return dict(zip(COCO_KEYS, map(float, E.stats))), E


class CocoBootstrap:
    """
    AP COCO untuk subset gambar (boleh berulang, untuk bootstrap) langsung dari evalImgs COCOeval:
    replikasi accumulate() + summarize() (maxDet 100) tanpa menjalankan evaluate() ulang.
    Diverifikasi sama dengan COCOeval.stats pada himpunan penuh (lihat verify()).
    """
    AREAS = {"all": 0, "small": 1, "medium": 2, "large": 3}

    def __init__(self, E):
        p = E.params
        self.T, self.R, self.I = len(p.iouThrs), p.recThrs, len(p.imgIds)
        self.A, self.K, self.maxDet = len(p.areaRng), len(p.catIds), p.maxDets[-1]
        self.iou50 = int(np.where(np.isclose(p.iouThrs, 0.5))[0][0])
        self.stats_full = np.asarray(E.stats, float)
        # pra-ekstrak array per (kategori, area, gambar); None = tidak ada GT & deteksi
        self.cells = {}
        for k in range(self.K):
            for name, a in self.AREAS.items():
                base = k * self.A * self.I + a * self.I
                lst = []
                for i in range(self.I):
                    e = E.evalImgs[base + i]
                    if e is None:
                        lst.append(None); continue
                    lst.append((np.asarray(e["dtScores"][:self.maxDet], float),
                                np.asarray(e["dtMatches"][:, :self.maxDet]) > 0,
                                np.asarray(e["dtIgnore"][:, :self.maxDet], bool),
                                int(np.count_nonzero(np.asarray(e["gtIgnore"]) == 0))))
                self.cells[(k, name)] = lst

    def _precision(self, idx, area):
        """List per kategori: array (T, R) presisi terinterpolasi, atau None (kategori tanpa GT valid)."""
        out = []
        for k in range(self.K):
            cell = self.cells[(k, area)]
            E = [cell[i] for i in idx if cell[i] is not None]
            if not E:
                out.append(None); continue
            npig = sum(e[3] for e in E)
            if npig == 0:
                out.append(None); continue
            scores = np.concatenate([e[0] for e in E])
            order = np.argsort(-scores, kind="mergesort")
            dtm = np.concatenate([e[1] for e in E], axis=1)[:, order]
            dtig = np.concatenate([e[2] for e in E], axis=1)[:, order]
            tp = np.cumsum(dtm & ~dtig, axis=1, dtype=float)
            fp = np.cumsum(~dtm & ~dtig, axis=1, dtype=float)
            nd = tp.shape[1]
            q = np.zeros((self.T, len(self.R)))
            if nd:
                rc = tp / npig
                pr = tp / (fp + tp + np.spacing(1))
                pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]      # presisi monoton (= loop COCO)
                for t in range(self.T):
                    ii = np.searchsorted(rc[t], self.R, side="left")
                    ok = ii < nd
                    q[t, ok] = pr[t, ii[ok]]
            out.append(q)
        return out

    def metrics(self, idx):
        """dict mAP50-95, mAP50 (area all) & mAP_small untuk subset gambar idx (indeks ke E.params.imgIds)."""
        P_all = [p for p in self._precision(idx, "all") if p is not None]
        P_s = [p for p in self._precision(idx, "small") if p is not None]
        m = {}
        if P_all:
            P = np.stack(P_all); m["mAP50-95"] = float(P.mean()); m["mAP50"] = float(P[:, self.iou50].mean())
        else:
            m["mAP50-95"] = m["mAP50"] = np.nan
        m["mAP_small"] = float(np.stack(P_s).mean()) if P_s else np.nan
        return m

    def verify(self, atol=1e-9):
        full = self.metrics(np.arange(self.I))
        ref = {"mAP50-95": self.stats_full[0], "mAP50": self.stats_full[1], "mAP_small": self.stats_full[3]}
        for k, v in ref.items():
            v = np.nan if v < 0 else v
            assert (np.isnan(v) and np.isnan(full[k])) or abs(full[k] - v) <= atol, f"CocoBootstrap {k}: {full[k]} != COCOeval {v}"
        return full


BOOT_METRICS = ("mAP50-95", "mAP50", "mAP_small")


def paired_bootstrap(boot_a, boots_b, n_boot=1000, seed=0):
    """
    Selisih metrik A − rata-rata(B_list) pada resample gambar YANG SAMA (bootstrap berpasangan per gambar).
    boot_a: CocoBootstrap; boots_b: list CocoBootstrap (mis. 3 undian router acak) — semua dari GT yang sama.
    Returns DataFrame: metric, diff, ci_lo, ci_hi (persentil 95%), p_two_sided (proporsi bootstrap melewati 0;
    batas bawah 1/B — tulis 'p < 1/B', bukan p = 0).
    """
    I = boot_a.I
    assert all(b.I == I for b in boots_b)
    rng = np.random.default_rng(seed)

    def diff(idx):
        a = boot_a.metrics(idx)
        bs = [b.metrics(idx) for b in boots_b]
        return {m: a[m] - np.nanmean([b[m] for b in bs]) for m in BOOT_METRICS}

    point = diff(np.arange(I))
    samples = {m: [] for m in BOOT_METRICS}
    for _ in range(n_boot):
        d = diff(rng.integers(0, I, I))
        for m in BOOT_METRICS:
            samples[m].append(d[m])
    rows = []
    for m in BOOT_METRICS:
        s = np.asarray(samples[m]); s = s[np.isfinite(s)]
        lo, hi = np.percentile(s, [2.5, 97.5])
        p = min(1.0, max(2 * min((s <= 0).mean(), (s >= 0).mean()), 1.0 / len(s)))   # resolusi p = 1/B
        rows.append(dict(metric=m, diff=point[m], ci_lo=lo, ci_hi=hi, p_two_sided=p, n_boot=len(s)))
    return pd.DataFrame(rows)


# =============================================================================
# PERILAKU GATE (RQ3)
# =============================================================================
def holm(pvals):
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def gate_behaviour(cache):
    """Spearman z vs kepadatan/ukuran, parsial (proporsi kecil | jumlah objek), AUC vs oracle; Holm."""
    from scipy import stats as st
    df = pd.DataFrame({k: cache[k] for k in ("z", "n_obj", "n_small", "small_ratio", "mean_area")})
    tests = []

    def sp(x, y, label):
        m = df[[x, y]].dropna()
        r, p = st.spearmanr(m[x], m[y]); tests.append(dict(test=label, r=r, p=p, n=len(m)))

    sp("z", "n_obj", "Spearman z vs jumlah objek")
    sp("z", "small_ratio", "Spearman z vs proporsi objek kecil")
    sp("z", "n_small", "Spearman z vs jumlah objek kecil")
    sp("z", "mean_area", "Spearman z vs rata-rata luas objek")
    m = df[["z", "small_ratio", "n_obj"]].dropna()
    rx, ry, rz = st.rankdata(m.z), st.rankdata(m.small_ratio), st.rankdata(m.n_obj)
    res = lambda a, b: a - np.polyval(np.polyfit(b, a, 1), b)
    r_part, p_part = st.pearsonr(res(rx, rz), res(ry, rz))
    tests.append(dict(test="Parsial (Spearman) z vs proporsi kecil | jumlah objek (H3)", r=r_part, p=p_part, n=len(m)))
    T = pd.DataFrame(tests); T["p_holm"] = holm(T.p)
    auc = _auc_with_ci(cache["oracle_A"], cache["z"])
    v = cache["valid_small"]
    auc_s = _auc_with_ci(cache["oracle_A_small"][v], cache["z"][v]) if v.sum() >= 20 else (np.nan, np.nan, np.nan)
    A = pd.DataFrame([dict(target="loss_A < loss_B (total)", n=len(cache["z"]), oracle_rate=float(cache["oracle_A"].mean()),
                           auc=auc[0], ci_lo=auc[1], ci_hi=auc[2]),
                      dict(target="loss_A_small < loss_B_small (gambar valid)", n=int(v.sum()),
                           oracle_rate=float(cache["oracle_A_small"][v].mean()) if v.sum() else np.nan,
                           auc=auc_s[0], ci_lo=auc_s[1], ci_hi=auc_s[2])])
    return T, A


# =============================================================================
# DIAGNOSTIK BATCHNORM GATE (hipotesis close_mosaic)
# =============================================================================
def _squeeze_bns(rt):
    return [m for seq in (rt.squeeze_p3, rt.squeeze_p2) for m in seq.modules() if isinstance(m, nn.BatchNorm2d)]


@torch.no_grad()
def gate_logits_on(model, loader, device, max_images=None):
    """z per gambar lewat forward penuh model (hook input router) — mode BN mengikuti status modul."""
    rt = find_router(model)
    feats, zs, files = {}, [], []
    h = rt.register_forward_pre_hook(lambda m, inp: feats.update(p3=inp[0][0], p2=inp[0][1]))
    try:
        n = 0
        for batch in loader:
            im = batch["img"].to(device).float() / 255.0
            model._predict_once_dual(im)
            zs.append(_raw_gate_logit(rt, feats["p3"], feats["p2"]).cpu().numpy())
            files += [Path(f).name for f in batch["im_file"]]
            n += im.shape[0]
            if max_images and n >= max_images:
                break
    finally:
        h.remove()
    return np.concatenate(zs), files


@torch.no_grad()
def bn_diagnostic(model, val_cache, val_loader_b16, train_loader_b16, device, n_train=1297):
    """
    Bandingkan peringkat gate (AUC vs oracle 'total' val) & sebaran logit di val untuk:
      (a) eval (running stats)             — referensi, harus = cache
      (b) BN squeeze memakai statistik batch val (batch 16) — meniru mode training
      (c) running stats BN squeeze diestimasi ulang di gambar TRAIN tanpa augmentasi (kumulatif)
      (d) gambar TRAIN tanpa augmentasi, mode eval — apakah gate berbeda di data training?
    Semua pada salinan model; model asli tidak berubah.
    """
    rows = []
    oracle = dict(zip(val_cache["file"], val_cache["oracle_A"]))

    def row(name, z, files, labels=None):
        y = np.array([oracle[f] for f in files]) if labels is None else labels
        a = _auc_with_ci(y, z, n_boot=500)
        rows.append(dict(variant=name, n=len(z), logit_mean=float(np.mean(z)), logit_std=float(np.std(z)),
                         act_z_gt_0=float(np.mean(z > 0)), auc=a[0], ci_lo=a[1], ci_hi=a[2]))

    z, f = gate_logits_on(model, val_loader_b16, device); row("(a) val, eval (running stats)", z, f)
    mb = copy.deepcopy(model)
    for bn in _squeeze_bns(find_router(mb)):
        bn.train()
    z, f = gate_logits_on(mb, val_loader_b16, device); row("(b) val, BN squeeze statistik batch", z, f)
    mc = copy.deepcopy(model); rt = find_router(mc)
    for bn in _squeeze_bns(rt):
        bn.reset_running_stats(); bn.momentum = None; bn.train()
    gate_logits_on(mc, train_loader_b16, device, max_images=n_train)       # hanya mengisi running stats
    for bn in _squeeze_bns(rt):
        bn.eval()
    z, f = gate_logits_on(mc, val_loader_b16, device); row("(c) val, running stats BN squeeze diestimasi ulang di train", z, f)
    return pd.DataFrame(rows)


def train_split_gate(model, train_loader_b1, device, max_images=1297):
    """(d) Gate & oracle di gambar train tanpa augmentasi, mode eval (dipotong max_images)."""
    stages = make_stages(model)
    sub = _LimitedLoader(train_loader_b1, max_images)
    c = cache_split(model, stages, sub, device, keep_outputs=False)
    a = _auc_with_ci(c["oracle_A"], c["z"], n_boot=500)
    return dict(variant="(d) train tanpa augmentasi, eval", n=len(c["z"]), logit_mean=float(c["z"].mean()),
                logit_std=float(c["z"].std()), act_z_gt_0=float((c["z"] > 0).mean()), auc=a[0], ci_lo=a[1], ci_hi=a[2])


class _LimitedLoader:
    """Bungkus loader: berhenti setelah n gambar; dataset.labels dipotong agar cache_split tetap sinkron."""
    def __init__(self, loader, n):
        self.loader, self.n = loader, n
        self.dataset = SimpleNamespace(labels=loader.dataset.labels[:n])

    def __iter__(self):
        for i, b in enumerate(self.loader):
            if i >= self.n:
                break
            yield b


# =============================================================================
# BASELINE STATIS (Vanilla, P2-Static)
# =============================================================================
def load_static(path, device):
    from ultralytics import YOLO
    m = YOLO(path).model.float().to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def predict_static_pt(m):
    @torch.no_grad()
    def fn(im):
        preds = []
        for i in range(im.shape[0]):
            out = m(im[i:i + 1])
            preds.append(out[0] if isinstance(out, (tuple, list)) else out)
        return preds, [{} for _ in range(im.shape[0])]
    return fn


# =============================================================================
# ONNX & TENSORRT (FP32, 3 engine router + 2 baseline; sama dgn notebook uji 2026-10-08)
# =============================================================================
class _Stage1Out(nn.Module):
    def __init__(self, s):
        super().__init__(); self.s = s

    def forward(self, x):
        return self.s(x)


class _StaticOut(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        out = self.m(x)
        return out[0] if isinstance(out, (tuple, list)) else out


def export_onnx(module, args, path, in_names, out_names):
    with torch.no_grad():
        torch.onnx.export(module.eval(), args, str(path), input_names=in_names, output_names=out_names,
                          opset_version=17, dynamic_axes=None, dynamo=False)


def cpu_copy(m):
    """
    Salinan model di CPU yang aman untuk ekspor. Detect menyimpan anchors/strides sebagai ATRIBUT biasa (bukan
    buffer) dan hanya menghitung ulang bila shape input berubah; .cpu() tidak memindahkannya, sehingga salinan
    CPU dari model yang pernah jalan di GPU masih memegang anchors cuda:0 ("Expected all tensors to be on the
    same device"). shape=None memaksa anchors dihitung ulang di device yang benar pada forward berikutnya.
    """
    c = copy.deepcopy(m).cpu().eval()
    for mod in c.modules():
        if hasattr(mod, "anchors") and hasattr(mod, "shape"):
            mod.shape = None
            mod.anchors, mod.strides = torch.empty(0), torch.empty(0)
    return c


def export_all_onnx(model, vanilla, p2static, x0, out_dir, tag):
    """x0: (1,3,H,W) float CPU. Returns dict path ONNX + salinan CPU stage (untuk verifikasi)."""
    m_cpu = cpu_copy(model)
    assert get_calibration(find_router(m_cpu)) == get_calibration(find_router(model)), "Kalibrasi tidak ikut tersalin"
    s1c, s2ac, s2bc = make_stages(m_cpu)
    with torch.no_grad():
        _, p3, p2, p4, p5 = s1c(x0)
    P = {k: Path(out_dir) / f"{k}_{tag}.onnx" for k in ("stage1", "stage2a", "stage2b")}
    P["vanilla"], P["p2static"] = Path(out_dir) / "vanilla.onnx", Path(out_dir) / "p2static.onnx"
    export_onnx(_Stage1Out(s1c), (x0,), P["stage1"], ["image"], ["gate", "p3_neck", "p2_backbone", "p4_neck", "p5_sppf"])
    export_onnx(s2ac, (p3, p2, p4, p5), P["stage2a"], ["p3_neck", "p2_backbone", "p4_neck", "p5_sppf"], ["y_A"])
    export_onnx(s2bc, (p3, p4, p5), P["stage2b"], ["p3_neck", "p4_neck", "p5_sppf"], ["y_B"])
    statics = {"vanilla": _StaticOut(cpu_copy(vanilla)).eval(), "p2static": _StaticOut(cpu_copy(p2static)).eval()}
    for k, mod in statics.items():
        export_onnx(mod, (x0,), P[k], ["image"], ["y"])
    return P, (s1c, s2ac, s2bc), statics


def verify_onnx(P, cpu_stages, statics, loader, n=5):
    import onnxruntime as ort
    sess = {k: ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]) for k, p in P.items()}
    run = lambda k, feed: sess[k].run(None, {i.name: feed[i.name] for i in sess[k].get_inputs()})
    rel = lambda a, b: float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))
    s1c, s2ac, s2bc = cpu_stages
    worst = {}
    with torch.no_grad():
        for k, batch in enumerate(loader):
            if k >= n:
                break
            x = batch["img"].float() / 255.0
            g, p3, p2, p4, p5 = s1c(x)
            o1 = run("stage1", {"image": x.numpy()})
            assert float(o1[0].flatten()[0]) == float(g.flatten()[0]), f"Gate ONNX != PyTorch pada {batch['im_file'][0]}"
            worst["stage1"] = max(worst.get("stage1", 0), *[rel(a, b.numpy()) for a, b in zip(o1[1:], (p3, p2, p4, p5))])
            f = {"p3_neck": p3.numpy(), "p2_backbone": p2.numpy(), "p4_neck": p4.numpy(), "p5_sppf": p5.numpy()}
            worst["stage2a"] = max(worst.get("stage2a", 0), rel(run("stage2a", f)[0], s2ac(p3, p2, p4, p5).numpy()))
            worst["stage2b"] = max(worst.get("stage2b", 0), rel(run("stage2b", f)[0], s2bc(p3, p4, p5).numpy()))
            for nm, mod in statics.items():
                worst[nm] = max(worst.get(nm, 0), rel(run(nm, {"image": x.numpy()})[0], mod(x).numpy()))
    assert all(v < 1e-4 for v in worst.values()), f"Selisih ONNX vs PyTorch > 1e-4: {worst}"
    return worst


def init_trt():
    """Import TensorRT + pycuda (buat context CUDA). Returns namespace atau None bila tidak tersedia."""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
    except Exception as e:
        print("TensorRT/pycuda tidak tersedia:", e)
        return None
    try:
        cuda.Device(0)                                  # GPU dicek lewat pycuda: torch.cuda TIDAK disentuh
    except Exception as e:
        print("GPU tidak tersedia untuk pycuda:", e)
        return None
    NS = SimpleNamespace(trt=trt, cuda=cuda, logger=trt.Logger(trt.Logger.WARNING))

    def build_engine(onnx_path, engine_path):
        builder = trt.Builder(NS.logger)
        network = builder.create_network()
        parser = trt.OnnxParser(network, NS.logger)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)   # tanpa flag presisi = FP32
        eb = builder.build_serialized_network(network, config)
        assert eb is not None, f"Gagal build {onnx_path}"
        Path(engine_path).write_bytes(eb)

    def _io(engine):
        inp, out = [], []
        for i in range(engine.num_io_tensors):
            n = engine.get_tensor_name(i)
            (inp if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT else out).append(n)
        return inp, out

    class TRTEngineOptimized:
        """Buffer dialokasikan sekali. Vanilla & P2-Static."""
        def __init__(self, path):
            with open(path, "rb") as f, trt.Runtime(NS.logger) as rt:
                self.engine = rt.deserialize_cuda_engine(f.read())
            self.ctx = self.engine.create_execution_context(); self.stream = cuda.Stream()
            self.inp, self.out = _io(self.engine); self.d, self.h = {}, {}
            for n in self.inp:
                shape, dt = tuple(self.engine.get_tensor_shape(n)), trt.nptype(self.engine.get_tensor_dtype(n))
                self.ctx.set_input_shape(n, shape)
                self.d[n] = cuda.mem_alloc(int(np.prod(shape)) * np.dtype(dt).itemsize)
                self.ctx.set_tensor_address(n, int(self.d[n]))
            for n in self.out:
                self.h[n] = cuda.pagelocked_empty(tuple(self.ctx.get_tensor_shape(n)), trt.nptype(self.engine.get_tensor_dtype(n)))
                self.d[n] = cuda.mem_alloc(self.h[n].nbytes)
                self.ctx.set_tensor_address(n, int(self.d[n]))

        def infer(self, image_np):
            cuda.memcpy_htod_async(self.d[self.inp[0]], image_np, self.stream)
            self.ctx.execute_async_v3(stream_handle=self.stream.handle)
            for n in self.out:
                cuda.memcpy_dtoh_async(self.h[n], self.d[n], self.stream)
            self.stream.synchronize()
            return self.h[self.out[0]]

    class TRTEngineChainable:
        """Router: Stage1 dari host; Stage2 menerima pointer device Stage1 (tanpa salinan fitur)."""
        def __init__(self, path):
            with open(path, "rb") as f, trt.Runtime(NS.logger) as rt:
                self.engine = rt.deserialize_cuda_engine(f.read())
            self.ctx = self.engine.create_execution_context(); self.stream = cuda.Stream()
            self.inp, self.out = _io(self.engine); self.d_in, self.d_out, self.h_out = {}, {}, {}
            for n in self.inp:
                shape, dt = tuple(self.engine.get_tensor_shape(n)), trt.nptype(self.engine.get_tensor_dtype(n))
                self.ctx.set_input_shape(n, shape)
                self.d_in[n] = cuda.mem_alloc(int(np.prod(shape)) * np.dtype(dt).itemsize)
            for n in self.out:
                self.h_out[n] = cuda.pagelocked_empty(tuple(self.ctx.get_tensor_shape(n)), trt.nptype(self.engine.get_tensor_dtype(n)))
                self.d_out[n] = cuda.mem_alloc(self.h_out[n].nbytes)
                self.ctx.set_tensor_address(n, int(self.d_out[n]))

        def run_from_host(self, image_np):
            n = self.inp[0]
            cuda.memcpy_htod_async(self.d_in[n], image_np, self.stream)
            self.ctx.set_tensor_address(n, int(self.d_in[n]))
            self.ctx.execute_async_v3(stream_handle=self.stream.handle)

        def read_scalar(self, name):
            cuda.memcpy_dtoh_async(self.h_out[name], self.d_out[name], self.stream)
            self.stream.synchronize()
            return float(self.h_out[name].flatten()[0])

        def run_from_device(self, ptrs):
            for n, p in ptrs.items():
                self.ctx.set_tensor_address(n, int(p))
            self.ctx.execute_async_v3(stream_handle=self.stream.handle)
            for n in self.out:
                cuda.memcpy_dtoh_async(self.h_out[n], self.d_out[n], self.stream)
            self.stream.synchronize()
            return self.h_out[self.out[0]]

    NS.build_engine, NS.Optimized, NS.Chainable = build_engine, TRTEngineOptimized, TRTEngineChainable
    return NS


class TRTSystem:
    """Router 3-engine + 2 baseline. router(image_np, force=None|'A'|'B') -> (y, gate, branch)."""
    A_KEYS, B_KEYS = ("p3_neck", "p2_backbone", "p4_neck", "p5_sppf"), ("p3_neck", "p4_neck", "p5_sppf")

    def __init__(self, NS, engines):
        self.s1, self.s2a, self.s2b = (NS.Chainable(engines[k]) for k in ("stage1", "stage2a", "stage2b"))
        self.vanilla, self.p2static = NS.Optimized(engines["vanilla"]), NS.Optimized(engines["p2static"])

    def router(self, image_np, force=None):
        self.s1.run_from_host(image_np)
        g = self.s1.read_scalar("gate")
        use_a = (g > 0.5) if force is None else (force == "A")
        if use_a:
            y = self.s2a.run_from_device({k: self.s1.d_out[k] for k in self.A_KEYS})
        else:
            y = self.s2b.run_from_device({k: self.s1.d_out[k] for k in self.B_KEYS})
        return y, g, "A" if use_a else "B"


def predict_router_trt(sysm):
    def fn(im):
        preds, infos = [], []
        for i in range(im.shape[0]):
            y, g, br = sysm.router(np.ascontiguousarray(im[i:i + 1].cpu().numpy(), dtype=np.float32))
            preds.append(torch.from_numpy(y.copy()).to(im.device)); infos.append({"gate": g, "branch": br})
        return preds, infos
    return fn


def predict_static_trt(eng):
    def fn(im):
        preds = [torch.from_numpy(eng.infer(np.ascontiguousarray(im[i:i + 1].cpu().numpy(), dtype=np.float32)).copy()).to(im.device)
                 for i in range(im.shape[0])]
        return preds, [{} for _ in range(im.shape[0])]
    return fn


def lat_stats(a):
    a = np.asarray(a, float)
    return dict(mean=a.mean(), std=a.std(), p50=np.percentile(a, 50), p95=np.percentile(a, 95),
                p99=np.percentile(a, 99), min=a.min(), max=a.max(), n=int(a.size))


def benchmark_latency(sysm, loader, n_warmup=50, n_reps=3):
    """
    Interleaved (urutan dibalik tiap putaran), gambar test asli yang sama, cakupan: H2D gambar + inferensi
    (+ baca 1 skalar gate) + D2H output. Router dipaksa A / B diukur untuk kurva latensi t(r) = r·T_A + (1−r)·T_B.
    """
    as_np = lambda b: np.ascontiguousarray(b["img"].numpy().astype(np.float32) / 255.0)
    for k, batch in enumerate(loader):
        if k >= n_warmup:
            break
        im = as_np(batch)
        sysm.vanilla.infer(im); sysm.p2static.infer(im); sysm.router(im, "A"); sysm.router(im, "B")
    fns = {"Vanilla": lambda im: sysm.vanilla.infer(im), "P2-Static": lambda im: sysm.p2static.infer(im),
           "Router": lambda im: sysm.router(im), "Router paksa A": lambda im: sysm.router(im, "A"),
           "Router paksa B": lambda im: sysm.router(im, "B")}
    t = {k: [] for k in fns}
    branches, by_branch = [], {"A": [], "B": []}
    for rep in range(n_reps):
        for batch in loader:
            im = as_np(batch)
            order = list(fns) if rep % 2 == 0 else list(fns)[::-1]
            for name in order:
                t0 = time.perf_counter(); out = fns[name](im); dt = (time.perf_counter() - t0) * 1e3
                t[name].append(dt)
                if name == "Router":
                    by_branch[out[2]].append(dt)
                    if rep == 0:
                        branches.append(out[2])
    LAT = {k: lat_stats(v) for k, v in t.items()}
    for b in "AB":
        LAT[f"Router (saat Branch {b})"] = lat_stats(by_branch[b]) if by_branch[b] else None
    return LAT, float(np.mean([b == "A" for b in branches]))


# =============================================================================
# JOB TENSORRT DI PROSES TERPISAH  (python -m tahap2_eval <trt_job.json>)
# =============================================================================
def _rss_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        return float("nan")


def trt_job_main(job_path):
    """
    Build engine + mAP dari output TensorRT + latensi, di proses TERPISAH dari kernel notebook:
      - kernel notebook memegang cache besar (output Branch A/B, jdict, COCOeval) -> RAM kernel tidak ikut
        terbebani builder TensorRT;
      - proses ini TIDAK pernah membuat context CUDA PyTorch (validator & NMS di CPU), sehingga hanya ada satu
        context (pycuda) -> tidak ada konflik dua context di satu proses;
      - bila TensorRT/pycuda crash (segfault) atau proses dibunuh OOM, kernel notebook tetap hidup dan
        returncode memberi tahu penyebabnya.
    Masukan: JSON (onnx, data_yaml, split, imgsz, conf, iou, max_det, gt_json, out_json, n_warmup, n_reps).
    """
    global COCO_MAX_DETS
    job = json.loads(Path(job_path).read_text())
    COCO_MAX_DETS = list(job.get("coco_max_dets", COCO_MAX_DETS))
    t0 = time.perf_counter()
    log = lambda *a: print(f"[trt-job {time.perf_counter() - t0:6.0f}s | RSS {_rss_gb():.1f} GB]", *a, flush=True)
    from ultralytics.data.utils import check_det_dataset
    dev = torch.device("cpu")
    DATA = check_det_dataset(job["data_yaml"])
    # pin_memory=False: pinned memory membuat context CUDA PyTorch -> dua context (torch + pycuda) di satu
    # proses -> "CUDA error: invalid resource handle" (penyebab kernel mati di Kaggle, 10 Okt).
    loader = make_loader(DATA, job["split"], 1, job["imgsz"], workers=2, pin_memory=False)
    EV = {k: job[k] for k in ("data_yaml", "conf", "iou", "max_det", "imgsz")}; EV["report_split"] = job["split"]
    NS = init_trt()
    assert not torch.cuda.is_initialized(), "Context CUDA PyTorch terbentuk di proses job TRT — harus tidak ada"
    if NS is None:
        Path(job["out_json"]).write_text(json.dumps({"trt_available": False}))
        sys.exit(3)
    log("TensorRT", NS.trt.__version__)
    eng = {}
    for k, p in job["onnx"].items():
        eng[k] = str(Path(p).with_suffix(".engine"))
        NS.build_engine(p, eng[k]); log("engine", k, "OK")
    SYS = TRTSystem(NS, eng)
    from pycocotools.coco import COCO
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(job["gt_json"])
    res = {"trt_available": True, "trt_version": NS.trt.__version__, "acc": {}}
    for name, fn in (("Router (TRT)", predict_router_trt(SYS)), ("Vanilla (TRT)", predict_static_trt(SYS.vanilla)),
                     ("P2-Static (TRT)", predict_static_trt(SYS.p2static))):
        r = run_eval(name, fn, loader, DATA, EV, dev, save_json=True, verbose=False)
        res["acc"][name] = {**summarize_acc(r), **{f"coco_{m}": v for m, v in coco_eval(gt, r["jdict"])[0].items()}}
        if name.startswith("Router"):
            res["files"] = [x["file"] for x in r["records"]]
            res["router_branch"] = [x["branch"] for x in r["records"]]
            res["router_gate"] = [x["gate"] for x in r["records"]]
        log(name, "mAP50", round(res["acc"][name]["mAP50"], 4)); del r
    lat, act = benchmark_latency(SYS, loader, job["n_warmup"], job["n_reps"])
    res.update(latency=lat, activation_bench=act)
    Path(job["out_json"]).write_text(json.dumps(res, default=float))
    assert not torch.cuda.is_initialized(), "Context CUDA PyTorch terbentuk di proses job TRT"
    log("selesai")


if __name__ == "__main__":
    import sys
    trt_job_main(sys.argv[1])
