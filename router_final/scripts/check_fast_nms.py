"""
Cek NMS cepat (fast_nms_k) sebelum dipakai suite/ablasi: apakah deteksi IDENTIK dengan NMS penuh, dan seberapa cepat.

    python scripts/check_fast_nms.py --config configs/visdrone.yaml --weights RUN/weights/last.pt --out RUN/cek_nms
    python scripts/check_fast_nms.py ... --assert-config        # gagal (kode 1) bila K di config tidak identik 100%

Per gambar val, untuk Branch A dan B, presisi FP32 dan FP16 (model.half(), meniru pola tie keluaran TensorRT FP16),
dan setiap max_det (gate.nms & analysis): bandingkan to_dets(fast_nms_k=K) dengan to_dets() penuh, byte per byte.
rfinal.data.nms_one menjamin kesamaan kecuali bila ada skor yang persis sama (tie) di antara kotak yang bertumpuk;
skrip ini mengukur seberapa sering itu terjadi, plus jalur yang diambil (penuh langsung / cepat / fallback) dan waktu.
Bila ada yang tidak identik: selisih metrik COCO (mAP50-95, mAP50, mAP_small) penuh vs cepat ikut dilaporkan.

Keluaran: ringkasan.csv, kandidat.csv, (selisih_metrik.csv), provenance.json.
"""
import argparse
import copy
import sys
import time
from pathlib import Path

import _path  # noqa: F401
import numpy as np
import pandas as pd
import torch

from rfinal import data as D
from rfinal.data import build_coco_gt, coco_eval, image_id_of, jdict, load_data, make_loader, to_dets
from rfinal.env import load_config, provenance, write_json
from rfinal.model import BranchA, BranchB, load_detector, reset_anchor_cache


def timed(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.perf_counter()
    r = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return r, (time.perf_counter() - t) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=0, help="batasi jumlah gambar (0 = semua)")
    ap.add_argument("--k", type=int, nargs="*", default=[2000, 5000, 10000])
    ap.add_argument("--precisions", nargs="*", default=["fp32", "fp16"], choices=("fp32", "fp16"))
    ap.add_argument("--assert-config", action="store_true",
                    help="kode keluar 1 bila K dari config (gate.nms / analysis) tidak identik di semua gambar")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = False

    g_nms, an = cfg["gate"]["nms"], cfg.get("analysis", {})
    conf, iou = float(g_nms["conf"]), float(g_nms["iou"])
    assert (float(an.get("conf", conf)), float(an.get("iou", iou))) == (conf, iou), "conf/iou gate.nms != analysis"
    max_dets = sorted({int(g_nms["max_det"]), int(an.get("max_det", g_nms["max_det"]))})
    k_cfg = sorted({int(v) for v in (g_nms.get("fast_nms_k"), an.get("fast_nms_k")) if v})
    ks = sorted(set(a.k) | set(k_cfg))
    if a.assert_config:
        assert k_cfg, "--assert-config: fast_nms_k tidak ada di gate.nms maupun analysis"
    print(f"max_det {max_dets} | K {ks} (config: {k_cfg or '-'}) | conf {conf} iou {iou} | {device}")

    m32, _ = load_detector(a.weights, device)
    models = {"fp32": m32}
    if "fp16" in a.precisions:
        m16 = reset_anchor_cache(copy.deepcopy(m32).half())
        models["fp16"] = m16
    models = {p: models[p] for p in a.precisions}
    data = load_data(cfg["data"], cfg.get("expected_counts"))
    loader = make_loader(data, a.split, 1, int(cfg["train"]["imgsz"]), a.workers)

    rows, cand_rows = [], []
    dets = {}                          # (prec, branch, md, K atau 'penuh') -> list deteksi per gambar (hanya bila perlu)
    ids = []
    with torch.no_grad():
        for i, bt in enumerate(loader):
            if a.n and i >= a.n:
                break
            rp, osh = bt["ratio_pad"][0], bt["ori_shape"][0]
            ids.append(image_id_of(bt["im_file"][0]))
            for prec, m in models.items():
                im = bt["img"].to(device).float() / 255.0
                im = im.half() if prec == "fp16" else im
                p3, p2, p4, p5 = m.trunk(im)
                outs = {"A": BranchA(m)(p3, p2, p4, p5), "B": BranchB(m)(p3, p4, p5)}
                for br, y in outs.items():
                    y = y.float()
                    n_c = int((y[0, 4:] > conf).sum())
                    cand_rows.append(dict(presisi=prec, branch=br, gambar=i, kandidat=n_c))
                    for md in max_dets:
                        ref, t_ref = timed(lambda: to_dets(y, rp, osh, conf, iou, md))
                        dets.setdefault((prec, br, md, "penuh"), []).append(ref)
                        for k in ks:
                            before = dict(D.NMS_STATS)
                            got, t_k = timed(lambda: to_dets(y, rp, osh, conf, iou, md, fast_nms_k=k))
                            path = next(p for p in ("penuh_langsung", "cepat", "fallback")
                                        if D.NMS_STATS[p] != before[p])
                            same = ref.shape == got.shape and np.array_equal(ref, got)
                            rows.append(dict(presisi=prec, branch=br, max_det=md, K=k, gambar=i, kandidat=n_c,
                                             jalur=path, identik=same, ms_penuh=t_ref, ms_cepat=t_k))
                            dets.setdefault((prec, br, md, k), []).append(got)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1} gambar", flush=True)

    R = pd.DataFrame(rows)
    C = pd.DataFrame(cand_rows)
    C.to_csv(out / "kandidat.csv", index=False)
    kand = C.groupby(["presisi", "branch"])["kandidat"].describe(percentiles=[0.5, 0.9, 0.99])
    print("\nJumlah kandidat NMS per gambar (skor kelas > conf):\n" + kand.round(0).to_string())

    S = (R.groupby(["presisi", "branch", "max_det", "K"])
         .agg(n=("identik", "size"), identik=("identik", "mean"),
              penuh_langsung=("jalur", lambda s: (s == "penuh_langsung").mean()),
              cepat=("jalur", lambda s: (s == "cepat").mean()), fallback=("jalur", lambda s: (s == "fallback").mean()),
              ms_penuh=("ms_penuh", "mean"), ms_cepat=("ms_cepat", "mean"))
         .reset_index())
    S["percepatan"] = S["ms_penuh"] / S["ms_cepat"]
    S.to_csv(out / "ringkasan.csv", index=False, float_format="%.6g")
    print("\n" + S.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # Selisih metrik untuk kombinasi yang tidak identik 100%
    bad = S[S["identik"] < 1.0]
    if len(bad):
        gt = build_coco_gt(loader.dataset, data["names"], out / f"gt_{a.split}.json")
        if a.n:
            sub = set(ids)
            gt.dataset["images"] = [im for im in gt.dataset["images"] if im["id"] in sub]
            gt.dataset["annotations"] = [an_ for an_ in gt.dataset["annotations"] if an_["image_id"] in sub]
            gt.createIndex()
        mdets = tuple(cfg["eval"]["coco_max_dets"])
        md_rows = []
        for _, r in bad.iterrows():
            key = (r["presisi"], r["branch"], int(r["max_det"]))
            s_ref = coco_eval(gt, jdict(dets[(*key, "penuh")], ids), mdets)[0]
            s_k = coco_eval(gt, jdict(dets[(*key, int(r["K"]))], ids), mdets)[0]
            md_rows.append(dict(presisi=key[0], branch=key[1], max_det=key[2], K=int(r["K"]),
                                **{f"d_{m}": s_k[m] - s_ref[m] for m in ("mAP50-95", "mAP50", "mAP_small", "AR_small")}))
        M = pd.DataFrame(md_rows)
        M.to_csv(out / "selisih_metrik.csv", index=False, float_format="%.6g")
        print("\nSelisih metrik (cepat − penuh) pada kombinasi yang tidak identik:\n" + M.to_string(index=False))

    write_json(out / "provenance.json", provenance(files=[cfg["_config_path"], a.weights],
                                                   extra=dict(split=a.split, n=a.n, ks=ks, k_config=k_cfg,
                                                              max_dets=max_dets, precisions=a.precisions)))
    if a.assert_config:
        chk = S[S["K"].isin(k_cfg)]
        ok = bool((chk["identik"] == 1.0).all())
        print(f"\n{'✅' if ok else '❌'} K config {k_cfg}: identik di semua gambar/branch/presisi/max_det = {ok}")
        sys.exit(0 if ok else 1)
    print("✅", out)


if __name__ == "__main__":
    main()
