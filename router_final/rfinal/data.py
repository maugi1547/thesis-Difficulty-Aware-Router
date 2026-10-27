"""
Data & evaluasi COCO (router_final). Diporting tanpa perubahan semantik dari eksperimen/tahap2_eval.py dan
tahap2b_plafon.py (versi yang menghasilkan angka tesis).

Format evaluasi = format deployment: letterbox persegi 640, batch 1, tanpa augmentasi, urutan file tetap.
Deteksi: NMS conf 0.001, IoU 0.7, max_det 300, dikonversi ke piksel ASLI per sumbu; GT COCO resolusi asli
(category_id = cls + 1, image_id = int(stem)), COCOeval maxDets [1, 10, 100].
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import torch

COCO_KEYS = ["mAP50-95", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large",
             "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
COCO_MAX_DETS = (1, 10, 100)
SMALL_PX = 32.0


def load_data(data_yaml, expected_counts=None):
    from ultralytics.data.utils import check_det_dataset
    data = check_det_dataset(str(data_yaml))
    data["_expected_counts"] = dict(expected_counts or {})
    return data


def make_loader(data, split, batch=1, imgsz=640, workers=2):
    """
    DataLoader biasa (BUKAN InfiniteDataLoader Ultralytics: iterator persistennya menggeser urutan bila loop
    di-`break`, sehingga hasil per gambar tidak sejajar lagi dengan dataset.labels).
    """
    from torch.utils.data import DataLoader
    from ultralytics.data import YOLODataset
    ds = YOLODataset(img_path=data[split], data=data, task="detect", imgsz=imgsz, augment=False, rect=False,
                     cache=False, batch_size=batch)
    exp = data.get("_expected_counts", {}).get(split)
    assert exp is None or len(ds) == exp, f"split {split}: {len(ds)} gambar, diharapkan {exp}"
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, collate_fn=ds.collate_fn,
                      pin_memory=torch.cuda.is_available(), drop_last=False)


def image_id_of(im_file):
    stem = Path(im_file).stem
    return int(stem) if stem.isnumeric() else stem


def small_area_of(ratio_pad):
    """Batas 'kecil' COCO (32² px asli) dalam piksel letterbox²."""
    (gh, gw), _ = ratio_pad
    return SMALL_PX ** 2 * float(gh) * float(gw)


def _scale_xyxy(b, ratio_pad, ori_shape):
    (gh, gw), (px, py) = ratio_pad
    b = b.clone()
    b[:, [0, 2]] = ((b[:, [0, 2]] - px) / gw).clamp(0, ori_shape[1])
    b[:, [1, 3]] = ((b[:, [1, 3]] - py) / gh).clamp(0, ori_shape[0])
    return b


NMS_STATS = {"panggilan": 0, "penuh_langsung": 0, "cepat": 0, "fallback": 0}


def nms_one(y, conf=0.001, iou=0.7, max_det=300, fast_nms_k=None):
    """
    NMS Ultralytics stok untuk satu gambar (multi-label, class-aware, max_nms 30000) -> tensor (n, 6) xyxy, skor, cls.

    fast_nms_k = K (opsional; None = perilaku lama persis): NMS dijalankan dulu hanya pada K kandidat berskor tertinggi
    (`max_nms=K`), dan hasilnya DIPAKAI hanya bila terbukti identik dengan NMS penuh:
      NMS greedy hanya menekan sebuah kotak dengan kotak yang berskor lebih tinggi, sehingga kotak yang selamat di
      antara K teratas sama persis dengan versi penuh. Bila ada >= max_det kotak selamat dan skor kotak ke-max_det
      LEBIH BESAR (ketat) dari skor kandidat ke-K, semua kandidat yang dibuang berskor lebih rendah daripada seluruh
      keluaran: tidak bisa menekan kotak keluaran dan tidak bisa masuk max_det teratas -> keluaran identik.
    Bila syarat itu tidak terpenuhi -> NMS penuh (protokol asli). Satu-satunya celah: skor yang PERSIS sama (tie)
    di antara kandidat yang bertumpuk bisa diproses dalam urutan berbeda; frekuensinya diukur scripts/check_fast_nms.py.
    """
    from ultralytics.utils.nms import non_max_suppression
    NMS_STATS["panggilan"] += 1
    full = lambda: non_max_suppression(y.clone(), conf, iou, nc=0, multi_label=True, agnostic=False,
                                       max_det=max_det)[0]
    if not fast_nms_k:
        return full()
    k = int(fast_nms_k)
    sc = y[0, 4:]                                    # (nc, N); kandidat multi-label Ultralytics = skor kelas > conf
    cand = sc[sc > conf]
    if cand.numel() <= k:                            # tidak ada yang dibuang -> sama dengan penuh
        NMS_STATS["penuh_langsung"] += 1
        return full()
    kth = torch.topk(cand, k).values[-1]
    x = non_max_suppression(y.clone(), conf, iou, nc=0, multi_label=True, agnostic=False, max_det=max_det,
                            max_nms=k)[0]
    if len(x) == max_det and bool(x[max_det - 1, 4] > kth):
        NMS_STATS["cepat"] += 1
        return x
    NMS_STATS["fallback"] += 1
    return full()


def to_dets(y, ratio_pad, ori_shape, conf=0.001, iou=0.7, max_det=300, fast_nms_k=None):
    """y (1, 4+nc, N) -> array (n, 6): x, y, w, h (COCO, piksel asli), skor, category_id (= cls + 1)."""
    x = nms_one(y, conf, iou, max_det, fast_nms_k)
    if not len(x):
        return np.zeros((0, 6), np.float32)
    b = _scale_xyxy(x[:, :4], ratio_pad, ori_shape)
    out = torch.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1], x[:, 4], x[:, 5] + 1], 1)
    return out.float().cpu().numpy()


def jdict(dets, image_ids):
    return [{"image_id": iid, "category_id": int(c), "bbox": [float(x), float(y), float(w), float(h)], "score": float(s)}
            for d, iid in zip(dets, image_ids) for x, y, w, h, s, c in d]


def build_coco_gt(dataset, names, path):
    """GT COCO resolusi asli; category_id = cls + 1, image_id = int(stem) (= pred_to_json Ultralytics)."""
    from pycocotools.coco import COCO
    images, anns, aid = [], [], 1
    for lab in dataset.labels:
        iid = image_id_of(lab["im_file"])
        h, w = lab["shape"]
        images.append({"id": iid, "file_name": Path(lab["im_file"]).name, "height": int(h), "width": int(w)})
        for c, (cx, cy, bw, bh) in zip(lab["cls"].reshape(-1), lab["bboxes"].reshape(-1, 4)):
            bw_px, bh_px = float(bw * w), float(bh * h)
            anns.append({"id": aid, "image_id": iid, "category_id": int(c) + 1,
                         "bbox": [float(cx * w - bw_px / 2), float(cy * h - bh_px / 2), bw_px, bh_px],
                         "area": bw_px * bh_px, "iscrowd": 0})
            aid += 1
    gt = {"images": images, "annotations": anns, "categories": [{"id": i + 1, "name": str(n)} for i, n in names.items()]}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(gt))
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(path))


def coco_eval(coco_gt, js, max_dets=COCO_MAX_DETS):
    """COCOeval lengkap. Returns (stats dict, objek COCOeval — evalImgs dipakai target gate & routing)."""
    from pycocotools.cocoeval import COCOeval
    ids = set(coco_gt.getImgIds())
    assert {d["image_id"] for d in js} <= ids, "image_id prediksi tidak ada di GT"
    with contextlib.redirect_stdout(io.StringIO()):
        if len(js):
            dt = coco_gt.loadRes(js)
        else:
            from pycocotools.coco import COCO
            dt = COCO(); dt.dataset = {"images": coco_gt.dataset["images"], "categories": coco_gt.dataset["categories"],
                                      "annotations": []}
            dt.createIndex()
        E = COCOeval(coco_gt, dt, "bbox")
        E.params.imgIds = sorted(ids)
        E.params.maxDets = list(max_dets)
        E.evaluate(); E.accumulate(); E.summarize()
    # pycocotools 2.0.x: stats[0] memakai maxDets=100 tertanam; hitung ulang dari maxDets[-1], area 'all'
    prec = E.eval["precision"][:, :, :, 0, -1]
    E.stats[0] = float(prec[prec > -1].mean()) if (prec > -1).any() else -1.0
    return dict(zip(COCO_KEYS, map(float, E.stats))), E


def per_image_soft_small(E, image_ids):
    """
    soft-recall objek kecil per gambar dari evalImgs: Σ skor deteksi TP (IoU 0.5, tidak di-ignore, maxDet terakhir)
    pada area 'small', dijumlah atas kategori. Urutan keluaran = urutan `image_ids`. Juga jumlah GT kecil.
    (= P.Router.targets()['soft_small'] di eksperimen/tahap2b_plafon.py)
    """
    p = E.params
    A, I, K, maxDet = len(p.areaRng), len(p.imgIds), len(p.catIds), p.maxDets[-1]
    a_small = [tuple(r) for r in p.areaRng].index(tuple(p.areaRng[p.areaRngLbl.index("small")]))
    t50 = int(np.where(np.isclose(p.iouThrs, 0.5))[0][0])
    soft, n_gt = np.zeros(I), np.zeros(I)
    for k in range(K):
        base = k * A * I + a_small * I
        for i in range(I):
            e = E.evalImgs[base + i]
            if e is None:
                continue
            sc = np.asarray(e["dtScores"][:maxDet], float)
            tp = (np.asarray(e["dtMatches"][:, :maxDet]) > 0)[t50] & ~np.asarray(e["dtIgnore"][:, :maxDet], bool)[t50]
            soft[i] += sc[tp].sum()
            n_gt[i] += int(np.count_nonzero(np.asarray(e["gtIgnore"]) == 0))
    pos = {iid: j for j, iid in enumerate(p.imgIds)}
    perm = np.array([pos[i] for i in image_ids])
    return soft[perm], n_gt[perm]
