"""
Buat berkas acuan dari model M0 LAMA, untuk tests/test_m0_equivalence.py.

HARUS dijalankan dengan fork ultralytics lama (root repo, branch eksperimen-uji) di sys.path — berbeda dari
semua skrip router_final lain. Contoh (dari root repo):

    python router_final/tools/m0_reference.py --ckpt last.pt --data kitti_2class.yaml --out m0_ref.pt --n 8

Isi berkas: state_dict M0 (FP32), keluaran eval Branch A/B + fitur trunk untuk n gambar test (letterbox 640,
urutan file), keluaran Detect mode training + item loss lama [box, cls, dfl] per branch untuk satu batch train.
"""
import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()

    import ultralytics
    assert (Path(ultralytics.__file__).parent / "model_yolov8_router.py").exists(), "harus memakai fork lama"
    from torch.utils.data import DataLoader
    from ultralytics.cfg import get_cfg
    from ultralytics.data import YOLODataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.model_yolov8_router import DualBranchDetectionModel
    from ultralytics.utils import DEFAULT_CFG

    data = check_det_dataset(a.data)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    src = ck["ema"] if ck.get("ema") is not None else ck["model"]
    sd = {k: (v.float() if v.is_floating_point() else v) for k, v in src.state_dict().items()}
    m = DualBranchDetectionModel(cfg=str(REPO / "ultralytics/cfg/models/v8/yolov8n-p2-router.yaml"
                                         if (REPO / "ultralytics/cfg/models/v8/yolov8n-p2-router.yaml").exists()
                                         else "yolov8n-p2-router.yaml"), ch=3, nc=data["nc"], verbose=False)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not [k for k in missing if "calib" not in k] and not unexpected, (missing, unexpected)
    m.args = get_cfg(DEFAULT_CFG)
    m.eval()

    def loader(split, batch):
        ds = YOLODataset(img_path=data[split], data=data, task="detect", imgsz=640, augment=False, rect=False,
                         cache=False, batch_size=batch)
        return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0, collate_fn=ds.collate_fn)

    ref = dict(state_dict=sd, eval=[], files=[])
    torch.set_grad_enabled(False)
    for i, b in enumerate(loader("test", 1)):
        if i >= a.n:
            break
        x = b["img"].float() / 255.0
        (yA, _), (yB, _) = m._predict_once_dual(x)
        ml, y = list(m.model), []
        h = x
        for mod in ml[:16]:
            if mod.f != -1:
                h = y[mod.f] if isinstance(mod.f, int) else [h if j == -1 else y[j] for j in mod.f]
            h = mod(h); y.append(h)
        ref["eval"].append(dict(x=x, yA=yA, yB=yB, p3=h, p2=y[2], p4=y[12], p5=y[9]))
        ref["files"].append(Path(b["im_file"][0]).name)

    # Loss lama pada keluaran Detect mode training (Detect.training=True, BN tetap eval agar deterministik)
    b = next(iter(loader("train", 4)))
    m.detect_A.training = m.detect_B.training = True
    x = b["img"].float() / 255.0
    det_A, det_B = m._predict_once_dual(x)
    crit = m.init_criterion()
    batch = {k: b[k] for k in ("batch_idx", "cls", "bboxes")}
    _, items_A = crit.loss_A(det_A, batch)
    _, items_B = crit.loss_B(det_B, batch)
    ref["loss"] = dict(x=x, batch=batch, det_A=det_A, det_B=det_B, items_A=items_A[:3], items_B=items_B[:3])
    ref["meta"] = dict(ckpt=str(a.ckpt), epoch=ck.get("epoch"), ultralytics_file=ultralytics.__file__,
                       torch=torch.__version__)
    torch.save(ref, a.out)
    print(f"✅ {a.out}: {len(ref['eval'])} gambar test, loss batch {tuple(x.shape)}")


if __name__ == "__main__":
    main()
