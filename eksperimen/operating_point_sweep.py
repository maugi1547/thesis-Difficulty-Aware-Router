"""
operating_point_sweep.py — Pemilihan titik operasi router secara post-hoc (tanpa router_target_activation)

Keputusan 2026-10-09: training memakai router_hybrid_alpha = 0 (suku target mati); sparsity
bekerja sebagai "harga" P2, c = lambda * Diff_W / gate_value_weight. Activation rate untuk
deployment dipilih di sini, dari skor gate model BEKU, dengan dua aturan yang dikunci di VAL:
  (a) r* = argmax_r [ mAP50_router(r) - mAP50_acak(r) ]   (keunggulan atas routing acak)
  (b) oracle rate = fraksi gambar val dengan loss_A < loss_B, + AUC skor gate vs oracle
Threshold selalu dari kuantil logit VAL; test hanya untuk melaporkan kurva.

Tempel sebagai sel-sel tambahan di uji-akurasi-latensi-model-dualbranch_diperbaiki.ipynb SETELAH
sel yang mendefinisikan: DEVICE, model, router, stage1, stage2a, stage2b, gate_logit_raw,
collect_calibration_data, make_loader, run_eval, summarize_acc, set_calibration, cal_loader, test_loader.
T_A_ROUTE / T_B_ROUTE = latensi CHAINED (ms) rute paksa A dan paksa B dari notebook yang sama.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def cache_branches(loader):
    """Satu forward per gambar: logit gate mentah + output Stage2A & Stage2B (disimpan di CPU)."""
    out = []
    for batch in loader:
        img = batch["img"].to(DEVICE).float() / 255.0
        for i, f in enumerate(batch["im_file"]):
            _, p3, p2, p4, p5 = stage1(img[i:i + 1])
            out.append(dict(file=Path(f).name, raw=float(gate_logit_raw(router, p3, p2)),
                            A=stage2a(p3, p2, p4, p5).cpu(), B=stage2b(p3, p4, p5).cpu()))
    return out


def predict_from_cache(cache, use_a):
    """predict_fn untuk run_eval; urutan loader (shuffle=False) harus = urutan cache (dicek di eval_mask)."""
    pos = {"k": 0}

    def fn(im):
        preds, infos = [], []
        for _ in range(im.shape[0]):
            k = pos["k"]; pos["k"] += 1
            a = bool(use_a[k])
            preds.append((cache[k]["A"] if a else cache[k]["B"]).to(DEVICE))
            infos.append({"branch": "A" if a else "B"})
        return preds, infos
    return fn


def eval_mask(cache, loader, use_a, name):
    r = run_eval(name, predict_from_cache(cache, use_a), loader)
    assert [x["file"] for x in r["records"]] == [c["file"] for c in cache], "Urutan loader != cache"
    return summarize_acc(r)


def sweep(cache, loader, thr_source, rates, n_random=3, seed=0):
    """thr_source: logit mentah VAL (threshold selalu dari val, juga saat mengevaluasi test)."""
    raw = np.array([c["raw"] for c in cache]); n = len(raw)
    rng = np.random.default_rng(seed); rows = []
    for r in rates:
        thr = -np.inf if r >= 1 else (np.inf if r <= 0 else np.quantile(thr_source, 1 - r))
        use_a = raw > thr
        k = int(use_a.sum())
        acc = eval_mask(cache, loader, use_a, f"router r={r:.2f}")
        rnd = []
        for _ in range(n_random if 0 < k < n else 0):
            ua = np.zeros(n, bool); ua[rng.choice(n, k, replace=False)] = True
            rnd.append(eval_mask(cache, loader, ua, f"acak r={k / n:.2f}")["mAP50"])
        rows.append(dict(rate_target=r, rate_realized=k / n,
                         calib_b=float(-thr) if np.isfinite(thr) else None,
                         mAP50=acc["mAP50"], mAP50_95=acc["mAP50-95"],
                         acak_mAP50=np.mean(rnd) if rnd else acc["mAP50"],
                         acak_sd=np.std(rnd) if rnd else 0.0))
    df = pd.DataFrame(rows)
    df["gain_vs_acak"] = df.mAP50 - df.acak_mAP50
    return df


# ------------------------------- eksekusi -------------------------------
RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
OUT_SWEEP = OUT / "operating_point"; OUT_SWEEP.mkdir(parents=True, exist_ok=True)

cal_loader_b1 = make_loader(CFG["calib_split"], 1)        # val, batch 1, shuffle=False
cache_val = cache_branches(cal_loader_b1)
raw_val = np.array([c["raw"] for c in cache_val])
sat = float((np.abs(raw_val) > 9.5).mean())
print(f"Logit val: mean={raw_val.mean():.3f} std={raw_val.std():.3f} | tersaturasi (|raw|>9.5) = {sat:.1%}")
assert raw_val.std() > 0.05, "Logit gate hampir konstan -> ranking seri, kalibrasi persentil tidak bermakna."

# (b) oracle rate + kualitas ranking (model beku, val)
raw_cal, tgt_cal = collect_calibration_data(model, cal_loader)        # tgt = 1 bila loss_A < loss_B
ORACLE = dict(oracle_rate=float(tgt_cal.mean()),
              auc_gate_vs_oracle=float(roc_auc_score(tgt_cal.numpy(), raw_cal.numpy())))
print(ORACLE)

# (a) pilih r* di VAL, kunci, baru evaluasi TEST
df_val = sweep(cache_val, cal_loader_b1, raw_val, RATES)
r_star = float(df_val.loc[df_val.gain_vs_acak.idxmax(), "rate_target"])
print(df_val.round(4).to_string(index=False)); print("r* (val) =", r_star)
df_val.to_csv(OUT_SWEEP / "sweep_val.csv", index=False)

cache_test = cache_branches(test_loader)
df_test = sweep(cache_test, test_loader, raw_val, RATES)              # threshold tetap dari val
df_test["latensi_ms"] = df_test.rate_realized * T_A_ROUTE + (1 - df_test.rate_realized) * T_B_ROUTE
print(df_test.round(4).to_string(index=False))
df_test.to_csv(OUT_SWEEP / "sweep_test.csv", index=False)

b_star = float(-np.quantile(raw_val, 1 - r_star))
set_calibration(router, 1.0, b_star)                                   # operating point untuk export
(OUT_SWEEP / "operating_point.json").write_text(pd.Series(
    dict(r_star=r_star, calib_T=1.0, calib_b=b_star, **ORACLE)).to_json(indent=1))
