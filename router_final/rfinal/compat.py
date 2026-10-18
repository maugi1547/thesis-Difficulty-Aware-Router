"""
Pemetaan bobot M0 (DualBranchDetectionModel lama, yolov8-p2-router.yaml) -> DualBranchDetector (arsitektur final).

Hanya dipakai untuk UJI KESETARAAN graf (tests/test_m0_equivalence.py): bila bobot M0 dimuat ke graf final dan
keluaran Branch A/B identik dengan model lama, berarti graf final = jaringan deteksi yang sama, dan yang dibuang
(gate terlatih, cabang proxy, kalibrasi) memang tidak ikut menghitung deteksi.

Lama (layer 16 = UltraLightWeightDifficultyAwareRouter)         Final
  model.0–15.*                                                   model.0–15.*        (sama)
  model.16.c2f_p2.*                (expert P2)                   model.18.*          (C2f stok; 16 = Upsample, 17 = Concat)
  model.16.{squeeze_*, classifier, proxy_*, tau_param, _calib_*} —                   (dibuang)
  model.17–33.*                                                  model.19–35.*       (indeks + 2)
"""
from __future__ import annotations

import re

DROPPED_ROUTER_PREFIXES = ("squeeze_p3", "squeeze_p2", "classifier", "proxy_stem", "proxy_cls", "proxy_reg_dist",
                           "tau_param", "_calib_T_micro", "_calib_b_micro")


def map_m0_state_dict(sd_old):
    """Returns (sd_baru, daftar kunci yang dibuang). Gagal bila ada kunci layer 16 yang tidak dikenali."""
    out, dropped = {}, []
    for k, v in sd_old.items():
        if k.startswith(("detect_A.", "detect_B.")):          # alias atribut modul lama (duplikat model.32/33)
            continue
        m = re.match(r"model\.(\d+)\.(.*)", k)
        if not m:
            raise KeyError(f"kunci tak dikenal: {k}")
        i, rest = int(m.group(1)), m.group(2)
        if i <= 15:
            out[k] = v
        elif i == 16:
            if rest.startswith("c2f_p2."):
                out[f"model.18.{rest[len('c2f_p2.'):]}"] = v
            elif rest.startswith(DROPPED_ROUTER_PREFIXES):
                dropped.append(k)
            else:
                raise KeyError(f"kunci router tak dikenal: {k}")
        else:
            out[f"model.{i + 2}.{rest}"] = v
    return out, dropped
