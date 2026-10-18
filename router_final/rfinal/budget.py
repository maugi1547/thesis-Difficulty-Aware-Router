"""
Kendali anggaran B2 (router_final): threshold gate mengikuti distribusi skor terbaru, sehingga aktivasi Branch A
mendekati ρ walau distribusi skor bergeser dari val ke data deployment.

    thr_t = kuantil (1 − ρ) dari skor gambar t−window … t−1     (hanya skor MASA LALU)
    thr_t = thr0 = kuantil (1 − ρ) skor val                      bila riwayat < min_hist
    rute_t = A  bila  skor_t > thr_t  (perbandingan float32, sama dengan engine A3)

Identik dengan eksperimen/tahap4_final.py (b2_thresholds, b2_decide); B2Controller = versi streaming untuk
runtime per gambar (diuji sama persis dengan versi batch di tests/test_budget.py).
"""
from __future__ import annotations

import collections

import numpy as np


def b2_thresholds(scores, rho, window, min_hist, thr0):
    s = np.asarray(scores, np.float64)
    out = np.empty(len(s), np.float32)
    for t in range(len(s)):
        hist = s[max(0, t - window):t]
        out[t] = np.float32(np.quantile(hist, 1 - rho)) if len(hist) >= min_hist else np.float32(thr0)
    return out


def b2_decide(scores, rho, window, min_hist, thr0):
    thr = b2_thresholds(scores, rho, window, min_hist, thr0)
    return np.asarray(scores, np.float32) > thr, thr


def thr0_from_val(val_scores, rho):
    return float(np.quantile(np.asarray(val_scores, np.float64), 1 - rho))


class B2Controller:
    """Runtime: thr = controller.threshold() -> jalankan engine -> controller.push(skor)."""

    def __init__(self, rho, window, min_hist, thr0):
        self.rho, self.min_hist, self.thr0 = float(rho), int(min_hist), float(thr0)
        self.hist = collections.deque(maxlen=int(window))

    def threshold(self):
        if len(self.hist) >= self.min_hist:
            return np.float32(np.quantile(np.asarray(self.hist, np.float64), 1 - self.rho))
        return np.float32(self.thr0)

    def push(self, score):
        self.hist.append(float(score))

    def decide(self, score):
        """Untuk evaluasi offline per gambar: keputusan dengan threshold saat ini, lalu skor masuk riwayat."""
        thr = self.threshold()
        use_a = bool(np.float32(score) > thr)
        self.push(score)
        return use_a, thr
