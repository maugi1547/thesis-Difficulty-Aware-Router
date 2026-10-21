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

import bisect
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


class SortedWindowQuantile:
    """Kuantil jendela bergulir via daftar terurut (bisect). Nilai IDENTIK dengan np.quantile (interpolasi linear)
    atas jendela yang sama, tanpa membuat array baru per gambar (= tahap4b.SortedWindowQuantile)."""

    def __init__(self, window):
        self.window, self.h, self.s = int(window), collections.deque(), []

    def __len__(self):
        return len(self.h)

    def push(self, x):
        if len(self.h) == self.window:
            old = self.h.popleft(); del self.s[bisect.bisect_left(self.s, old)]
        self.h.append(float(x)); bisect.insort(self.s, float(x))

    def quantile(self, q):
        n = len(self.s); pos = q * (n - 1); lo = int(pos); fr = pos - lo
        a, b = self.s[lo], self.s[min(lo + 1, n - 1)]
        # rumus lerp numpy (np.quantile 'linear'): a + d·t bila t < 0,5, selain itu b − d·(1 − t) -> bit-identik
        return a + (b - a) * fr if fr < 0.5 else b - (b - a) * (1 - fr)


class B2Controller:
    """
    Runtime: thr = controller.threshold() -> jalankan engine -> controller.push(skor).
    efficient=True (default deployment): jendela terurut; False: np.quantile atas deque (implementasi Tahap 4).
    Keputusan kedua mode identik (tests/test_budget.py); hanya overhead host yang berbeda.
    """

    def __init__(self, rho, window, min_hist, thr0, efficient=True):
        self.rho, self.min_hist, self.thr0, self.efficient = float(rho), int(min_hist), float(thr0), bool(efficient)
        self.hist = SortedWindowQuantile(window) if self.efficient else collections.deque(maxlen=int(window))

    def threshold(self):
        if len(self.hist) >= self.min_hist:
            if self.efficient:
                return np.float32(self.hist.quantile(1 - self.rho))
            return np.float32(np.quantile(np.asarray(self.hist, np.float64), 1 - self.rho))
        return np.float32(self.thr0)

    def push(self, score):
        if self.efficient:
            self.hist.push(score)
        else:
            self.hist.append(float(score))

    def decide(self, score):
        """Untuk evaluasi offline per gambar: keputusan dengan threshold saat ini, lalu skor masuk riwayat."""
        thr = self.threshold()
        use_a = bool(np.float32(score) > thr)
        self.push(score)
        return use_a, thr


def b2_host_overhead(window, rho, n=4000):
    """ms per gambar: (np.quantile atas deque = Tahap 4) vs jendela terurut; nilai keduanya dicek identik."""
    import time
    xs = np.random.default_rng(0).standard_normal(n + window)
    h = collections.deque(xs[:window], maxlen=window)
    t0 = time.perf_counter()
    for i in range(n):
        np.quantile(np.asarray(h, np.float64), 1 - rho); h.append(xs[window + i])
    t_np = (time.perf_counter() - t0) / n * 1e3
    sw = SortedWindowQuantile(window)
    for x in xs[:window]:
        sw.push(float(x))
    h = collections.deque(xs[:window], maxlen=window)
    for i in range(200):
        assert abs(sw.quantile(1 - rho) - np.quantile(np.asarray(h, np.float64), 1 - rho)) < 1e-12
        sw.push(float(xs[window + i])); h.append(xs[window + i])
    t0 = time.perf_counter()
    for i in range(n):
        sw.quantile(1 - rho); sw.push(float(xs[i]))
    return t_np, (time.perf_counter() - t0) / n * 1e3
