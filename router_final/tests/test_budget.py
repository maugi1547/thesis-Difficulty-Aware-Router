import numpy as np

from rfinal.budget import B2Controller, b2_decide, thr0_from_val


def _ref_b2(scores, rho, window, min_hist, thr0):
    """Salinan apa adanya eksperimen/tahap4_final.py::b2_thresholds + b2_decide."""
    s = np.asarray(scores, np.float64); out = np.empty(len(s), np.float32)
    for t in range(len(s)):
        hist = s[max(0, t - window):t]
        out[t] = np.float32(np.quantile(hist, 1 - rho)) if len(hist) >= min_hist else np.float32(thr0)
    return np.asarray(scores, np.float32) > out, out


def test_b2_matches_reference_and_streaming():
    rng = np.random.default_rng(0)
    s = np.concatenate([rng.normal(0, 1, 300), rng.normal(0.7, 1.3, 400)]).astype(np.float32).astype(np.float64)
    thr0 = thr0_from_val(rng.normal(0, 1, 500), 0.5)
    d, thr = b2_decide(s, 0.5, 200, 20, thr0)
    dr, thrr = _ref_b2(s, 0.5, 200, 20, thr0)
    assert np.array_equal(d, dr) and np.array_equal(thr, thrr)
    ctl = B2Controller(0.5, 200, 20, thr0)
    st = [ctl.decide(x) for x in s]
    assert np.array_equal(np.array([u for u, _ in st]), d) and np.array_equal(np.array([t for _, t in st]), thr)
    assert abs(d[200:].mean() - 0.5) < 0.05
