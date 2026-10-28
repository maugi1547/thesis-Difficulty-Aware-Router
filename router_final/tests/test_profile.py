"""rfinal.profile: Timer (CPU), summarize, derive — tanpa GPU."""
import numpy as np

from rfinal.profile import SEGMENTS, Timer, derive, profile_pass, summarize


def test_timer_and_derive():
    import time

    def op(dur):
        def f(mark, s):
            mark(); time.sleep(dur[0] / 1e3)
            mark(); time.sleep(dur[1] / 1e3)
            mark(); time.sleep(dur[2] / 1e3)
            mark()
        return f
    ops = {"Vanilla": op((1, 4, 1)), "A3 rute B": op((1, 4, 3))}

    class Im:
        def float(self):
            return np.zeros(1)
    H = {"img": type("B", (), {"copy_": lambda self, x: None})()}
    rec = profile_pass(ops, H, [Im(), Im()], Timer(False, None), n_warmup=1, n_reps=2)
    assert set(rec) == set(ops) and all(len(rec[k]["wall"]) == 4 for k in ops)
    seg = summarize({"b0": rec, "b1": rec})
    assert set(seg.segmen) == set(SEGMENTS)
    d = derive(seg).set_index("selisih")
    r = d.loc["rute B − Vanilla"]
    assert 1.5 < r["d2h"] < 3.5 and abs(r["exec"]) < 1.0 and abs(r["wall"] - (r["h2d"] + r["exec"] + r["d2h"] + r["host"])) < 1e-9
