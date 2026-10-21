"""best_epoch_from_results: fitness bawaan Ultralytics, seri -> epoch terakhir (perilaku best.pt)."""
import pandas as pd

from rfinal.ckcheck import best_epoch_from_results


def test_best_epoch(tmp_path):
    pd.DataFrame({"epoch": [1, 2, 3, 4], "metrics/mAP50(B)": [.5, .8, .7, .8],
                  "metrics/mAP50-95(B)": [.3, .5, .6, .5]}).to_csv(tmp_path / "results.csv", index=False)
    ep, df = best_epoch_from_results(tmp_path)
    assert ep == 3 and abs(df["fitness_A"].iloc[2] - (0.07 + 0.54)) < 1e-12
    pd.DataFrame({"epoch": [1, 2, 3], "metrics/mAP50(B)": [.5, .8, .8],
                  "metrics/mAP50-95(B)": [.3, .5, .5]}).to_csv(tmp_path / "results.csv", index=False)
    assert best_epoch_from_results(tmp_path)[0] == 3
