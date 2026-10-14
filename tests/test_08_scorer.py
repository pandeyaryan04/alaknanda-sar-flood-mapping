"""Brief 9.8: the real, unchanged 04_stratified_metrics.py scores a model map."""
from conftest import FYP, run, run_cli

import models_common as mc


def test_scorer_runs_on_model_map(trained):
    p = mc.Paths(trained)
    out = p.flood_map(2021, "rf", "hand")
    if not out.exists():
        assert run("25_predict_corridor.py", ["--project", trained, "--model", "rf",
                                              "--trainset", "hand", "--year", "2021"]) == 0
    tables = p.tables / "2021_rf_hand"
    r = run_cli([FYP / "04_stratified_metrics.py", "--pred", out, "--truth", p.reference(2021),
                 "--strata", p.strata(2021), "--windows", p.windows(2021),
                 "--label", "Table A, rf hand, 2021", "--out", tables])
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tables / "metrics_by_stratum.csv").exists()
    assert (tables / "metrics_summary.md").exists()
    assert "gorge_above_devprayag" in (tables / "metrics_by_stratum.csv").read_text()
