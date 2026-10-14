"""Brief 9.12: the whole chain, through the very commands RUN MODELS SMOKE.bat runs."""
import shlex
import shutil

import rasterio

from conftest import FYP, SCRIPTS, make_project, run_cli

import models_common as mc


def bat_commands(bat):
    cmds = []
    for line in (FYP / bat).read_text(encoding="ascii").splitlines():
        s = line.strip()
        if s.lower().startswith("python "):
            cmds.append(shlex.split(s.replace("\\", "/"))[1:])
    return cmds


def test_smoke_chain(tmp_path):
    root = make_project(tmp_path / "smoke run", prepare=False)
    code = root / "04 Code" / "FYP_code"
    for s in SCRIPTS:
        shutil.copy2(FYP / s, code / s)
    cmds = bat_commands("RUN MODELS SMOKE.bat")
    assert len(cmds) >= 12
    for args in cmds:
        r = run_cli(args, cwd=code)
        assert r.returncode == 0, f"{' '.join(args)}\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}"
    p = mc.Paths(root, smoke=True)
    assert p.chip_index.exists() and (p.model_dir("rf", "hand") / "model.joblib").exists()
    assert (p.model_dir("unet", "hand") / "best.pt").exists()
    for m in ("rf", "unet"):
        with rasterio.open(p.flood_map(2021, m, "hand")) as d:
            assert d.dtypes[0] == "uint8" and d.nodata == 255
    assert (p.tables / "2021_rf_hand" / "metrics_by_stratum.csv").exists()
    assert (p.tables / "MODELS_SUMMARY.md").exists()
    # nothing of the smoke run lands in the real trees
    assert not (root / "04 Code" / "FYP_code" / "data" / "models").exists()
    assert not (root / "06 Outputs" / "tables").exists()
