"""TerraMind-small on CPU, synthetic data.

The weights file here is a stand-in with random numbers in the real file's layout
(models_synthetic.build_terramind_weights); the code never downloads anything.
"""
import json

import numpy as np
import pytest
import rasterio
import torch

import models_common as mc
import models_synthetic as syn
from conftest import FYP, copy_project, make_project, run, run_cli

TM_FAST = ["--crop", "64", "--batch-size", "2", "--steps-per-epoch", "3", "--epochs", "2",
           "--overlap", "16", "--workers", "0", "--cpu-ok"]
LOWO_TM_FAST = ["--steps", "3", "--crop", "64", "--batch-size", "2", "--overlap", "16",
                "--cpu-ok"]
BACKBONE_PARAMS = 21_481_728         # TerraMind-small's encoder with one 2-band input
lowo = mc.load_script("26_corridor_lowo.py")


@pytest.fixture(autouse=True)
def _free_disk(tmp_path):
    """A TerraMind checkpoint is about 280 MB: delete a test's own once it has passed."""
    yield
    for f in tmp_path.rglob("*.pt"):
        f.unlink()


@pytest.fixture(scope="module")
def tm_project(tmp_path_factory):
    """Synthetic project, chips prepared, stand-in TerraMind weights."""
    root = make_project(tmp_path_factory.mktemp("tm data"))
    syn.build_terramind_weights(root)
    return root


@pytest.fixture(scope="module")
def tm_trained(tm_project, tmp_path_factory):
    """tm_project with tm_s1 and tm_s1t trained on 'hand' (2 epochs of 3 steps)."""
    root = copy_project(tm_project, tmp_path_factory.mktemp("tm trained") / "Final Year Project")
    for v in mc.TM_VARIANTS:
        assert run("30_train_terramind.py", ["--project", root, "--variant", v, "--trainset",
                                             "hand"] + TM_FAST) == 0
    return root


def _last(root, variant, trainset="hand"):
    return torch.load(mc.Paths(root).model_dir(variant, trainset) / "last.pt", weights_only=False)


def _valid_chip(root):
    p = mc.Paths(root)
    row = mc.split_rows(mc.load_chip_index(p), "valid")[0]
    return mc.read_chip(row, p.data)


# ----------------------------------------------------------------------------- 1. network

@pytest.mark.parametrize("variant", mc.TM_VARIANTS)
@pytest.mark.parametrize("modality", ["S1GRD", "S1RTC"])
def test_network_builds_with_random_weights(variant, modality, capsys):
    torch.manual_seed(0)
    net = mc.TerraMindNet(variant, modality)
    assert net.backbone.mod_name_mapping == {modality: mc.TM_KEYS[modality]}
    counts = net.parameter_counts()
    print(f"{variant} {modality}: backbone {counts['backbone']:,}, decoder {counts['decoder']:,}")
    assert counts["backbone"] == BACKBONE_PARAMS
    assert 1_000_000 < counts["decoder"] < 2_000_000
    with torch.no_grad():
        assert tuple(net(torch.randn(2, 4, 224, 224)).shape) == (2, 224, 224)
        assert tuple(net(torch.randn(1, 4, 96, 160)).shape) == (1, 96, 160)
    assert "backbone" in capsys.readouterr().out


def test_importing_terratorch_leaves_torch_settings_alone():
    r = run_cli(["-c", "import torch, models_common as mc; a = "
                       "torch.get_float32_matmul_precision(); mc.tm_import(); "
                       "print(a, torch.get_float32_matmul_precision())"])
    assert r.returncode == 0 and r.stdout.split() == ["highest", "highest"], r.stderr


def test_tokens_are_laid_out_row_by_row():
    """terratorch numbers the patches row by row; the decoder reshapes them that way."""
    net = mc.TerraMindNet("tm_s1", "S1GRD")
    emb = net.backbone.encoder_embeddings[mc.TM_KEYS["S1GRD"]]
    x = torch.zeros(1, 2, 64, 96)                     # 4 x 6 patches of 16
    x[0, :, 16:32, 48:64] = 1.0                       # patch row 1, column 3
    with torch.no_grad():
        tok = emb(x)["x"][0]
    hot = torch.nonzero(tok.abs().sum(dim=1) > 0).flatten().tolist()
    assert hot == [1 * 6 + 3]
    assert net.backbone.out_channels[mc.TM_LAYERS[-1]] == 384


def test_memory_estimate_under_limit_and_rng_untouched():
    torch.manual_seed(5)
    before = torch.get_rng_state()
    for v in mc.TM_VARIANTS:
        est = mc.terramind_memory_estimate(v, batch=8, crop=224, amp=True)
        assert est["estimate_gb"] < 3.3, est
        assert est["backbone_parameters"] == BACKBONE_PARAMS
    assert torch.equal(before, torch.get_rng_state())
    assert mc.terramind_memory_estimate("tm_s1t", 16, 224)["estimate_gb"] > \
        mc.terramind_memory_estimate("tm_s1t", 8, 224)["estimate_gb"]


def test_decoder_parameters_are_everything_but_the_encoder():
    net = mc.TerraMindNet("tm_s1t")
    enc = {id(p) for p in net.backbone.parameters()}
    dec = {id(p) for p in net.decoder_parameters()}
    assert not enc & dec and enc | dec == {id(p) for p in net.parameters()}
    assert any(id(p) in dec for p in net.stem.parameters())


# ----------------------------------------------------------------------------- 2. weights

def _checkpoint(path, net, drop=0, extra=True, bad_shape=None):
    sd = dict(net.backbone.state_dict())
    names = [n for n, _ in net.backbone.named_parameters()]
    for n in names[len(names) - drop:]:
        del sd[n]
    if extra:
        sd["decoder.0.attn.qkv.weight"] = torch.zeros(1152, 384)
        sd["encoder_embeddings.untok_sen2l2a@224.proj.weight"] = torch.zeros(384, 3072)
    if bad_shape:
        sd[bad_shape] = torch.zeros(3, 3)
    torch.save(sd, path)
    return sd


def test_weights_with_extra_keys_load_and_report(tmp_path):
    torch.manual_seed(1)
    src = mc.TerraMindNet("tm_s1", "S1GRD")
    f = tmp_path / "weights dir" / "TerraMind_v1_small.pt"
    f.parent.mkdir()
    _checkpoint(f, src)
    torch.manual_seed(2)
    dst = mc.TerraMindNet("tm_s1t", "S1GRD")
    take, rep = mc.tm_read_weights(f, dst.backbone)
    assert rep["filled_parameter_tensors"] == rep["backbone_parameter_tensors"] == 123
    assert rep["filled_share"] == 1.0 and rep["missing"] == [] and rep["wrong_shape"] == []
    assert rep["unused_tensors"] == 2 and rep["filled_values"] == BACKBONE_PARAMS
    mc.tm_apply_weights(dst.backbone, take)
    a, b = src.backbone.state_dict(), dst.backbone.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
    text = "\n".join(mc.tm_weights_lines(rep))
    assert "123 of 123" in text and "2 of the" in text and "not used" in text


def test_weights_threshold_is_95_percent(tmp_path):
    net = mc.TerraMindNet("tm_s1", "S1GRD")
    f = tmp_path / "w.pt"
    _checkpoint(f, net, drop=6)                       # 117 of 123 = 95.1 %
    _, rep = mc.tm_read_weights(f, net.backbone)
    assert rep["filled_parameter_tensors"] == 117 and len(rep["missing"]) == 6
    _checkpoint(f, net, drop=7)                       # 116 of 123 = 94.3 %
    with pytest.raises(SystemExit) as e:
        mc.tm_read_weights(f, net.backbone)
    msg = str(e.value)
    assert "116 of the 123" in msg and "at least 95%" in msg and "TerraMind_v1_small.pt" in msg


def test_weights_wrong_shape_missing_and_damaged(tmp_path):
    net = mc.TerraMindNet("tm_s1", "S1GRD")
    f = tmp_path / "w.pt"
    _checkpoint(f, net, bad_shape="encoder.0.mlp.fc1.bias")
    _, rep = mc.tm_read_weights(f, net.backbone)
    assert rep["missing"] == ["encoder.0.mlp.fc1.bias"]
    assert rep["wrong_shape"] and "encoder.0.mlp.fc1.bias" in rep["wrong_shape"][0]
    with pytest.raises(SystemExit) as e:
        mc.tm_read_weights(tmp_path / "nothing here.pt", net.backbone)
    assert "missing" in str(e.value) and "never downloads" in str(e.value)
    (tmp_path / "bad.pt").write_bytes(b"not a checkpoint at all")
    with pytest.raises(SystemExit) as e:
        mc.tm_read_weights(tmp_path / "bad.pt", net.backbone)
    assert "could not be read" in str(e.value)


def test_training_stops_on_a_wrong_weights_file(tm_project, tmp_path):
    root = copy_project(tm_project, tmp_path / "wrong weights")
    syn.break_terramind_weights(root)
    with pytest.raises(SystemExit) as e:
        mc.load_script("30_train_terramind.py").main(
            ["--project", str(root), "--variant", "tm_s1"] + TM_FAST)
    assert "at least 95%" in str(e.value)


# ----------------------------------------------------------------------------- 3. normalisation

def test_radar_normalisation_is_terramind_s(tmp_path):
    _, tmr = mc.tm_import()
    for modality, key in mc.TM_KEYS.items():
        r = mc.tm_radar_norm(modality)
        assert r["mean"] == list(tmr.v1_pretraining_mean[key])
        assert r["std"] == list(tmr.v1_pretraining_std[key])
    assert mc.tm_radar_norm("S1GRD")["mean"] == [-12.599, -20.293]
    assert mc.tm_radar_norm("S1GRD")["std"] == [5.195, 5.890]
    assert mc.tm_radar_norm("S1RTC")["mean"] == [-10.93, -17.329]
    assert mc.tm_radar_norm("S1RTC")["std"] == [4.391, 4.459]


def test_normalise_radar_unclipped_terrain_by_norm_stats():
    proj = mc.compute_norm_stats(np.c_[np.linspace(-20, 0, 1001), np.linspace(-30, -5, 1001),
                                       np.linspace(0, 50, 1001), np.linspace(0, 40, 1001)], "t")
    norm = mc.tm_norm_record(mc.tm_radar_norm("S1GRD"), proj, "norm_stats.json")
    st = mc.tm_stats(norm)
    x = np.array([-12.599 + 5.195, 50.0, 1000.0, 7.0], "float32").reshape(4, 1, 1)
    n = mc.normalise(x, st)
    assert np.isclose(n[0, 0, 0], 1.0, atol=1e-5)
    assert np.isclose(n[1, 0, 0], (50.0 + 20.293) / 5.890, atol=1e-4)     # no clipping
    ref = mc.normalise(x, proj)                     # the project's own rule for terrain
    assert n[2, 0, 0] == ref[2, 0, 0] and n[3, 0, 0] == ref[3, 0, 0]
    nan = mc.normalise(np.full((4, 1, 1), np.nan, "float32"), st)
    assert (nan == 0).all()
    assert norm["terrain"]["mean"] == proj["mean"][2:4]
    assert mc.tm_norm_record(mc.tm_radar_norm("S1GRD"))["terrain"] is None


def test_model_folders_hold_the_values_used(tm_trained):
    p = mc.Paths(tm_trained)
    proj = mc.load_json(p.norm_stats)
    for v in mc.TM_VARIANTS:
        d = p.model_dir(v, "hand")
        tmn = mc.load_json(d / "tm_norm.json")
        assert tmn["radar"]["mean"] == [-12.599, -20.293] and tmn["radar"]["modality"] == "S1GRD"
        rec = mc.load_json(d / "run_record.json")
        assert rec["terramind_norm"] == tmn
        if v == "tm_s1t":
            assert tmn["terrain"]["mean"] == proj["mean"][2:4]
            assert tmn["terrain"]["clip_hi"] == proj["clip_hi"][2:4]
        else:
            assert tmn["terrain"] is None


def test_tm_s1_never_reads_hand_or_slope(tm_project, tmp_path, monkeypatch):
    root = copy_project(tm_project, tmp_path / "radar only")
    calls = []
    real = mc.read_terrain
    monkeypatch.setattr(mc, "read_terrain", lambda p, w=None: (calls.append(str(p)), real(p, w))[1])
    assert run("30_train_terramind.py", ["--project", root, "--variant", "tm_s1", "--force"]
               + TM_FAST) == 0
    assert calls == []
    assert run("30_train_terramind.py", ["--project", root, "--variant", "tm_s1t", "--force"]
               + TM_FAST) == 0
    assert calls                                    # tm_s1t does read them
    net = mc.TerraMindNet("tm_s1").eval()
    x = torch.randn(1, 4, 64, 64)
    y = x.clone()
    y[:, 2:] = 1e6
    with torch.no_grad():
        assert torch.equal(net(x), net(y))
    net_t = mc.TerraMindNet("tm_s1t").eval()
    with torch.no_grad():
        assert not torch.equal(net_t(x), net_t(y))


def test_tm_s1_predictions_ignore_terrain(tm_trained):
    pred = mc.load_predictor("tm_s1", mc.Paths(tm_trained).model_dir("tm_s1", "hand"),
                             cpu_ok=True, overlap=16)
    assert pred.uses_terrain is False
    x, _ = _valid_chip(tm_trained)
    v = mc.input_valid(x)
    a = pred.predict_image(x, v)
    x2 = x.copy()
    x2[2:] = 1e6
    assert np.array_equal(a, pred.predict_image(x2, v), equal_nan=True)


# ----------------------------------------------------------------------------- 4. train, reload

@pytest.mark.parametrize("variant", mc.TM_VARIANTS)
def test_saved_reloaded_predicts(tm_trained, variant):
    p = mc.Paths(tm_trained)
    d = p.model_dir(variant, "hand")
    for f in ("best.pt", "last.pt", "history.csv", "model_meta.json", "norm_stats.json",
              "tm_norm.json", "terramind_weights.json", "run_record.json", "train_chips.txt"):
        assert (d / f).exists(), f
    rec = mc.load_json(d / "run_record.json")
    assert rec["finished"] is True and rec["epochs_done"] == 2
    assert rec["terramind_weights_sha256"] == mc.sha256(p.tm_weights())
    assert rec["terramind_weights_report"]["filled_share"] == 1.0
    meta = mc.load_json(d / "model_meta.json")
    assert meta["variant"] == variant and meta["modality"] == "S1GRD" and meta["crop"] == 64
    assert 0 < meta["threshold_valid"] < 1
    pred = mc.load_predictor(variant, d, cpu_ok=True, tile=512, overlap=16)
    assert pred.tile == 64                          # its training crop, not --tile
    x, _ = _valid_chip(tm_trained)
    v = mc.input_valid(x)
    prob = pred.predict_image(x, v)
    assert np.isnan(prob[~v]).all() and np.isfinite(prob[v]).all()
    assert ((prob[v] >= 0) & (prob[v] <= 1)).all()


def test_encoder_and_decoder_learning_rates(tm_trained):
    ck = _last(tm_trained, "tm_s1t")
    groups = ck["optimizer"]["param_groups"]
    assert [g["initial_lr"] for g in groups] == [5e-5, 1e-3]
    assert [g["weight_decay"] for g in groups] == [1e-4, 1e-4]
    assert len(groups[0]["params"]) == 123


def test_same_seed_same_first_loss(tm_project, tm_trained, tmp_path):
    losses = []
    for k in range(2):
        other = copy_project(tm_project, tmp_path / f"copy {k}")
        assert run("30_train_terramind.py", ["--project", other, "--variant", "tm_s1t",
                                             "--force", "--stop-after-steps", "1"] + TM_FAST) == 0
        losses.append(_last(other, "tm_s1t")["first_step_loss"])
        (mc.Paths(other).model_dir("tm_s1t", "hand") / "last.pt").unlink()
    ref = mc.load_json(mc.Paths(tm_trained).model_dir("tm_s1t", "hand") / "run_record.json")
    assert losses[0] == losses[1] == ref["first_step_loss"]


def test_hand_weak_with_spawned_workers(tm_project, tmp_path):
    """Two DataLoader workers started with spawn, as on Windows: everything must pickle."""
    other = copy_project(tm_project, tmp_path / "workers data")
    argv = list(TM_FAST)
    argv[argv.index("--workers") + 1] = "2"
    argv[argv.index("--epochs") + 1] = "1"
    assert run("30_train_terramind.py", ["--project", other, "--variant", "tm_s1t",
                                         "--trainset", "hand_weak"] + argv) == 0
    d = mc.Paths(other).model_dir("tm_s1t", "hand_weak")
    rec = mc.load_json(d / "run_record.json")
    assert rec["finished"] and rec["weak_chips"] == 5
    chips = (d / "train_chips.txt").read_text().split()
    assert "Ghana_5001" in chips and "Bolivia_5005" not in chips and "Ghana_2001" not in chips


def test_refuses_cpu_without_flag_and_overwrite_without_force(tm_project, tm_trained, tmp_path):
    rc = run("30_train_terramind.py", ["--project", tm_trained, "--variant", "tm_s1"] + TM_FAST)
    assert rc == 1                                  # finished: needs --force or --skip-existing
    assert run("30_train_terramind.py", ["--project", tm_trained, "--variant", "tm_s1",
                                         "--resume", "--skip-existing"] + TM_FAST) == 0
    if torch.cuda.is_available():
        return
    other = copy_project(tm_project, tmp_path / "cpu data")
    argv = [a for a in TM_FAST if a != "--cpu-ok"]
    assert run("30_train_terramind.py", ["--project", other, "--variant", "tm_s1", "--force"]
               + argv) == 1


# ----------------------------------------------------------------------------- 5. resume

def _same(a, b):
    sa, sb = a["model"], b["model"]
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_resume_after_epoch_matches_uninterrupted(tm_project, tm_trained, tmp_path):
    ref = _last(tm_trained, "tm_s1")                 # 2 epochs, no stop
    other = copy_project(tm_project, tmp_path / "resume data")
    base = ["--project", other, "--variant", "tm_s1"] + TM_FAST
    assert run("30_train_terramind.py", base + ["--stop-after-epochs", "1"]) == 0
    mid = _last(other, "tm_s1")
    assert mid["epoch"] == 1 and mid["step_in_epoch"] == 0
    assert mc.load_json(mc.Paths(other).model_dir("tm_s1", "hand") / "run_record.json")[
        "finished"] is False
    assert run("30_train_terramind.py", base + ["--resume"]) == 0
    end = _last(other, "tm_s1")
    assert end["epoch"] == ref["epoch"] == 2
    assert _same(end, ref) and end["history"] == ref["history"]
    for g in (0, 1):
        i = end["optimizer"]["param_groups"][g]["params"][0]
        assert torch.equal(end["optimizer"]["state"][i]["exp_avg"],
                           ref["optimizer"]["state"][i]["exp_avg"])


def test_resume_mid_epoch_matches_uninterrupted(tm_project, tm_trained, tmp_path):
    """The Ctrl+C path: stop inside epoch 2, save, resume, same result."""
    ref = _last(tm_trained, "tm_s1t")
    other = copy_project(tm_project, tmp_path / "ctrl c data")
    base = ["--project", other, "--variant", "tm_s1t"] + TM_FAST
    assert run("30_train_terramind.py", base + ["--stop-after-steps", "4"]) == 0
    mid = _last(other, "tm_s1t")
    assert mid["epoch"] == 1 and mid["step_in_epoch"] == 1
    assert run("30_train_terramind.py", base + ["--resume"]) == 0
    end = _last(other, "tm_s1t")
    assert _same(end, ref) and end["history"] == ref["history"]


def test_resume_refuses_other_settings(tm_project, tmp_path):
    other = copy_project(tm_project, tmp_path / "other settings")
    base = ["--project", other, "--variant", "tm_s1"]
    assert run("30_train_terramind.py", base + ["--stop-after-steps", "2"] + TM_FAST) == 0
    with pytest.raises(SystemExit) as e:
        mc.load_script("30_train_terramind.py").main(
            [str(a) for a in base] + ["--resume", "--lr-encoder", "1e-4"] + TM_FAST)
    assert "--lr-encoder was 5e-05, now 0.0001" in str(e.value)


# ----------------------------------------------------------------------------- 6. corridor map

def test_corridor_map_contract_and_scorer(tm_trained):
    p = mc.Paths(tm_trained)
    assert run("25_predict_corridor.py", ["--project", tm_trained, "--model", "tm_s1t",
                                          "--trainset", "hand", "--year", "2021", "--cpu-ok",
                                          "--overlap", "16"]) == 0
    out = p.flood_map(2021, "tm_s1t", "hand")
    scene = p.scene(2021, "grd")
    with rasterio.open(out) as m, rasterio.open(scene) as s:
        assert m.count == 1 and m.dtypes[0] == "uint8" and m.nodata == 255
        assert m.crs == s.crs and m.transform == s.transform and m.shape == s.shape
        assert m.profile.get("compress", "").lower() == "lzw"
        a, vv, tf, crs = m.read(1), s.read(1), m.transform, m.crs
    assert set(np.unique(a)) <= {0, 1, 255}
    assert np.array_equal(a == 255, ~np.isfinite(vv))
    prof = {"height": a.shape[0], "width": a.shape[1], "transform": tf, "crs": crs}
    perm = mc.reproject_onto(p.perm_water(), prof, categorical=True) == 100
    ls = mc.reproject_onto(p.strata(2021) / "strata_layover_shadow.tif", prof, categorical=True)
    valid = np.isfinite(vv)
    assert (a[valid & perm] == 0).all() and (valid & perm).any()
    dist = np.isin(ls, (1, 2))
    assert (a[valid & dist] == 0).all() and (valid & dist).any()
    diag = json.loads((out.parent / f"{out.stem}_diagnostics.json").read_text())
    assert diag["flood_cells"] == diag["water_raw_cells"] - diag["permanent_removed_cells"] \
        - diag["distortion_removed_cells"]
    rec = json.loads((out.parent / "run_record.json").read_text())
    assert rec["terramind_weights_sha256"] == mc.sha256(p.tm_weights())
    assert rec["modality"] == "S1GRD" and rec["terramind_norm"]["terrain"] is not None
    tables = p.tables / "2021_tm_s1t_hand"
    r = run_cli([FYP / "04_stratified_metrics.py", "--pred", out, "--truth", p.reference(2021),
                 "--strata", p.strata(2021), "--windows", p.windows(2021),
                 "--label", "Table A, tm_s1t hand, 2021", "--out", tables])
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tables / "metrics_by_stratum.csv").exists() and (tables / "metrics_summary.md").exists()


def test_sen1floods11_evaluation_of_terramind(tm_trained):
    assert run("24_eval_sen1floods11.py", ["--project", tm_trained, "--model", "tm_s1",
                                           "--cpu-ok", "--overlap", "16"]) == 0
    out = mc.Paths(tm_trained).tables / "s1f11_tm_s1_hand"
    assert (out / "metrics.csv").exists()
    rec = mc.load_json(out / "run_record.json")
    assert rec["variant"] == "tm_s1" and rec["terramind_norm"]["terrain"] is None


# ----------------------------------------------------------------------------- 7. Table D

def test_lowo_grd_and_rtc(tm_project, tmp_path):
    root = copy_project(tm_project, tmp_path / "lowo tm")
    p = mc.Paths(root)
    for radar in ("grd", "rtc"):
        assert run("26_corridor_lowo.py", ["--project", root, "--model", "tm_s1t", "--radar",
                                           radar] + LOWO_TM_FAST) == 0
    want = {"grd": ("S1GRD", [-12.599, -20.293]), "rtc": ("S1RTC", [-10.93, -17.329])}
    sha = mc.sha256(p.tm_weights())
    for radar, (modality, mean) in want.items():
        d = p.lowo_dir(2021, "tm_s1t", radar)
        assert sorted(f.name for f in d.glob("fold_*")) == ["fold_1", "fold_2", "fold_3",
                                                           "fold_4"]
        for k in range(1, 5):
            rec = mc.load_json(d / f"fold_{k}" / "run_record.json")
            assert k not in rec["train_windows"] and rec["train_cells_in_heldout_window"] == 0
            assert rec["heldout_labels_in_training_crops"] == 0
            assert rec["modality"] == modality and rec["terramind_norm"]["radar"]["mean"] == mean
            assert rec["terramind_norm"]["terrain"]["mean"] == rec["norm_stats"]["mean"][2:4]
            assert rec["terramind_weights_sha256"] == sha and rec["steps"] == 3
            assert (d / f"fold_{k}" / "history.csv").exists()
        assert mc.load_json(d / "run_record.json")["terramind_weights_sha256"] == [sha]
    with rasterio.open(p.lowo_map(2021, "tm_s1t", "grd")) as a, \
            rasterio.open(p.lowo_map(2021, "tm_s1t", "rtc")) as b:
        ma, mb = a.read(1), b.read(1)
        assert a.dtypes[0] == "uint8" and a.nodata == 255
    assert np.array_equal(ma == 255, mb == 255)
    with rasterio.open(p.scene(2021, "rtc")) as r:
        assert (ma[~np.isfinite(r.read(1))] == 255).all()


def test_lowo_terramind_defaults(monkeypatch):
    """Fixed in advance: 1000 steps of batch 8, 224 x 224 crops; the U-Net keeps its own."""
    ap_args = []

    class Stop(Exception):
        pass

    def grab(args):
        ap_args.append(args)
        raise Stop

    monkeypatch.setattr(mc, "paths_from_args", grab)
    for model in ("tm_s1", "unet"):
        with pytest.raises(Stop):
            lowo.main(["--model", model])
    tm, un = ap_args
    assert (tm.steps, tm.batch_size, tm.crop, tm.lr, tm.lr_encoder) == (1000, 8, 224, 1e-3, 5e-5)
    assert (un.steps, un.batch_size, un.crop) == (1500, 16, 128)


def test_preflight_names_the_batch_size_that_fits(monkeypatch):
    """Out of GPU memory at batch 8 (simulated: there is no GPU here)."""
    pf = mc.load_script("29_terramind_preflight.py")

    class Dummy:
        def to(self, device):
            return self

    def oom(*a, **k):
        raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(mc, "gpu_info", lambda: {"gpu": "test GPU", "gpu_vram_gb": 4.0})
    monkeypatch.setattr(mc, "set_deterministic", lambda on: None)
    monkeypatch.setattr(mc, "TerraMindNet", lambda variant: Dummy())
    monkeypatch.setattr(mc, "terramind_memory_estimate", lambda *a: {"estimate_gb": 2.3})
    monkeypatch.setattr(mc, "find_batch_size", lambda net, dev, crop, batch, amp: 4)
    monkeypatch.setattr(pf, "measure", oom)
    c = pf.Checks()
    assert pf.check_gpu(c, 8, 224, 2, cpu_ok=False) is None
    bad = [r for r in c.rows if r["status"] == "FAIL"]
    assert len(bad) == 1 and "--batch-size 4 fits" in bad[0]["detail"]


def test_summary_lists_terramind_runs(tmp_path):
    root = tmp_path / "summary" / "Final Year Project"
    (root / "06 Outputs").mkdir(parents=True)
    assert run("28_models_summary.py", ["--project", root]) == 0
    text = (mc.Paths(root).tables / "MODELS_SUMMARY.md").read_text(encoding="utf-8")
    for item in ("Table A 2021_tm_s1t_hand_weak", "Table D lowo_2023_tm_s1_rtc",
                 "Sen1Floods11 s1f11_tm_s1t_hand"):
        assert item in text, item
