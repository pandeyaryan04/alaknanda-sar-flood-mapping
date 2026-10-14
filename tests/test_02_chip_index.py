"""Brief 9.2: chip index counts, drops, splits; no held-out chip in a training list."""
import pytest

import models_common as mc


def _index(project):
    return {r["chip"]: r for r in mc.load_chip_index(mc.Paths(project))}


def test_counts(project):
    idx = _index(project)
    hand = [r for r in idx.values() if r["set"] == "hand"]
    weak = [r for r in idx.values() if r["set"] == "weak"]
    assert len(hand) == 18 and len(weak) == 8
    kept = {s: sum(r["kept"] == "yes" for r in idx.values() if r["split"] == s)
            for s in ("train", "valid", "test", "bolivia", "weak")}
    assert kept == {"train": 8, "valid": 3, "test": 3, "bolivia": 2, "weak": 5}


def test_drops_and_reasons(project):
    idx = _index(project)
    assert idx["Paraguay_34417"]["reason"] == "empty chip (no radar data)"
    assert idx["USA_5007"]["reason"] == "empty chip (no radar data)"
    assert idx["Spain_1007"]["reason"] == "no labelled pixels"
    assert idx["Bolivia_5005"]["reason"] == "weak chip of event Bolivia"
    assert idx["Mekong_5006"]["reason"] == "overlaps a hand valid, test or Bolivia chip"
    assert idx["Paraguay_5008"]["kept"] == "yes"       # overlaps a TRAIN chip: allowed
    assert idx["Sri-Lanka_1008"]["event"] == "Sri-Lanka"


def test_splits_respected(project):
    paths = mc.Paths(project)
    idx = _index(project)
    for split, fname in mc.SPLIT_FILES.items():
        for c in mc.read_split_csv(paths.splits / fname):
            assert idx[c]["split"] == split


def test_no_held_out_chip_in_training(project):
    paths = mc.Paths(project)
    index = mc.load_chip_index(paths)
    held = set()
    for split in ("valid", "test", "bolivia"):
        held |= set(mc.read_split_csv(paths.splits / mc.SPLIT_FILES[split]))
    for ts in ("hand", "hand_weak"):
        hand, weak = mc.training_rows(index, ts)
        ids = {r["chip"] for r in hand + weak}
        assert not ids & held
        assert not any(mc.event_of(c) == "Bolivia" for c in ids)
    hand, weak = mc.training_rows(index, "hand")
    assert weak == []


def test_training_rows_refuses_a_leak(project):
    index = mc.load_chip_index(mc.Paths(project))
    bad = [dict(r) for r in index]
    victim = next(r for r in bad if r["split"] == "test")
    bad.append(dict(victim, split="train"))          # the same chip id listed as train
    with pytest.raises(RuntimeError):
        mc.training_rows(bad, "hand")


def test_footprint_intersection_rule():
    a = [(0, 0, 1, 1), (2, 2, 3, 3), (1, 0, 2, 1)]
    b = [(0.5, 0.5, 1.5, 1.5)]
    assert mc.boxes_intersect(a, b).tolist() == [True, False, True]
    assert mc.boxes_intersect([(1, 1, 2, 2)], [(0, 0, 1, 1)]).tolist() == [True]   # touching
