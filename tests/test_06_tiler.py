"""Brief 9.6: tiled prediction: identity reproduces the input exactly; blank tiles skipped."""
import numpy as np
import pytest

import models_common as mc


@pytest.mark.parametrize("shape,tile,overlap", [((300, 517), 128, 32), ((64, 64), 64, 16),
                                                ((3044 // 8, 7915 // 8), 96, 24), ((40, 50), 64, 8)])
def test_identity_exact(shape, tile, overlap):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4,) + shape).astype("float32")
    valid = np.ones(shape, bool)
    prob, st = mc.predict_tiled(lambda t: t[:, 0], x, valid, tile, overlap, batch=5)
    assert prob.shape == shape and prob.dtype == np.float32
    assert np.array_equal(prob, x[0])


def test_blank_tiles_skipped_and_never_seen():
    shape = (256, 640)
    x = np.random.default_rng(1).normal(size=(4,) + shape).astype("float32")
    valid = np.zeros(shape, bool)
    valid[10:60, 10:200] = True
    seen = []

    def fn(t):
        seen.append(len(t))
        return t[:, 0]

    prob, st = mc.predict_tiled(fn, x, valid, 128, 32, batch=2)
    assert st["tiles_skipped"] > 0 and sum(seen) == st["tiles_predicted"]
    assert np.array_equal(prob[valid], x[0][valid]) and np.isnan(prob[~valid]).all()


def test_all_blank_raster():
    x = np.zeros((4, 100, 100), "float32")
    prob, st = mc.predict_tiled(lambda t: t[:, 0], x, np.zeros((100, 100), bool), 64, 16)
    assert st["tiles_predicted"] == 0 and np.isnan(prob).all()


def test_blending_hides_seams():
    """A model that only sees tile-local position would show seams; the ramp blends them."""
    shape = (200, 200)
    x = np.zeros((4,) + shape, "float32")
    tile = 64

    def fn(t):
        out = np.zeros(t.shape[:1] + t.shape[2:], "float32")
        out[:, :, :] = np.linspace(0, 1, tile)[None, None, :]
        return out

    prob, _ = mc.predict_tiled(fn, x, np.ones(shape, bool), tile, 32)
    jumps = np.abs(np.diff(prob, axis=1))
    assert jumps.max() < 0.2


def test_starts_cover_everything():
    for n in (1, 63, 64, 65, 500, 3044, 7915):
        s = mc.tile_starts(n, 64, 16)
        covered = np.zeros(max(n, 64), bool)
        for a in s:
            covered[a:a + 64] = True
        assert covered[:n].all() and s[-1] + 64 >= n
