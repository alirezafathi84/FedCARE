import numpy as np
from fedcare.data import (apply_site_shift, dirichlet_partition, hed2rgb, rgb2hed,
                           sample_site_params, stratified_subsample)


def test_dirichlet_partition_is_exact_cover_and_deterministic():
    y = np.random.default_rng(0).integers(0, 7, 3000)
    p1 = dirichlet_partition(y, 10, 0.1, np.random.default_rng(1), min_size=10)
    p2 = dirichlet_partition(y, 10, 0.1, np.random.default_rng(1), min_size=10)
    allidx = np.sort(np.concatenate(p1))
    assert np.array_equal(allidx, np.arange(len(y)))          # every sample exactly once
    assert min(len(p) for p in p1) >= 10
    assert all(np.array_equal(a, b) for a, b in zip(p1, p2))  # seeded => reproducible


def test_small_alpha_gives_label_skew():
    y = np.repeat(np.arange(8), 500)
    parts = dirichlet_partition(y, 10, 0.1, np.random.default_rng(0))
    # with alpha=0.1 most clients are dominated by a few classes
    dominant = [np.bincount(y[p], minlength=8).max() / len(p) for p in parts]
    assert np.median(dominant) > 0.5


def test_stratified_subsample_keeps_all_classes():
    y = np.array([0] * 900 + [1] * 90 + [2] * 10)
    idx = stratified_subsample(y, 100, np.random.default_rng(0))
    assert set(np.unique(y[idx])) == {0, 1, 2} and 95 <= len(idx) <= 105


def test_hed_roundtrip_and_site_shift():
    rng = np.random.default_rng(0)
    hed = rng.uniform(0.0, 0.08, size=(4, 8, 8, 3))           # non-negative stain densities
    x = hed2rgb(hed)
    assert np.allclose(rgb2hed(x), hed, atol=1e-6)            # exact inverse on valid stains
    imgs = (x * 255).astype(np.uint8)
    p = sample_site_params(rng)
    for mode in ("hed", "od"):
        out = apply_site_shift(imgs, p, rng, mode)
        assert out.shape == imgs.shape and out.dtype == np.uint8
        assert np.abs(out.astype(int) - imgs.astype(int)).mean() > 1.0   # visibly shifted
