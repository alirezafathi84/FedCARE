import numpy as np
import torch
from fedcare.stats import (calibrate_head, class_stats, merge_stats, reconstruct_cov,
                           stats_num_floats, subspace_distance, subspace_prior)


def _clients(rng, C=3, d=6, K=3):
    data = []
    for k in range(K):
        n = rng.integers(20, 60)
        y = rng.integers(0, C, n)
        h = rng.normal(size=(n, d)) + y[:, None] * 2.0 + k * 0.5   # site offset
        data.append((h, y))
    return data


def test_full_rank_merge_equals_pooled_statistics():
    rng = np.random.default_rng(0)
    C, d = 3, 6
    data = _clients(rng, C, d)
    for rank in (None, d):                                   # full cov and full-rank PPCA
        stats = [class_stats(h, y, np.bincount(y, minlength=C), C, rank) for h, y in data]
        m = merge_stats(stats)
        H = np.concatenate([h for h, _ in data]); Y = np.concatenate([y for _, y in data])
        for c in range(C):
            hc = H[Y == c]
            assert np.allclose(m["mean"][c], hc.mean(0))
            assert np.allclose(m["total"][c], np.cov(hc, rowvar=False, bias=True), atol=1e-8)


def test_lowrank_reconstruction_preserves_trace():
    rng = np.random.default_rng(1)
    h = rng.normal(size=(200, 10)) @ rng.normal(size=(10, 10))
    y = np.zeros(200, dtype=int)
    st = class_stats(h, y, np.array([200]), 1, rank=3)
    S = np.cov(h, rowvar=False, bias=True)
    R = reconstruct_cov(st)[0]
    assert np.isclose(np.trace(R), np.trace(S))
    w = np.linalg.eigvalsh(S)[::-1]
    assert np.allclose(np.linalg.eigvalsh(R)[::-1][:3], w[:3])
    assert stats_num_floats(st) == 1 + 10 + 3 * 10 + 3 + 1


def test_subspace_distance_matches_projection_formula_and_is_bounded():
    rng = np.random.default_rng(2)
    C, d = 2, 8
    stats = []
    for h, y in _clients(rng, C, d):
        hn = h / np.linalg.norm(h, axis=1, keepdims=True)
        stats.append(class_stats(hn, y, np.bincount(y, minlength=C), C, 3))
    m = merge_stats(stats)
    for r in (0, 2):
        pr = subspace_prior(m, r)
        h = torch.randn(6, d)
        y = torch.tensor([0, 1, 0, 1, 1, 0])
        got = subspace_distance(h, y, pr, gamma=0.1)
        for i in range(6):
            c = int(y[i]); U = pr["U"][c]
            dl = h[i] / h[i].norm() - pr["mean"][c]
            P = U @ U.T
            ref = ((dl - P @ dl) ** 2).sum() + 0.1 * ((P @ dl) ** 2).sum()
            assert torch.isclose(got[i], ref, atol=1e-5)
        assert bool((got <= 4.0 + 1e-5).all())


def test_calibration_removes_majority_bias():
    torch.manual_seed(0)
    C, d = 3, 4
    mean = np.eye(C, d) * 3.0
    merged = {"N": np.array([100.0, 100.0, 100.0]), "mean": mean,
              "total": np.stack([np.eye(d)] * C), "within": np.stack([np.eye(d)] * C)}
    head = torch.nn.Linear(d, C)
    with torch.no_grad():
        head.weight.zero_(); head.bias.copy_(torch.tensor([5.0, 0.0, 0.0]))   # always predicts class 0
    calibrate_head(head, merged, n_virtual=200, steps=200, lr=0.1)
    pred = head(torch.tensor(mean, dtype=torch.float32)).argmax(1)
    assert pred.tolist() == [0, 1, 2]
