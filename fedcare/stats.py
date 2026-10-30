"""Class-conditional feature statistics used by FedCARE (and by post-hoc CCVR).

Client side
    ``class_stats`` summarises the penultimate features of each local class by
    its count, mean and either the full covariance (CCVR) or a rank-r
    eigen-decomposition plus an isotropic residual (probabilistic PCA, FedCARE).
Server side
    ``merge_stats`` combines client summaries exactly (law of total covariance)
    into a pooled *within-client* class covariance and a *between-client*
    covariance of the client means; ``calibrate_head`` re-trains a linear
    classifier on class-balanced virtual features (post-hoc calibration).
Training
    ``subspace_prior`` builds global class subspaces from the merged statistics
    and ``subspace_distance`` is the alignment penalty.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def extract_features(model: nn.Module, x: torch.Tensor, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    out = [model(x[i:i + batch_size], return_features=True)[0].double()
           for i in range(0, len(x), batch_size)]
    return torch.cat(out).numpy() if out else np.zeros((0, model.feat_dim))


def capped_indices(y: np.ndarray, cap: Optional[int], rng: np.random.Generator) -> np.ndarray:
    """At most ``cap`` random samples per class (keeps statistics cheap on edge clients)."""
    if cap is None:
        return np.arange(len(y))
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        keep.append(rng.choice(idx, cap, replace=False) if len(idx) > cap else idx)
    return np.sort(np.concatenate(keep))


def class_stats(h: np.ndarray, y: np.ndarray, counts: np.ndarray, num_classes: int,
                rank: Optional[int]) -> Dict:
    """Per-class summary of features ``h`` (n x d).

    ``counts`` are the client's true class counts (used as merge weights);
    ``rank=None`` keeps the full covariance, otherwise the top-``rank``
    eigenpairs and the mean residual eigenvalue are kept.
    """
    d = h.shape[1]
    st: Dict = {"n": np.asarray(counts, dtype=np.float64).copy(),
                "mean": np.zeros((num_classes, d)), "rank": rank}
    if rank is None:
        st["cov"] = np.zeros((num_classes, d, d))
    else:
        r = min(rank, d)
        st.update(U=np.zeros((num_classes, d, r)), lam=np.zeros((num_classes, r)),
                  res=np.zeros(num_classes))
    for c in range(num_classes):
        hc = h[y == c]
        if len(hc) == 0:
            st["n"][c] = 0.0
            continue
        st["mean"][c] = hc.mean(0)
        if len(hc) < 2:
            continue
        cov = np.cov(hc, rowvar=False, bias=True)
        if rank is None:
            st["cov"][c] = cov
        else:
            w, v = np.linalg.eigh(cov)
            w, v = w[::-1], v[:, ::-1]
            r = st["U"].shape[2]
            st["U"][c] = v[:, :r]
            st["lam"][c] = np.maximum(w[:r], 0.0)
            st["res"][c] = max(w[r:].sum(), 0.0) / (d - r) if d > r else 0.0
    return st


def stats_num_floats(st: Dict) -> int:
    """Number of floats a client uploads for this summary."""
    present = int((st["n"] > 0).sum())
    d = st["mean"].shape[1]
    if st["rank"] is None:
        per = 1 + d + d * (d + 1) // 2
    else:
        r = st["U"].shape[2]
        per = 1 + d + r * d + r + 1
    return present * per


def reconstruct_cov(st: Dict) -> np.ndarray:
    """Client covariance per class (C x d x d) from its summary."""
    if st["rank"] is None:
        return st["cov"]
    d = st["mean"].shape[1]
    U, lam, res = st["U"], st["lam"], st["res"]
    low = np.einsum("cdr,cr,cer->cde", U, lam, U)
    proj = np.einsum("cdr,cer->cde", U, U)
    return low + res[:, None, None] * (np.eye(d)[None] - proj)


def merge_stats(stats: List[Dict]) -> Dict:
    """Exact merge: mean, pooled within-client covariance, between-client covariance."""
    n = np.stack([s["n"] for s in stats])                      # K x C
    means = np.stack([s["mean"] for s in stats])               # K x C x d
    covs = np.stack([reconstruct_cov(s) for s in stats])       # K x C x d x d
    N = n.sum(0)
    w = n / np.maximum(N, 1e-12)                               # K x C
    mu = np.einsum("kc,kcd->cd", w, means)
    within = np.einsum("kc,kcde->cde", w, covs)
    dm = means - mu[None]
    between = np.einsum("kc,kcd,kce->cde", w, dm, dm)
    return {"N": N, "mean": mu, "within": within, "between": between,
            "total": within + between}


def subspace_prior(merged: Dict, rank: int, min_count: float = 2.0) -> Dict:
    """Global class subspaces from the pooled *within-client* class covariance.

    For each class c the prior is the affine subspace mu_c + span(U_c), where U_c
    holds the top-``rank`` principal directions of the within-client scatter.
    Because the between-client spread of the class means is excluded, directions
    along which sites disagree are *not* absorbed into U_c and are penalised by
    ``subspace_distance``. ``rank=0`` reduces the prior to class prototypes.
    """
    mean, within, N = merged["mean"], merged["within"], merged["N"]
    C, d = mean.shape
    r = min(rank, d)
    U = np.zeros((C, d, r))
    ok = N >= min_count
    for c in np.flatnonzero(ok):
        w, v = np.linalg.eigh((within[c] + within[c].T) / 2)
        U[c] = v[:, ::-1][:, :r]
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    return {"valid": torch.tensor(ok), "mean": t(mean), "U": t(U)}


def subspace_distance(h: torch.Tensor, y: torch.Tensor, prior: Dict, gamma: float = 0.1) -> torch.Tensor:
    """Anisotropic distance of l2-normalised features to their class subspace.

    R = ||(I - U U^T)(h - mu)||^2 + gamma * ||U U^T (h - mu)||^2,  h on the unit sphere.
    Off-subspace (site-specific) deviations are penalised fully, deviations along
    the shared principal directions only by ``gamma``. Bounded (R <= 4), so the
    penalty stays well-conditioned while the feature space drifts across rounds.
    """
    delta = F.normalize(h, dim=1) - prior["mean"][y]
    sq = (delta * delta).sum(1)
    if prior["U"].shape[-1] == 0:
        return sq
    a2 = torch.einsum("bd,bdr->br", delta, prior["U"][y]).pow(2).sum(1)
    return (sq - a2).clamp_min(0.0) + gamma * a2


def sample_virtual_features(merged: Dict, n_per_class: int, rng: np.random.Generator,
                            cov_key: str = "total", min_count: float = 1.0):
    feats, labels = [], []
    C, d = merged["mean"].shape
    for c in range(C):
        if merged["N"][c] < min_count:
            continue
        S = merged[cov_key][c]
        w, v = np.linalg.eigh((S + S.T) / 2)
        L = v * np.sqrt(np.clip(w, 0.0, None))
        feats.append(rng.standard_normal((n_per_class, d)) @ L.T + merged["mean"][c])
        labels.append(np.full(n_per_class, c))
    return (torch.tensor(np.concatenate(feats), dtype=torch.float32),
            torch.tensor(np.concatenate(labels), dtype=torch.long))


def calibrate_head(head: nn.Linear, merged: Dict, n_virtual: int = 128, steps: int = 100,
                   lr: float = 0.05, batch_size: int = 256, seed: int = 0,
                   cov_key: str = "total") -> nn.Linear:
    """Fine-tune the linear classifier in place on class-balanced virtual features."""
    rng = np.random.default_rng(seed)
    X, Y = sample_virtual_features(merged, n_virtual, rng, cov_key)
    gen = torch.Generator().manual_seed(int(seed))
    opt = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9)
    head.train()
    for _ in range(steps):
        idx = torch.randint(0, len(Y), (min(batch_size, len(Y)),), generator=gen)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(head(X[idx]), Y[idx]).backward()
        opt.step()
    return head
