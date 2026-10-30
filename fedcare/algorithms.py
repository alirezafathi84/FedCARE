"""Federated algorithms.

Every algorithm implements the same hooks, called by ``runner.Runner``:

    client_start(k, client, local, global) -> ctx      (before local training)
    loss(ctx, model, x, y) -> scalar                  (each local step)
    after_backward(ctx, model)                        (gradient correction)
    client_end(ctx, local, client, runner) -> payload (extra upload)
    aggregate(global, results)                        (server update)
    finalize(global, runner) -> {variant: model}      (post-hoc variants)
"""
from __future__ import annotations

import copy
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .stats import (calibrate_head, merge_stats, stats_num_floats, subspace_distance,
                    subspace_prior)


def weighted_average_states(states: List[Dict[str, torch.Tensor]], weights) -> Dict[str, torch.Tensor]:
    out = {}
    for key, ref in states[0].items():
        if not torch.is_floating_point(ref):        # e.g. BatchNorm num_batches_tracked
            out[key] = ref.clone()
            continue
        acc = torch.zeros_like(ref, dtype=torch.float64)
        for s, w in zip(states, weights):
            acc += float(w) * s[key].double()
        out[key] = acc.to(ref.dtype)
    return out


def _frozen_copy(model: nn.Module) -> nn.Module:
    m = copy.deepcopy(model).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


class FedAvg:
    """McMahan et al., AISTATS 2017."""
    name = "fedavg"
    defaults: Dict = {}

    def __init__(self, hp: Dict, num_classes: int, feat_dim: int, num_clients: int):
        self.hp = {**self.defaults, **(hp or {})}
        self.C, self.d, self.K = num_classes, feat_dim, num_clients
        self.rnd = 0

    # ---- client side ------------------------------------------------------ #
    def client_start(self, k, client, local, global_model) -> Dict:
        return {"k": k, "counts": client["counts"]}

    def loss(self, ctx, model, x, y):
        return F.cross_entropy(model(x), y)

    def after_backward(self, ctx, model):
        pass

    def client_end(self, ctx, local, client, runner) -> Dict:
        return {}

    # ---- server side ------------------------------------------------------ #
    def sample_weights(self, results):
        w = np.array([r["n"] for r in results], dtype=np.float64)
        return w / w.sum()

    def aggregate(self, global_model, results):
        global_model.load_state_dict(
            weighted_average_states([r["state"] for r in results], self.sample_weights(results)))

    def finalize(self, global_model, runner) -> Dict[str, nn.Module]:
        return {}

    def extra_upload_floats(self, payload, n_params: int) -> int:
        return 0


class FedAvgPostHoc(FedAvg):
    """FedAvg plus two post-hoc classifier re-calibrations of the final model:
    CCVR (Luo et al., NeurIPS 2021; full class covariances) and the low-rank
    calibration used by FedCARE (same procedure, rank-r summaries)."""
    name = "fedavg"
    defaults = {"stats_cap": 64, "rank": 4, "n_virtual": 128, "cal_steps": 100, "cal_lr": 0.05}

    def finalize(self, global_model, runner):
        out = {}
        for tag, rank in (("fedavg+ccvr", None), ("fedavg+lrcal", self.hp["rank"])):
            m = copy.deepcopy(global_model)
            stats = runner.client_feature_stats(m, rank, self.hp["stats_cap"])
            calibrate_head(m.head, merge_stats(stats), self.hp["n_virtual"], self.hp["cal_steps"],
                           self.hp["cal_lr"], seed=runner.seed)
            out[tag] = m
        return out


class FedProx(FedAvg):
    """Li et al., MLSys 2020: proximal term (mu/2)||w - w_global||^2."""
    name = "fedprox"
    defaults = {"mu": 0.01}

    def client_start(self, k, client, local, global_model):
        ctx = super().client_start(k, client, local, global_model)
        ctx["gparams"] = [p.detach().clone() for p in global_model.parameters()]
        return ctx

    def loss(self, ctx, model, x, y):
        prox = sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), ctx["gparams"]))
        return F.cross_entropy(model(x), y) + 0.5 * self.hp["mu"] * prox


def effective_steps(tau: int, rho: float) -> float:
    """Number of plain-SGD steps equivalent to ``tau`` heavy-ball steps with momentum
    ``rho`` (the normaliser a_k of FedNova, Wang et al., 2020)."""
    tau = max(int(tau), 1)
    return (tau - rho * (1 - rho ** tau) / (1 - rho)) / (1 - rho) if rho > 0 else float(tau)


class Scaffold(FedAvg):
    """Karimireddy et al., ICML 2020 (control variates, option II).

    With momentum SGD the accumulated update is ~a_k * lr * gradient rather than
    tau * lr * gradient, so option-II control variates use a_k (FedNova's
    normaliser); with rho=0 this is exactly the original rule.
    """
    name = "scaffold"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.c_global, self.c_local = None, {}

    def client_start(self, k, client, local, global_model):
        ctx = super().client_start(k, client, local, global_model)
        zeros = lambda: [torch.zeros_like(p) for p in global_model.parameters()]
        if self.c_global is None:
            self.c_global = zeros()
        if k not in self.c_local:
            self.c_local[k] = zeros()
        ctx["gparams"] = [p.detach().clone() for p in global_model.parameters()]
        ctx["corr"] = [cg - cl for cg, cl in zip(self.c_global, self.c_local[k])]
        return ctx

    def after_backward(self, ctx, model):
        for p, c in zip(model.parameters(), ctx["corr"]):
            if p.grad is not None:
                p.grad.add_(c)

    def client_end(self, ctx, local, client, runner):
        k = ctx["k"]
        scale = 1.0 / (effective_steps(ctx["steps"], self.hp.get("momentum", 0.0)) * runner.lr)
        new = [cl - cg + (g - p.detach()) * scale for cl, cg, g, p in
               zip(self.c_local[k], self.c_global, ctx["gparams"], local.parameters())]
        delta = [n - o for n, o in zip(new, self.c_local[k])]
        self.c_local[k] = new
        return {"delta_c": delta}

    def aggregate(self, global_model, results):
        super().aggregate(global_model, results)
        for r in results:
            for cg, dc in zip(self.c_global, r["payload"]["delta_c"]):
                cg.add_(dc / self.K)

    def extra_upload_floats(self, payload, n_params):
        return n_params


class FedNova(FedAvg):
    """Wang et al., NeurIPS 2020: normalised averaging of local updates."""
    name = "fednova"

    def aggregate(self, global_model, results):
        w = self.sample_weights(results)
        rho = self.hp.get("momentum", 0.9)
        a = np.asarray([effective_steps(r["steps"], rho) for r in results])
        tau_eff = float((w * a).sum())
        gstate = global_model.state_dict()
        new = weighted_average_states([r["state"] for r in results], w)   # buffers
        for name, _ in global_model.named_parameters():
            g = gstate[name].double()
            d = sum(wk * (g - r["state"][name].double()) / ak for wk, ak, r in zip(w, a, results))
            new[name] = (g - tau_eff * d).to(gstate[name].dtype)
        global_model.load_state_dict(new)


class Moon(FedAvg):
    """Li, He and Song, CVPR 2021: model-contrastive loss on penultimate features."""
    name = "moon"
    defaults = {"mu": 1.0, "temperature": 0.5}

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.prev: Dict[int, nn.Module] = {}

    def client_start(self, k, client, local, global_model):
        ctx = super().client_start(k, client, local, global_model)
        ctx["gmodel"] = _frozen_copy(global_model)
        ctx["pmodel"] = self.prev.get(k, ctx["gmodel"])
        return ctx

    def loss(self, ctx, model, x, y):
        h, z = model(x, return_features=True)
        with torch.no_grad():
            hg = ctx["gmodel"](x, return_features=True)[0]
            hp = ctx["pmodel"](x, return_features=True)[0]
        logits = torch.stack([F.cosine_similarity(h, hg, dim=1),
                              F.cosine_similarity(h, hp, dim=1)], 1) / self.hp["temperature"]
        con = F.cross_entropy(logits, torch.zeros(len(y), dtype=torch.long))
        return F.cross_entropy(z, y) + self.hp["mu"] * con

    def client_end(self, ctx, local, client, runner):
        self.prev[ctx["k"]] = _frozen_copy(local)
        return {}


class FedRS(FedAvg):
    """Li and Zhan, KDD 2021: restricted softmax (logits of locally absent classes x alpha)."""
    name = "fedrs"
    defaults = {"alpha": 0.5}

    def client_start(self, k, client, local, global_model):
        ctx = super().client_start(k, client, local, global_model)
        ctx["mask"] = torch.where(torch.tensor(client["counts"] > 0),
                                  torch.tensor(1.0), torch.tensor(self.hp["alpha"]))
        return ctx

    def loss(self, ctx, model, x, y):
        return F.cross_entropy(model(x) * ctx["mask"], y)


class FedLC(FedAvg):
    """Zhang et al., ICML 2022: logit calibration z_c - tau * n_c^(-1/4)."""
    name = "fedlc"
    defaults = {"tau": 1.0}

    def client_start(self, k, client, local, global_model):
        ctx = super().client_start(k, client, local, global_model)
        n = np.maximum(client["counts"].astype(np.float64), 1e-8)
        ctx["shift"] = torch.tensor(self.hp["tau"] * n ** -0.25, dtype=torch.float32)
        return ctx

    def loss(self, ctx, model, x, y):
        return F.cross_entropy(model(x) - ctx["shift"], y)


class FedCARE(FedAvg):
    """Proposed: Federated Class-Aware REcalibration.

    Client: prior-corrected (balanced-softmax) objective CE(z + tau*log pi_k, y),
    where pi_k is the client's add-one-smoothed label distribution, so locally
    absent classes are not pushed down by the local softmax.
    Server: FedAvg for the feature extractor and class-aware aggregation of the
    classifier, i.e. row c of the global classifier is the average of client rows
    c weighted by the clients' counts of class c (only label counts are shared).
    Options evaluated as ablations (off by default): alignment of l2-normalised
    features to global low-rank (PCA) class subspaces (beta > 0); post-hoc
    low-rank classifier calibration (always reported as ``fedcare+lrcal``).
    """
    name = "fedcare"
    defaults = {"tau_la": 1.0, "class_aware_head": True, "beta": 0.0, "rank": 4,
                "prior_rank": None, "gamma": 0.1, "stats_cap": 32, "final_stats_cap": 64,
                "n_virtual": 128, "cal_steps": 100, "cal_lr": 0.05}

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        if self.hp["prior_rank"] is None:
            self.hp["prior_rank"] = self.hp["rank"]
        self.prior = None
        self.cache: Dict[int, Dict] = {}

    def client_start(self, k, client, local, global_model):
        ctx = super().client_start(k, client, local, global_model)
        counts = client["counts"].astype(np.float64)
        pi = (counts + 1.0) / (counts.sum() + self.C)
        ctx["log_prior"] = torch.tensor(self.hp["tau_la"] * np.log(pi), dtype=torch.float32)
        ctx["prior"] = self.prior
        if self.hp["beta"] > 0:   # summaries of the received global model (common feature space)
            ctx["stats"] = self.runner.feature_stats_for(local, client, k, self.hp["rank"],
                                                         self.hp["stats_cap"], normalize=True)
        return ctx

    def loss(self, ctx, model, x, y):
        h, z = model(x, return_features=True)
        loss = F.cross_entropy(z + ctx["log_prior"], y)
        pr = ctx["prior"]
        if pr is not None and self.hp["beta"] > 0:
            ok = pr["valid"][y]
            if bool(ok.any()):
                loss = loss + self.hp["beta"] * subspace_distance(h[ok], y[ok], pr,
                                                                  self.hp["gamma"]).mean()
        return loss

    def client_end(self, ctx, local, client, runner):
        return {"stats": ctx["stats"]} if "stats" in ctx else {}

    def aggregate(self, global_model, results):
        w = self.sample_weights(results)
        states = [r["state"] for r in results]
        new = weighted_average_states(states, w)
        if self.hp["class_aware_head"]:
            counts = np.stack([r["counts"] for r in results]).astype(np.float64)   # K x C
            Nc = counts.sum(0)
            wc = np.where(Nc[None] > 0, counts / np.maximum(Nc, 1.0)[None], w[:, None])
            for key in ("head.weight", "head.bias"):
                stacked = torch.stack([s[key].double() for s in states])          # K x C (x d)
                wt = torch.tensor(wc).reshape(len(states), self.C, *([1] * (stacked.dim() - 2)))
                new[key] = (stacked * wt).sum(0).to(states[0][key].dtype)
        global_model.load_state_dict(new)
        for r in results:
            if "stats" in r["payload"]:
                self.cache[r["k"]] = r["payload"]["stats"]
        if self.cache:
            merged = merge_stats(list(self.cache.values()))
            if all(np.isfinite(merged[k]).all() for k in ("mean", "within")):
                self.prior = subspace_prior(merged, self.hp["prior_rank"])

    def finalize(self, global_model, runner):
        m = copy.deepcopy(global_model)
        stats = runner.client_feature_stats(m, self.hp["rank"], self.hp["final_stats_cap"])
        calibrate_head(m.head, merge_stats(stats), self.hp["n_virtual"], self.hp["cal_steps"],
                       self.hp["cal_lr"], seed=runner.seed)
        return {"fedcare+lrcal": m}

    def extra_upload_floats(self, payload, n_params):
        return self.C + (stats_num_floats(payload["stats"]) if "stats" in payload else 0)


ALGORITHMS = {cls.name: cls for cls in (FedProx, Scaffold, FedNova, Moon, FedRS, FedLC, FedCARE)}
ALGORITHMS["fedavg"] = FedAvgPostHoc


def make_algorithm(name: str, hp: Dict, num_classes: int, feat_dim: int, num_clients: int):
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown method {name!r}; choose from {sorted(ALGORITHMS)}")
    return ALGORITHMS[name](hp, num_classes, feat_dim, num_clients)
