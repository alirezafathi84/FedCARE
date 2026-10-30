"""Federated simulation loop, evaluation and result logging."""
from __future__ import annotations

import copy
import json
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

from .algorithms import make_algorithm
from .data import augment_batch, build_setting, to_tensor
from .models import make_model
from .stats import capped_indices, class_stats, extract_features


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y: np.ndarray, prob: np.ndarray, num_classes: int, full: bool = False) -> Dict:
    pred = prob.argmax(1)
    labels = list(range(num_classes))
    p, r, f1, sup = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    present = sup > 0
    rec = r[present]
    m = {"acc": float((pred == y).mean()),
         "bacc": float(rec.mean()),
         "macro_f1": float(f1[present].mean()),
         "worst_recall": float(rec.min()),
         "tail3_recall": float(np.sort(rec)[:3].mean())}
    try:
        m["macro_auc"] = float(roc_auc_score(y, prob, multi_class="ovr", average="macro", labels=labels))
    except ValueError:
        m["macro_auc"] = float("nan")
    if full:
        m["per_class"] = {"precision": p.tolist(), "recall": r.tolist(), "f1": f1.tolist(),
                          "support": sup.tolist()}
        m["confusion"] = confusion_matrix(y, pred, labels=labels).tolist()
    return m


@torch.no_grad()
def predict_proba(model, x: torch.Tensor, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    out = [F.softmax(model(x[i:i + batch_size]).float(), 1) for i in range(0, len(x), batch_size)]
    return torch.cat(out).numpy()


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
class Runner:
    def __init__(self, setting_cfg: Dict, train_cfg: Dict, method: str, method_hp: Optional[Dict],
                 seed: int, data_root: str, log=print):
        self.setting_cfg, self.cfg, self.method, self.seed = setting_cfg, train_cfg, method, seed
        self.log = log
        self.lr = train_cfg["lr"]
        self.cl = bool(train_cfg.get("channels_last", False))
        torch.set_num_threads(int(train_cfg.get("threads", 1)))

        S = build_setting(setting_cfg, seed, data_root)
        self.setting = S
        self.C = S["num_classes"]
        self.aug = S["augment"]
        self.clients = []
        for k, c in enumerate(S["clients"]):
            self.clients.append({"x": to_tensor(c["x"], self.cl), "y": torch.from_numpy(c["y"]),
                                 "y_np": c["y"], "counts": np.bincount(c["y"], minlength=self.C),
                                 "domain": c["domain"], "n": len(c["y"])})
        self.val = {"x": to_tensor(S["val"]["x"], self.cl), "y": S["val"]["y"]}
        self.tests = {k: {"x": to_tensor(v["x"], self.cl), "y": v["y"]} for k, v in S["tests"].items()}
        rng = np.random.default_rng(seed + 12345)
        n_sub = train_cfg.get("test_subsample")
        self.test_sub = {k: (np.sort(rng.choice(len(v["y"]), n_sub, replace=False))
                             if n_sub and n_sub < len(v["y"]) else np.arange(len(v["y"])))
                         for k, v in self.tests.items()}

        torch.manual_seed(seed)                     # identical initial model for every method
        self.model = make_model(train_cfg["model"], S["in_ch"], self.C, train_cfg.get("width"))
        if self.cl:
            self.model = self.model.to(memory_format=torch.channels_last)
        self.local = copy.deepcopy(self.model)
        hp = dict(method_hp or {})
        self.algo = make_algorithm(method, hp, self.C, self.model.feat_dim, len(self.clients))
        self.algo.hp.setdefault("momentum", train_cfg["momentum"])
        self.algo.runner = self
        self.n_params = sum(v.numel() for v in self.model.state_dict().values()
                            if torch.is_floating_point(v))

    # ---- helpers used by algorithms ---------------------------------------- #
    def feature_stats_for(self, model, client, k, rank, cap, normalize=False):
        rng = np.random.default_rng([self.seed, self.algo.rnd, int(k)])
        idx = capped_indices(client["y_np"], cap, rng)
        h = extract_features(model, client["x"][idx])
        if normalize:
            h = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
        return class_stats(h, client["y_np"][idx], client["counts"], self.C, rank)

    def client_feature_stats(self, model, rank, cap):
        return [self.feature_stats_for(model, c, k, rank, cap) for k, c in enumerate(self.clients)]

    # ---- evaluation ---------------------------------------------------------- #
    def evaluate(self, model, full: bool = False, subsample: bool = True) -> Dict:
        res = {"val": compute_metrics(self.val["y"], predict_proba(model, self.val["x"]), self.C, full)}
        ys, ps = [], []
        for name, t in self.tests.items():
            idx = self.test_sub[name] if subsample else np.arange(len(t["y"]))
            prob = predict_proba(model, t["x"][idx])
            res[name] = compute_metrics(t["y"][idx], prob, self.C, full)
            ys.append(t["y"][idx]); ps.append(prob)
        if len(self.tests) > 1:                     # multi-domain: pooled + worst domain
            res["pooled"] = compute_metrics(np.concatenate(ys), np.concatenate(ps), self.C, full)
            res["worst_domain_acc"] = min(res[n]["acc"] for n in self.tests)
            res["worst_domain_f1"] = min(res[n]["macro_f1"] for n in self.tests)
        return res

    # ---- training ------------------------------------------------------------ #
    def run(self) -> Dict:
        cfg, algo = self.cfg, self.algo
        R, E, bs = cfg["rounds"], cfg["local_epochs"], cfg["batch_size"]
        part = cfg.get("participation", 1.0)
        rng = np.random.default_rng(self.seed + 999)
        history, upload_extra = [], []
        t0 = time.time()
        for rnd in range(R):
            algo.rnd = rnd
            K = len(self.clients)
            sel = (np.arange(K) if part >= 1.0 else
                   np.sort(rng.choice(K, max(1, int(round(part * K))), replace=False)))
            results = []
            for k in sel:
                c = self.clients[k]
                self.local.load_state_dict(self.model.state_dict())
                opt = torch.optim.SGD(self.local.parameters(), lr=self.lr, momentum=cfg["momentum"],
                                      weight_decay=cfg["weight_decay"])
                ctx = algo.client_start(k, c, self.local, self.model)
                self.local.train()
                gen = torch.Generator().manual_seed(self.seed * 1_000_003 + rnd * 1009 + int(k))
                steps = 0
                for _ in range(E):
                    perm = torch.randperm(c["n"], generator=gen)
                    for i in range(0, c["n"], bs):
                        idx = perm[i:i + bs]
                        if len(idx) < 2:
                            continue
                        x = augment_batch(c["x"][idx], self.aug, gen)
                        if self.cl:
                            x = x.contiguous(memory_format=torch.channels_last)
                        opt.zero_grad(set_to_none=True)
                        loss = algo.loss(ctx, self.local, x, c["y"][idx])
                        loss.backward()
                        algo.after_backward(ctx, self.local)
                        opt.step()
                        steps += 1
                ctx["steps"] = steps
                payload = algo.client_end(ctx, self.local, c, self)
                upload_extra.append(algo.extra_upload_floats(payload, self.n_params))
                results.append({"k": int(k), "n": c["n"], "counts": c["counts"], "steps": steps,
                                "state": {n: v.detach().clone() for n, v in self.local.state_dict().items()},
                                "payload": payload})
            algo.aggregate(self.model, results)
            if (rnd + 1) % cfg.get("eval_every", 5) == 0 or rnd == R - 1:
                ev = self.evaluate(self.model)
                ev["round"] = rnd + 1
                ev["time_s"] = round(time.time() - t0, 1)
                history.append(ev)
                main = next(iter(self.tests))
                self.log(f"[{self.method} s{self.seed}] round {rnd + 1:3d}  val F1 {ev['val']['macro_f1']:.4f}"
                         f"  {main} acc {ev[main]['acc']:.4f} F1 {ev[main]['macro_f1']:.4f}  ({ev['time_s']}s)")
        train_time = time.time() - t0
        variants = algo.finalize(self.model, self)      # may re-calibrate self.model in place
        final = {self.method: self.evaluate(self.model, full=True, subsample=False)}
        for name, m in variants.items():
            final[name] = self.evaluate(m, full=True, subsample=False)
        return {"method": self.method, "seed": self.seed, "setting": self.setting_cfg,
                "train_cfg": self.cfg, "hp": algo.hp, "meta": self.setting["meta"],
                "class_names": self.setting["class_names"], "history": history, "final": final,
                "n_params": self.n_params, "feat_dim": self.model.feat_dim,
                "extra_upload_floats_per_client_round": float(np.mean(upload_extra)) if upload_extra else 0.0,
                "train_time_s": round(train_time, 1), "total_time_s": round(time.time() - t0, 1)}


def run_and_save(setting_cfg, train_cfg, method, hp, seed, data_root, out_file, log=print) -> Dict:
    res = Runner(setting_cfg, train_cfg, method, hp, seed, data_root, log).run()
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    tmp = out_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    os.replace(tmp, out_file)
    return res
