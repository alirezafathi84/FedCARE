#!/usr/bin/env python
"""Run one federated experiment.

Example:
  python scripts/run.py --setting blood-shift --method fedcare --seed 0
  python scripts/run.py --setting derma --method fedcare --seed 0 --hp beta=0.0 --train rounds=50
"""
import argparse, os, sys
import yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fedcare.runner import run_and_save  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def parse_kv(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def load_cfgs(setting, method, hp_over=None, train_over=None):
    with open(os.path.join(ROOT, "configs", "settings", f"{setting}.yaml")) as f:
        s = yaml.safe_load(f)
    with open(os.path.join(ROOT, "configs", "train.yaml")) as f:
        t = yaml.safe_load(f)
    with open(os.path.join(ROOT, "configs", "methods.yaml")) as f:
        hp = dict(yaml.safe_load(f).get(method) or {})
    hp.update(hp_over or {})
    t.update(train_over or {})
    return s, t, hp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hp", nargs="*", help="method hyper-parameter overrides key=value")
    ap.add_argument("--train", nargs="*", help="training overrides key=value")
    ap.add_argument("--tag", default=None, help="result name (default: method)")
    ap.add_argument("--data-root", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results", "raw"))
    a = ap.parse_args()
    s, t, hp = load_cfgs(a.setting, a.method, parse_kv(a.hp), parse_kv(a.train))
    out = os.path.join(a.out_dir, a.setting, a.tag or a.method, f"seed{a.seed}.json")
    run_and_save(s, t, a.method, hp, a.seed, a.data_root, out)
    print("saved", out)


if __name__ == "__main__":
    main()
