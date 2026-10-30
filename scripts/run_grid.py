#!/usr/bin/env python
"""Run every experiment of a grid file sequentially; finished runs are skipped,
so the script can be interrupted and restarted at any time.

  python scripts/run_grid.py configs/grids/main.yaml [--only main] [--dry-run]
"""
import argparse, os, sys, time
import yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fedcare.runner import run_and_save  # noqa: E402
from run import ROOT, load_cfgs  # noqa: E402


def jobs_from_grid(path):
    with open(path) as f:
        g = yaml.safe_load(f)
    jobs = []
    for b in g["blocks"]:
        variants = {m: {"method": m, "hp": {}} for m in b.get("methods", [])}
        variants.update(b.get("variants", {}))
        for seed in b["seeds"]:
            for setting in b["settings"]:
                for tag, v in variants.items():
                    jobs.append({"block": b["name"], "setting": setting, "tag": tag,
                                 "method": v["method"], "hp": v.get("hp", {}),
                                 "train": v.get("train", {}), "seed": seed})
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("--only", nargs="*", help="restrict to these block names")
    ap.add_argument("--data-root", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results", "raw"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    jobs = [j for j in jobs_from_grid(a.grid) if not a.only or j["block"] in a.only]
    todo = [j for j in jobs if not os.path.exists(
        os.path.join(a.out_dir, j["setting"], j["tag"], f"seed{j['seed']}.json"))]
    print(f"{len(jobs)} jobs, {len(todo)} to run", flush=True)
    for i, j in enumerate(todo):
        out = os.path.join(a.out_dir, j["setting"], j["tag"], f"seed{j['seed']}.json")
        print(f"[{i + 1}/{len(todo)}] {j['setting']} {j['tag']} seed{j['seed']}", flush=True)
        if a.dry_run:
            continue
        s, t, hp = load_cfgs(j["setting"], j["method"], j["hp"], j["train"])
        t0 = time.time()
        run_and_save(s, t, j["method"], hp, j["seed"], a.data_root, out,
                     log=lambda m: print("   ", m, flush=True))
        print(f"    done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
