#!/usr/bin/env python
"""Aggregate raw run files (results/raw) into the paper's tables and statistics.

  python scripts/summarize.py            -> results/summary/*.csv, *.md, paper_numbers.json
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd
from scipy import stats

METRICS = ["acc", "bacc", "macro_f1", "macro_auc", "worst_recall", "tail3_recall"]
BASELINES = ["fedavg", "fedprox", "scaffold", "fednova", "moon", "fedrs", "fedlc",
             "fedavg+ccvr", "fedavg+lrcal"]
MAIN = BASELINES + ["fedcare"]
ABLATION = ["fedcare", "fedcare-noCAH", "fedcare-noPC", "fedcare+align", "fedcare+proto",
            "fedcare+lrcal"]
LABEL = {"fedavg": "FedAvg", "fedprox": "FedProx", "scaffold": "SCAFFOLD", "fednova": "FedNova",
         "moon": "MOON", "fedrs": "FedRS", "fedlc": "FedLC", "fedavg+ccvr": "FedAvg+CCVR",
         "fedavg+lrcal": "FedAvg+LR-Cal", "fedcare": "FedCARE (ours)",
         "fedcare+lrcal": "FedCARE + LR-Cal", "fedcare-noCAH": "w/o class-aware head",
         "fedcare-noPC": "w/o prior correction", "fedcare+align": "+ subspace alignment",
         "fedcare+proto": "+ prototype alignment"}
SETTINGS = ["blood-shift", "derma"]


def load(root):
    rows, runs = [], {}
    for f in sorted(glob.glob(os.path.join(root, "*", "*", "seed*.json"))):
        setting, tag = f.split(os.sep)[-3], f.split(os.sep)[-2]
        r = json.load(open(f))
        runs[(setting, tag, r["seed"])] = r
        for variant, ev in r["final"].items():
            if variant == r["method"]:
                name = tag
            elif tag == r["method"]:
                name = variant                       # post-hoc variants of main runs
            else:
                continue                             # post-hoc variants of ablation runs
            row = {"setting": setting, "variant": name, "seed": r["seed"],
                   "val_macro_f1": ev["val"]["macro_f1"], "train_time_s": r["train_time_s"],
                   "extra_floats": r["extra_upload_floats_per_client_round"], "n_params": r["n_params"]}
            row.update({m: ev["test"][m] for m in METRICS})
            rows.append(row)
    return pd.DataFrame(rows), runs


def mean_std(df, variants):
    out = {}
    for (s, v), g in df[df.variant.isin(variants)].groupby(["setting", "variant"]):
        out.setdefault(s, {})[v] = {m: [float(g[m].mean()), float(g[m].std(ddof=1)) if len(g) > 1 else 0.0]
                                    for m in METRICS + ["val_macro_f1", "train_time_s"]}
        out[s][v]["n"] = int(len(g))
        out[s][v]["seeds_macro_f1"] = {int(k): float(x) for k, x in zip(g["seed"], g["macro_f1"])}
    return out


def paired_tests(df, ref="fedcare", metrics=("macro_f1", "bacc", "worst_recall")):
    res = {}
    piv = {m: df.pivot_table(index=["setting", "seed"], columns="variant", values=m) for m in metrics}
    for b in BASELINES:
        res[b] = {}
        for m in metrics:
            p = piv[m]
            if b not in p or ref not in p:
                continue
            pair = p[[ref, b]].dropna()
            d = (pair[ref] - pair[b]).values
            if len(d) < 2:
                continue
            w = stats.wilcoxon(d, alternative="two-sided", method="exact") if np.any(d != 0) else None
            t = stats.ttest_rel(pair[ref], pair[b])
            res[b][m] = {"n": int(len(d)), "mean_diff": float(d.mean()), "wins": int((d > 0).sum()),
                         "wilcoxon_p": float(w.pvalue) if w else 1.0, "ttest_p": float(t.pvalue)}
    return res


def per_class(runs, settings, variants):
    out = {}
    for s in settings:
        for v in variants:
            recs = [r for (ss, tag, _), r in runs.items() if ss == s and tag == v.split("+")[0]
                    and (v in r["final"])]
            if not recs:
                continue
            pcs = [r["final"][v]["test"]["per_class"] for r in recs]
            cm = np.sum([np.array(r["final"][v]["test"]["confusion"]) for r in recs], axis=0)
            out.setdefault(s, {})[v] = {
                k: np.mean([pc[k] for pc in pcs], axis=0).round(4).tolist()
                for k in ("precision", "recall", "f1", "support")}
            out[s][v]["confusion_sum"] = cm.tolist()
            out[s][v]["class_names"] = recs[0]["class_names"]
    return out


def history(runs, settings, variants, key="macro_f1"):
    out = {}
    for s in settings:
        for v in variants:
            hs = [r["history"] for (ss, tag, _), r in runs.items() if ss == s and tag == v]
            if not hs:
                continue
            rounds = [e["round"] for e in hs[0]]
            vals = np.array([[e["test"][key] for e in h] for h in hs])
            out.setdefault(s, {})[v] = {"round": rounds, "mean": vals.mean(0).tolist(),
                                        "std": vals.std(0, ddof=1).tolist() if len(hs) > 1 else [0] * len(rounds)}
    return out


def fmt(ms, digits=3):
    return f"{ms[0]:.{digits}f} ± {ms[1]:.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    root = os.path.join(os.path.dirname(__file__), "..", "results")
    ap.add_argument("--raw", default=os.path.join(root, "raw"))
    ap.add_argument("--out", default=os.path.join(root, "summary"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df, runs = load(a.raw)
    df.to_csv(os.path.join(a.out, "all_runs.csv"), index=False)
    main_tab = mean_std(df, MAIN)
    abl_tab = mean_std(df, ABLATION)
    tests = paired_tests(df)
    meta = {}
    for s in SETTINGS:
        r = next((r for (ss, tag, sd), r in runs.items() if ss == s and sd == 0), None)
        if r:
            meta[s] = {"client_class_counts": r["meta"]["client_class_counts"],
                       "client_sizes": r["meta"]["client_sizes"],
                       "train_class_counts": r["meta"]["train_class_counts"],
                       "class_names": r["class_names"], "n_params": r["n_params"],
                       "feat_dim": r["feat_dim"], "setting": r["setting"], "train_cfg": r["train_cfg"],
                       "site_params": r["meta"].get("site_params")}
    numbers = {"main": main_tab, "ablation": abl_tab, "tests": tests, "meta": meta,
               "per_class": per_class(runs, SETTINGS, ["fedavg", "fedlc", "fedrs", "fedcare",
                                                       "fedavg+ccvr", "moon", "fedprox", "scaffold",
                                                       "fednova"]),
               "history": history(runs, SETTINGS, ["fedavg", "fedprox", "scaffold", "fednova", "moon",
                                                    "fedrs", "fedlc", "fedcare"]),
               "labels": LABEL, "n_runs": len(runs)}
    with open(os.path.join(a.out, "paper_numbers.json"), "w") as f:
        json.dump(numbers, f, indent=1)
    # markdown tables (README / quick look)
    lines = []
    for s in SETTINGS:
        if s not in main_tab:
            continue
        lines += [f"\n### {s}\n", "| Method | Acc | Bal. acc | Macro-F1 | Macro-AUC | Worst-class recall | n |",
                  "|---|---|---|---|---|---|---|"]
        for v in MAIN:
            if v in main_tab[s]:
                t = main_tab[s][v]
                lines.append(f"| {LABEL[v]} | {fmt(t['acc'])} | {fmt(t['bacc'])} | {fmt(t['macro_f1'])} | "
                             f"{fmt(t['macro_auc'])} | {fmt(t['worst_recall'])} | {t['n']} |")
    lines += ["\n### FedCARE vs baselines (macro-F1, paired over settings x seeds)\n",
              "| Baseline | n | mean diff | wins | Wilcoxon p | paired t p |", "|---|---|---|---|---|---|"]
    for b, d in tests.items():
        if "macro_f1" in d:
            t = d["macro_f1"]
            lines.append(f"| {LABEL[b]} | {t['n']} | {t['mean_diff']:+.3f} | {t['wins']}/{t['n']} | "
                         f"{t['wilcoxon_p']:.4f} | {t['ttest_p']:.4f} |")
    with open(os.path.join(a.out, "tables.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n{len(runs)} run files summarised -> {a.out}")


if __name__ == "__main__":
    main()
