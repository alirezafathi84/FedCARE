#!/usr/bin/env python
"""Generate every figure of the paper from results/summary/paper_numbers.json.

  python scripts/summarize.py && python scripts/make_figures.py   -> results/figures/*.png
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "results", "figures")
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.6,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
COL = {"fedavg": "#7f7f7f", "fedprox": "#1f77b4", "scaffold": "#9467bd", "fednova": "#8c564b",
       "moon": "#e377c2", "fedrs": "#bcbd22", "fedlc": "#17becf", "fedcare": "#d62728",
       "fedavg+ccvr": "#ff7f0e", "fedavg+lrcal": "#2ca02c"}
TITLE = {"blood-shift": "BloodMNIST (label skew + site shift)",
         "derma": "DermaMNIST (long tail + label skew)"}
SETTINGS = ["blood-shift", "derma"]


def box(ax, x, y, w, h, text, fc="#f2f2f2", ec="#333333", ls="-", fs=6.2, weight="normal", ha="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec,
                                lw=0.8, ls=ls))
    tx = x + w / 2 if ha == "center" else x + 1.2
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, weight=weight, linespacing=1.35)


def arrow(ax, p, q, text="", color="#333333", ls="-", rad=0.0, fs=5.8, off=(0, 1.2)):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.9, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], text, ha="center", va="bottom",
                fontsize=fs, color=color)


def _layout_problems(fig, texts, boxes, arrows, marks):
    """Return a list of overlaps: text crossing its own box border, text touching another
    box, text/text overlaps, and arrows or histogram bars crossing text."""
    fig.set_dpi(plt.rcParams["savefig.dpi"])                # measure at the export resolution
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pad = 8.0                                               # pixels of clearance (~2 pt at 300 dpi)
    tb = [(t, t.get_window_extent(r), own) for t, own in texts]
    bb = [(bx, bx.get_window_extent(r)) for bx in boxes]
    out = []
    for t, e, own in tb:
        name = t.get_text()
        for bx, be in bb:
            if bx is own:
                if not (e.x0 >= be.x0 + pad and e.x1 <= be.x1 - pad and e.y0 >= be.y0 + pad and e.y1 <= be.y1 - pad):
                    out.append(f"text crosses its box border: {name!r}")
            elif e.overlaps(be):
                out.append(f"text touches another box: {name!r}")
        for a in list(arrows) + list(marks):
            if e.overlaps(a.get_window_extent(r)):
                out.append(f"arrow/bar crosses text: {name!r}")
    for i in range(len(tb)):
        for j in range(i + 1, len(tb)):
            if tb[i][1].overlaps(tb[j][1]):
                out.append(f"texts overlap: {tb[i][0].get_text()!r} / {tb[j][0].get_text()!r}")
    return out


def fig_framework():
    fig, ax = plt.subplots(figsize=(7.1, 3.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 43); ax.axis("off")
    texts, boxes, arrows, marks = [], [], [], []
    blue, red, green, grey = "#1f4e79", "#a8322d", "#1e6e1e", "#444444"

    def T(x, y, s, own=None, **kw):
        kw.setdefault("va", "center")
        t = ax.text(x, y, s, **kw); texts.append((t, own)); return t

    def B(x, y, w, h, fc, ec, ls="-"):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.0", fc=fc, ec=ec, lw=0.8, ls=ls)
        ax.add_patch(p); boxes.append(p); return p

    def A(p, q, color, ls="-"):
        a = FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.9, color=color, ls=ls,
                            shrinkA=0, shrinkB=0)
        ax.add_patch(a); arrows.append(a); return a

    # (1) clients
    T(14.0, 41.4, "(1) Local prior-corrected training", ha="center", fontsize=6.6, weight="bold", color=blue)
    rng = np.random.default_rng(3)
    ys, ch = [28.8, 16.8, 4.8], 9.6
    for i, yb in enumerate(ys):
        cb = B(1.0, yb, 27.6, ch, "#eaf1fb", blue)
        T(2.0, yb + 8.1, f"Client {['1', '2', 'K'][i]} (hospital / device)", cb, fontsize=6.0, weight="bold", color=blue)
        cnt = rng.dirichlet(np.full(8, 0.25))
        for c in range(8):
            marks.append(ax.add_patch(Rectangle((2.3 + c * 1.1, yb + 1.0), 0.8, 5.0 * cnt[c] / cnt.max(),
                                                fc="#1f77b4", ec="none")))
        T(12.3, yb + 5.7, "counts $n_{k,c}$ → prior $\pi_k$", cb, fontsize=5.4)
        T(12.3, yb + 3.5, r"$\ell=\mathrm{CE}(z+\tau\log\pi_k,\,y)$", cb, fontsize=5.6)
        T(12.3, yb + 1.5, "(prior-corrected loss)", cb, fontsize=5.0, color="#555555")
        A((29.2, yb + ch / 2), (39.4, yb + ch / 2), blue)
    T(34.3, ys[0] + ch / 2 + 2.2, r"upload $\theta_k$, $n_{k,c}$", fontsize=5.4, ha="center", color=blue)

    # (2) server
    sb = B(40.0, 4.8, 30.0, 33.6, "#fdf0ec", red)
    T(55.0, 36.2, "(2) Class-aware aggregation", sb, ha="center", fontsize=6.4, weight="bold", color=red)
    T(41.4, 32.0, "Feature extractor (sample-weighted):", sb, fontsize=5.8)
    T(43.0, 28.2, r"$\phi^{t+1}=\sum_k \frac{n_k}{N}\,\phi_k^{t+1}$", sb, fontsize=6.4)
    T(41.4, 24.2, "Classifier row of class $c$", sb, fontsize=5.8)
    T(41.4, 22.0, "(weighted by class counts):", sb, fontsize=5.8)
    T(43.0, 18.4, r"$w_c^{t+1}=\sum_k \frac{n_{k,c}}{N_c}\,w_{k,c}^{t+1}$", sb, fontsize=6.4)
    for j, line in enumerate(["clients without class $c$ get zero weight", "for row $c$; only label counts are",
                              "shared with the server"]):
        T(41.4, 13.0 - 2.2 * j, line, sb, fontsize=5.3, color="#555555")

    # broadcast path: down from the server, left along the bottom, up into the clients
    ax.plot([55.0, 55.0, 14.0], [4.4, 1.4, 1.4], color=red, lw=0.9, solid_capstyle="butt")
    A((14.0, 1.4), (14.0, 4.4), red)
    T(34.3, 2.6, r"broadcast $\theta^{t+1}$", fontsize=5.6, ha="center", color=red)

    # (3) LR-Cal and the ablation
    lb = B(73.0, 19.6, 26.5, 18.8, "#eef7ee", "#2ca02c", "--")
    T(86.25, 36.2, "(3) LR-Cal (optional)", lb, ha="center", fontsize=6.2, weight="bold", color=green)
    for j, line in enumerate(["clients upload rank-$r$ PCA", "summaries of class features",
                              r"$(n_{k,c},\mu_{k,c},U_{k,c},\Lambda_{k,c},\sigma^2_{k,c})$",
                              "server merges them, samples", "class-balanced virtual features",
                              "and re-fits the classifier"]):
        T(74.4, 32.8 - 2.15 * j, line, lb, fontsize=5.4)
    ab = B(73.0, 4.0, 26.5, 13.2, "#f6f6f6", "#777777", ":")
    T(86.25, 15.3, "Ablation: feature alignment", ab, ha="center", fontsize=6.2, weight="bold", color=grey)
    for j, line in enumerate([r"distance of $\hat h=h/\|h\|$ to the", r"class subspace $\mu_c+\mathrm{span}(U_c)$",
                              "(PCA of pooled within-client", "scatter); not used by default"]):
        T(74.4, 12.5 - 2.1 * j, line, ab, fontsize=5.4)
    A((70.6, 29.0), (72.6, 29.0), "#2ca02c", "--")
    A((70.6, 10.6), (72.6, 10.6), "#777777", ":")

    problems = _layout_problems(fig, texts, boxes, arrows, marks)
    for msg in problems:
        print("Fig. 1 layout:", msg)
    fig.savefig(os.path.join(OUT, "fig1_framework.png")); plt.close(fig)
    return problems


def fig_partition(N):
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.3))
    for ax, s in zip(axes, SETTINGS):
        m = N["meta"][s]
        cc = np.array(m["client_class_counts"])
        im = ax.imshow(cc, cmap="Blues", aspect="auto")
        for i in range(cc.shape[0]):
            for j in range(cc.shape[1]):
                if cc[i, j] > 0:
                    ax.text(j, i, str(cc[i, j]), ha="center", va="center", fontsize=4.6,
                            color="white" if cc[i, j] > cc.max() * 0.55 else "black")
        ax.set_xticks(range(cc.shape[1])); ax.set_xticklabels(m["class_names"], rotation=0)
        ax.set_yticks(range(cc.shape[0])); ax.set_yticklabels([f"C{i + 1}" for i in range(cc.shape[0])])
        a = m["setting"]["alpha"]
        ax.set_title(f"{TITLE[s]}, Dir($\\alpha$={a}), seed 0", fontsize=6.8)
        ax.set_ylabel("client")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).ax.tick_params(labelsize=5)
    fig.tight_layout(h_pad=0.8)
    fig.savefig(os.path.join(OUT, "fig2_partition.png")); plt.close(fig)


def fig_sites(N):
    from fedcare.data import apply_site_shift, load_medmnist
    d = load_medmnist("bloodmnist", os.path.join(ROOT, "data"))
    names = N["meta"]["blood-shift"]["class_names"]
    pick = [int(np.flatnonzero(d["test_y"] == c)[3]) for c in range(6)]
    imgs = d["test_x"][pick]
    params = N["meta"]["blood-shift"]["site_params"][:5]
    rows = [imgs] + [apply_site_shift(imgs, p, np.random.default_rng(i), "od") for i, p in enumerate(params)]
    fig, axes = plt.subplots(len(rows), 6, figsize=(3.45, 3.55))
    for i, r in enumerate(rows):
        for j in range(6):
            ax = axes[i, j]; ax.imshow(r[j]); ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(names[j], fontsize=6)
            if j == 0:
                ax.set_ylabel("original" if i == 0 else f"site {i}", fontsize=6)
    fig.subplots_adjust(wspace=0.04, hspace=0.04)
    fig.savefig(os.path.join(OUT, "fig3_sites.png")); plt.close(fig)


def fig_convergence(N):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.35))
    order = ["fedavg", "fedprox", "scaffold", "fednova", "moon", "fedrs", "fedlc", "fedcare"]
    for ax, s in zip(axes, SETTINGS):
        rounds = None
        for v in order:
            h = N["history"].get(s, {}).get(v)
            if not h:
                continue
            r, m, sd = np.array(h["round"]), np.array(h["mean"]), np.array(h["std"])
            rounds = h["round"]
            lw = 1.8 if v == "fedcare" else 1.0
            ax.plot(r, m, "-o", ms=2.2, lw=lw, color=COL[v], label=N["labels"][v].replace(" (ours)", ""),
                    zorder=3 if v == "fedcare" else 2)
            ax.fill_between(r, m - sd, m + sd, color=COL[v], alpha=0.12, lw=0)
        ax.set_title(TITLE[s]); ax.set_xlabel("communication round"); ax.set_ylabel("test macro-F1")
        ax.grid(alpha=0.3, lw=0.4)
        if rounds:
            ax.set_xticks(rounds)
    axes[1].legend(ncol=2, loc="lower right", frameon=False, handlelength=1.6)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(os.path.join(OUT, "fig4_convergence.png")); plt.close(fig)


def fig_perclass(N):
    meths = ["fedavg", "fedlc", "fedrs", "fedcare"]
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.7))
    for ax, s in zip(axes, SETTINGS):
        pc = N["per_class"][s]
        names = pc["fedavg"]["class_names"]
        x = np.arange(len(names)); wdt = 0.2
        for i, v in enumerate(meths):
            ax.bar(x + (i - 1.5) * wdt, pc[v]["f1"], wdt, color=COL[v], label=N["labels"][v].replace(" (ours)", ""))
        ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1.0); ax.set_ylabel("test F1")
        ax.set_title(TITLE[s]); ax.grid(axis="y", alpha=0.3, lw=0.4)
    axes[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.33), frameon=False)
    fig.tight_layout(h_pad=0.6)
    fig.savefig(os.path.join(OUT, "fig5_perclass.png")); plt.close(fig)


def fig_confusion(N):
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.05))
    for ax, (s, v) in zip(axes, [(s, v) for s in SETTINGS for v in ("fedavg", "fedcare")]):
        pc = N["per_class"][s][v]
        cm = np.array(pc["confusion_sum"], float)
        cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(cm, cmap="Reds", vmin=0, vmax=1)
        n = len(cm)
        for i in range(n):
            for j in range(n):
                if cm[i, j] >= 0.05:
                    ax.text(j, i, f"{cm[i, j] * 100:.0f}", ha="center", va="center", fontsize=3.9,
                            color="white" if cm[i, j] > 0.6 else "black")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(pc["class_names"], rotation=90, fontsize=4.6)
        ax.set_yticklabels(pc["class_names"], fontsize=4.6)
        ax.set_title(f"{N['labels'][v].replace(' (ours)', '')} - {s.split('-')[0].capitalize()}MNIST", fontsize=6.4)
        ax.set_xlabel("predicted", fontsize=5.5)
        if ax is axes[0] or ax is axes[2]:
            ax.set_ylabel("true", fontsize=5.5)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(os.path.join(OUT, "fig6_confusion.png")); plt.close(fig)


def fig_ablation(N):
    vs = ["fedcare", "fedcare-noCAH", "fedcare-noPC", "fedcare+align", "fedcare+proto", "fedcare+lrcal"]
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.3))
    for ax, s in zip(axes, SETTINGS):
        tab = N["ablation"].get(s, {})
        vv = [v for v in vs if v in tab]
        m = [tab[v]["macro_f1"][0] for v in vv]; e = [tab[v]["macro_f1"][1] for v in vv]
        y = np.arange(len(vv))[::-1]
        ax.barh(y, m, xerr=e, color=["#d62728"] + ["#bbbbbb"] * (len(vv) - 1), height=0.6,
                error_kw={"lw": 0.7, "capsize": 1.5})
        ax.set_yticks(y); ax.set_yticklabels([N["labels"][v].replace(" (ours)", "") for v in vv])
        ax.set_xlabel("test macro-F1 (mean ± s.d., 3 seeds)"); ax.set_title(TITLE[s])
        ax.set_xlim(0, max(np.array(m) + np.array(e)) * 1.15); ax.grid(axis="x", alpha=0.3, lw=0.4)
        for yy, mm in zip(y, m):
            ax.text(mm + 0.005, yy, f"{mm:.3f}", va="center", fontsize=5.4)
    fig.tight_layout(h_pad=0.8)
    fig.savefig(os.path.join(OUT, "fig7_ablation.png")); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    N = json.load(open(os.environ.get("NUMBERS", os.path.join(ROOT, "results", "summary", "paper_numbers.json"))))
    for f in (fig_framework, lambda: fig_partition(N), lambda: fig_sites(N), lambda: fig_convergence(N),
              lambda: fig_perclass(N), lambda: fig_confusion(N), lambda: fig_ablation(N)):
        try:
            f()
        except Exception as e:   # partial results: skip figures whose inputs are missing
            print("skipped figure:", repr(e))
    print("figures ->", OUT, sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
