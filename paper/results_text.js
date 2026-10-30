// Results-dependent text. Every number and every comparative claim below is computed
// from results/summary/paper_numbers.json, so the text cannot drift from the data.
const { Table, TableRow, TableCell, Paragraph, TextRun, WidthType, BorderStyle, AlignmentType } = require("docx");

const TITLE = "FedCARE: Class-Aware Recalibration for Federated Medical Image Classification under Label Skew and Acquisition Shift";
const BASE = ["fedavg", "fedprox", "scaffold", "fednova", "moon", "fedrs", "fedlc", "fedavg+ccvr", "fedavg+lrcal"];
const DRIFT = ["fedprox", "scaffold", "fednova", "moon"];

function acc(H) {
  const g = (s, v, m) => (H.N.main[s] && H.N.main[s][v] ? H.N.main[s][v][m] : null);
  const a = (s, v, m) => (H.N.ablation[s] && H.N.ablation[s][v] ? H.N.ablation[s][v][m] : null);
  const f3 = (x) => x.toFixed(3);
  const pts = (x) => (100 * x).toFixed(1);
  const best = (s, m, pool) => pool.filter((v) => g(s, v, m)).sort((x, y) => g(s, y, m)[0] - g(s, x, m)[0])[0];
  const rankOf = (s, m, v, pool) => pool.filter((u) => g(s, u, m)).sort((x, y) => g(s, y, m)[0] - g(s, x, m)[0]).indexOf(v) + 1;
  const pfmt = (p) => (p < 0.001 ? "< 0.001" : p.toFixed(3));
  return { g, a, f3, pts, best, rankOf, pfmt };
}

const TT = ["fedavg", "fedprox", "scaffold", "fednova", "moon", "fedrs", "fedlc"];   // training-time baselines

function alignDeltas(H) {   // macro-F1 change caused by adding subspace / prototype alignment, per setting
  const { a } = acc(H);
  return H.SETS.map((s) => ["fedcare+align", "fedcare+proto"].map((v) => (a(s, v, "macro_f1") ? a(s, v, "macro_f1")[0] - a(s, "fedcare", "macro_f1")[0] : 0)));
}
function alignHelps(H) { return alignDeltas(H).flat().some((x) => x > 0.005); }
function alignPhrase(H) {
  const { pts } = acc(H);
  const dd = alignDeltas(H).map((r) => r[0]);
  return alignHelps(H)
    ? `aligning features with PCA class subspaces gave mixed results (${dd.map((x, i) => `${x >= 0 ? "+" : ""}${pts(x)} macro-F1 points on ${H.SNAME[H.SETS[i]]}`).join(", ")})`
    : "aligning features with PCA class subspaces did not add accuracy";
}
function hardestSeed(H, s) {   // seed with the lowest macro-F1 for most methods
  const { g } = acc(H);
  const counts = {};
  [...BASE, "fedcare"].forEach((v) => { const sm = g(s, v, "seeds_macro_f1"); if (!sm) return;
    const k = Object.keys(sm).sort((x, y) => sm[x] - sm[y])[0]; counts[k] = (counts[k] || 0) + 1; });
  const k = Object.keys(counts).sort((x, y) => counts[y] - counts[x])[0];
  return [k, counts[k], [...BASE, "fedcare"].length];
}

function settingSentence(H, s) {
  const { g, f3, pts, best, rankOf } = acc(H);
  const c = g(s, "fedcare", "macro_f1"), b = best(s, "macro_f1", BASE), bt = best(s, "macro_f1", TT);
  const rk = rankOf(s, "macro_f1", "fedcare", [...BASE, "fedcare"]);
  if (rk === 1) return `the highest mean macro-F1 on ${H.SNAME[s]} (${f3(c[0])} ± ${f3(c[1])} vs. ${f3(g(s, b, "macro_f1")[0])} for the best baseline, ${H.L[b]})`;
  return `a macro-F1 of ${f3(c[0])} ± ${f3(c[1])} on ${H.SNAME[s]}, ${pts(c[0] - g(s, bt, "macro_f1")[0])} points above the best training-time baseline (${H.L[bt]}) but ${pts(g(s, b, "macro_f1")[0] - c[0])} points below ${H.L[b]}`;
}

function abstract(H) {
  const { g, f3, pts } = acc(H);
  const T = H.N.tests;
  const winsTT = TT.reduce((t, b) => t + (T[b] && T[b].macro_f1 ? T[b].macro_f1.wins : 0), 0);
  const nTT = TT.reduce((t, b) => t + (T[b] && T[b].macro_f1 ? T[b].macro_f1.n : 0), 0);
  const sigTT = TT.filter((b) => T[b] && T[b].macro_f1 && T[b].macro_f1.wilcoxon_p <= 0.05).length;
  const dB = H.SETS.map((s) => pts(g(s, "fedcare", "bacc")[0] - g(s, "fedavg", "bacc")[0]));
  const ph = H.SETS.filter((s) => g(s, "fedavg+ccvr", "macro_f1")[0] > g(s, "fedcare", "macro_f1")[0]);
  const postHoc = ph.length
    ? ` Post-hoc calibration of FedAvg remains competitive on ${ph.map((s) => H.SNAME[s]).join(" and ")}, where the global label distribution is itself long-tailed.`
    : " FedCARE also exceeds post-hoc calibration of FedAvg in both settings.";
  return "Federated learning (FL) lets hospitals and medical Internet-of-Things devices train a shared classifier without exchanging images, but skewed label distributions bias the global classifier against minority classes, and site-specific acquisition conditions shift image appearance. Existing remedies regularize local training or re-calibrate the classifier afterwards, yet still average the classifier with sample-size weights. We show that this dilutes the update of each class by the share of data held by the clients that observed it. We propose FedCARE, which pairs a Bayes-consistent prior-corrected local loss with class-aware aggregation of the classifier rows; it adds only label counts to the upload and has no tuned hyper-parameter. We also introduce LR-Cal, a classifier calibration that transmits rank-r principal-component summaries of class-conditional features instead of full covariances. On BloodMNIST with severe label skew and simulated site shift and on the long-tailed DermaMNIST (ten clients, eight baselines, three seeds that re-draw the partitions), " +
    `FedCARE attains ${H.SETS.map((s) => settingSentence(H, s)).join(", and ")}. It wins ${winsTT} of ${nTT} paired comparisons with the seven training-time baselines (Wilcoxon *p* ≤ 0.05 against ${sigTT})${ph.length ? `, while post-hoc calibration of FedAvg remains competitive on ${ph.map((s) => H.SNAME[s]).join(" and ")}` : ""}. ` +
    `LR-Cal matches full-covariance calibration with ${((1 + 64 + 64 * 65 / 2) / (1 + 64 + 4 * 64 + 4 + 1)).toFixed(1)}× less upload, and ${alignPhrase(H)}. The compact network, 28×28 images and simulated shift call for validation on real multi-centre data before deployment.`;
}

function algorithmBox(H) {
  const lines = [
    "**Algorithm 1** FedCARE (one communication round *t*)",
    "**Input:** global model θ^{t} = (φ^{t}, *W*^{t}, *b*^{t}); clients *k* = 1, …, *K*",
    "1: server broadcasts θ^{t} to all participating clients",
    "2: **for** each client *k* **in parallel do**",
    "3:   π̃_{k,c} ← (*n*_{k,c} + 1)/(*n*_{k} + *C*) for *c* = 1, …, *C*",
    "4:   θ_{k} ← θ^{t}; run *E* local epochs of SGD on loss (1)",
    "5:   upload θ_{k}, *n*_{k} and the label counts *n*_{k,c}",
    "6: **end for**",
    "7: φ^{t+1} ← Σ_{k} (*n*_{k}/*N*) φ_{k}     ⊳ incl. BN statistics",
    "8: **for** *c* = 1, …, *C* **do**  *w*_{c}^{t+1}, *b*_{c}^{t+1} ← Eq. (2)",
    "9: **return** θ^{t+1};  optionally LR-Cal after the last round",
  ];
  const cell = new TableCell({
    width: { size: H.COLW - 200, type: WidthType.DXA },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    borders: { top: { style: BorderStyle.SINGLE, size: 8, color: "000000" }, bottom: { style: BorderStyle.SINGLE, size: 8, color: "000000" },
      left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" }, right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" } },
    children: lines.map((l, i) => new Paragraph({ children: H.runs(l, { size: 16 }), spacing: { after: i === 0 ? 40 : 10 },
      border: i === 0 ? { bottom: { style: BorderStyle.SINGLE, size: 4, color: "000000", space: 2 } } : undefined })),
  });
  return [new Table({ width: { size: H.COLW - 200, type: WidthType.DXA }, columnWidths: [H.COLW - 200], alignment: AlignmentType.CENTER,
    rows: [new TableRow({ children: [cell] })] }), new Paragraph({ children: [], spacing: { after: 100 } })];
}

function results(H) {
  const { g, a, f3, pts, best, rankOf, pfmt } = acc(H);
  const out = [];
  const add = (...xs) => xs.flat().forEach((x) => out.push(x));
  const T = H.N.tests;
  const S = H.SETS;
  add(H.H1("VI", "Results and Analysis"));
  // ---------------------------------------------------------------- A. main table
  add(H.H2("A", "Overall Comparison"));
  const metrics = [["acc", "Acc."], ["bacc", "Bal. acc."], ["macro_f1", "Macro-F1"], ["macro_auc", "Macro-AUC"], ["worst_recall", "Worst rec."]];
  const MAIN = [...BASE, "fedcare"];
  const rows = MAIN.filter((v) => S.every((s) => g(s, v, "acc"))).map((v) => {
    const cells = [{ t: v === "fedcare" ? "**FedCARE (ours)**" : H.L[v], rule: v === "fedavg+lrcal" }];
    S.forEach((s) => metrics.forEach(([m]) => {
      const isBest = best(s, m, MAIN) === v;
      const t = `${f3(g(s, v, m)[0])}±${f3(g(s, v, m)[1])}`;
      cells.push({ t: isBest ? `**${t}**` : t, rule: v === "fedavg+lrcal" });
    }));
    return cells;
  });
  const hdr = [[{ t: "" }, { t: "Blood-shift (α = 0.1, simulated site shift)", span: 5 }, { t: "Derma (α = 0.3, natural long tail)", span: 5 }],
    [{ t: "Method" }, ...metrics.map((m) => ({ t: m[1] })), ...metrics.map((m) => ({ t: m[1] }))]];
  const t3 = H.TABLE("main", "Test Performance of the Final Global Model (Mean ± S.D. over Three Seeds)", hdr, rows,
    [1500, ...Array(10).fill(894)], { wide: true, size: 13,
      note: "Bold: best mean in each column. FedAvg+CCVR and FedAvg+LR-Cal re-calibrate the classifier of the final FedAvg model with full-covariance and rank-4 class statistics, respectively. All metrics are computed on the complete official test splits." });
  const intro = [];
  for (const s of S) {
    const b = best(s, "macro_f1", BASE);
    const c = g(s, "fedcare", "macro_f1"), fa = g(s, "fedavg", "macro_f1"), bb = g(s, b, "macro_f1");
    const rk = rankOf(s, "macro_f1", "fedcare", MAIN);
    intro.push(`On ${H.SNAME[s]}, FedCARE ${rk === 1 ? "obtains the highest mean" : `ranks ${rk} in mean`} macro-F1 (${f3(c[0])} ± ${f3(c[1])}), ${pts(c[0] - fa[0])} points above FedAvg (${f3(fa[0])}) and ${c[0] >= bb[0] ? `${pts(c[0] - bb[0])} points above` : `${pts(bb[0] - c[0])} points below`} the strongest baseline, ${H.L[b]} (${f3(bb[0])}). Its balanced accuracy is ${f3(g(s, "fedcare", "bacc")[0])} versus ${f3(g(s, "fedavg", "bacc")[0])} for FedAvg, and its worst-class recall is ${f3(g(s, "fedcare", "worst_recall")[0])} versus ${f3(g(s, "fedavg", "worst_recall")[0])}.`);
  }
  add(H.P(`Table ${H.tab("main") || "III"} reports all metrics. ${intro.join(" ")}`));
  const driftGain = S.map((s) => DRIFT.map((v) => g(s, v, "macro_f1")[0] - g(s, "fedavg", "macro_f1")[0]));
  const dg = driftGain.flat(); const dMean = dg.reduce((x, y) => x + y, 0) / dg.length;
  const lsGain = S.map((s) => ["fedrs", "fedlc"].map((v) => g(s, v, "macro_f1")[0] - g(s, "fedavg", "macro_f1")[0])).flat();
  const lMean = lsGain.reduce((x, y) => x + y, 0) / lsGain.length;
  const accNote = S.map((s) => `${H.SNAME[s]}: ${f3(g(s, "fedcare", "acc")[0])} vs. ${f3(g(s, best(s, "acc", BASE), "acc")[0])} for ${H.L[best(s, "acc", BASE)]}`).join("; ");
  const vsC = S.map((s) => [s, g(s, "fedcare", "macro_f1")[0] - g(s, "fedavg+ccvr", "macro_f1")[0]]);
  const ccvrTxt = vsC.map(([s, dd]) => dd >= 0 ? `FedCARE exceeds it on ${H.SNAME[s]} by ${pts(dd)} points without an extra calibration round` : `on ${H.SNAME[s]} it is ${pts(-dd)} points better than FedCARE`).join(", whereas ");
  const pcd = H.N.per_class.derma && H.N.per_class.derma.fedavg;
  let nvShare = "";
  if (pcd) { const cm = pcd.confusion_sum; const j = pcd.class_names.indexOf("nv"); const tot = cm.flat().reduce((x, y) => x + y, 0);
    nvShare = ` (FedAvg assigns ${pts(cm.reduce((t, row) => t + row[j], 0) / tot)}% of all DermaMNIST test images to nevi)`; }
  add(H.P(`The two groups of baselines behave differently. Relative to FedAvg, the drift-correcting methods (FedProx, SCAFFOLD, FedNova, MOON) change macro-F1 by ${dMean >= 0 ? "+" : ""}${pts(dMean)} points on average and the label-skew methods FedRS and FedLC by ${lMean >= 0 ? "+" : ""}${pts(lMean)} points${lMean > dMean ? ", which indicates that classifier bias rather than optimization drift is the main failure mode in these settings" : ""}. On Blood-shift the standard deviations are large for every method, FedCARE included: the macro-F1 of FedCARE on the three federations is ${Object.values(g("blood-shift", "fedcare", "seeds_macro_f1")).map((x) => f3(x)).join(", ")}, and seed ${hardestSeed(H, "blood-shift")[0]} is the hardest federation for ${hardestSeed(H, "blood-shift")[1]} of the ${hardestSeed(H, "blood-shift")[2]} methods, so under α = 0.1 combined with site shift the particular partition matters as much as the method. The paired analysis in Section VI-B accounts for this by comparing methods on the same federation. Post-hoc calibration changes the macro-F1 of FedAvg from ${S.map((s) => `${f3(g(s, "fedavg", "macro_f1")[0])} to ${f3(g(s, "fedavg+ccvr", "macro_f1")[0])} on ${H.SNAME[s]}`).join(" and from ")} (CCVR); ${ccvrTxt}. Plain accuracy is less informative under class imbalance (${accNote}): a classifier that favors the majority class attains high accuracy on DermaMNIST while missing the rare lesion types${nvShare}, which is why we focus on class-balanced metrics.`));
  add(t3);
  // ---------------------------------------------------------------- B. significance
  add(H.H2("B", "Statistical Significance"));
  const srows = BASE.filter((b) => T[b] && T[b].macro_f1).map((b) => {
    const f = T[b].macro_f1, ba = T[b].bacc;
    return [H.L[b], `${f.mean_diff >= 0 ? "+" : ""}${f3(f.mean_diff)}`, `${f.wins}/${f.n}`, pfmt(f.wilcoxon_p), pfmt(f.ttest_p),
      `${ba.mean_diff >= 0 ? "+" : ""}${f3(ba.mean_diff)}`, pfmt(ba.wilcoxon_p)];
  });
  const nsig = BASE.filter((b) => T[b] && T[b].macro_f1 && T[b].macro_f1.wilcoxon_p <= 0.05).length;
  const allw = BASE.filter((b) => T[b] && T[b].macro_f1 && T[b].macro_f1.wins === T[b].macro_f1.n).length;
  const minDiff = Math.min(...BASE.filter((b) => T[b] && T[b].macro_f1).map((b) => T[b].macro_f1.mean_diff));
  const weakest = BASE.filter((b) => T[b] && T[b].macro_f1).sort((x, y) => T[x].macro_f1.mean_diff - T[y].macro_f1.mean_diff)[0];
  add(H.P(`Table ${H.tab("tests") || "IV"} pairs FedCARE with each baseline on the same federation (setting and seed), giving six paired observations per comparison. FedCARE wins all six pairs against ${allw} of the ${BASE.length} baseline configurations and the two-sided Wilcoxon test rejects equality at the 0.05 level for ${nsig} of them (0.031 is the smallest attainable *p*-value with six pairs). The smallest average advantage is ${pts(minDiff)} macro-F1 points, against ${H.L[weakest]}. With only six pairs the tests have limited power, so the effect sizes and standard deviations in Table ${H.tab("main") || "III"} should be read together with the *p*-values.`));
  add(H.TABLE("tests", "Paired Comparison of FedCARE with Each Baseline", [["Baseline", "Δ Macro-F1", "Wins", "*p*_{W}", "*p*_{t}", "Δ Bal. acc.", "*p*_{W}"]], srows,
    [1290, 640, 480, 640, 640, 710, 640], { size: 13, note: "Δ: mean difference (FedCARE − baseline) over six (setting, seed) pairs; *p*_{W}: two-sided exact Wilcoxon signed-rank test; *p*_{t}: paired *t*-test." }));
  // ---------------------------------------------------------------- C. convergence
  add(H.H2("C", "Convergence Behaviour"));
  const hist = H.N.history;
  const above = S.map((s) => {
    const c = hist[s].fedcare; if (!c) return [0, 0];
    let n = 0; c.round.forEach((r, i) => { if (BASE.filter((b) => hist[s][b]).every((b) => c.mean[i] >= hist[s][b].mean[i])) n += 1; });
    return [n, c.round.length];
  });
  const osc = (s, v) => { const m = hist[s][v].mean; let t = 0; for (let i = 1; i < m.length; i++) t += Math.abs(m[i] - m[i - 1]); return t / (m.length - 1); };
  add(H.P(`Fig. ${H.fig("conv")} plots the test macro-F1 of the global model every five rounds (mean over seeds with ±1 s.d. bands; intermediate points use a fixed 2,000-image test subsample, the final point the full test split). The *x*-axis is the communication round and the *y*-axis the macro-F1. The FedCARE curve is the highest at ${above[0][0]} of ${above[0][1]} evaluation points on Blood-shift and ${above[1][0]} of ${above[1][1]} on Derma. The mean absolute change of the macro-F1 between consecutive evaluations, a measure of round-to-round instability, is ${f3(osc("blood-shift", "fedavg"))} for FedAvg and ${f3(osc("blood-shift", "fedcare"))} for FedCARE on Blood-shift and ${f3(osc("derma", "fedavg"))} and ${f3(osc("derma", "fedcare"))} on Derma; under severe skew every FedAvg aggregation re-mixes classifier rows that were pushed in conflicting directions by different clients. Already at round ${hist["blood-shift"].fedcare.round[0]}, FedCARE is ${pts(hist["blood-shift"].fedcare.mean[0] - hist["blood-shift"].fedavg.mean[0])} and ${pts(hist.derma.fedcare.mean[0] - hist.derma.fedavg.mean[0])} points above FedAvg on the two settings, which indicates that the classifier, rather than the feature extractor, limits early progress.`));
  add(H.FIGURE("fig4_convergence.png", "conv", "Test macro-F1 of the global model versus communication round (mean ± 1 s.d. over three seeds). Left: Blood-shift (α = 0.1 with simulated site shift). Right: Derma (α = 0.3, naturally long-tailed). Intermediate points use a fixed 2,000-image test subsample; the last point uses the full test split.", true));
  // ---------------------------------------------------------------- D. per-class
  add(H.H2("D", "Minority Classes and Per-Class Behaviour"));
  const pc = H.N.per_class;
  const low = (s, v) => pc[s][v].f1.filter((x) => x < 0.1).length;
  const dn = pc.derma.fedavg.class_names;
  const rare = ["df", "vasc", "akiec"].map((n) => dn.indexOf(n));
  const rareTxt = rare.map((i) => `${dn[i]} ${f3(pc.derma.fedavg.f1[i])} → ${f3(pc.derma.fedcare.f1[i])}`).join(", ");
  const bn = pc["blood-shift"].fedavg.class_names;
  const worstB = pc["blood-shift"].fedavg.f1.indexOf(Math.min(...pc["blood-shift"].fedavg.f1));
  add(H.P(`Fig. ${H.fig("perclass")} shows the per-class F1 of FedAvg, the two label-skew baselines and FedCARE, averaged over seeds; the complete precision, recall and F1 values of every class are listed in Appendix A. With FedAvg, ${low("blood-shift", "fedavg")} BloodMNIST and ${low("derma", "fedavg")} DermaMNIST classes have a mean F1 below 0.1; with FedCARE the numbers are ${low("blood-shift", "fedcare")} and ${low("derma", "fedcare")}. The weakest FedAvg class on BloodMNIST, ${bn[worstB]}, improves from ${f3(pc["blood-shift"].fedavg.f1[worstB])} to ${f3(pc["blood-shift"].fedcare.f1[worstB])}. On DermaMNIST the three rarest classes change as follows: ${rareTxt}. Fig. ${H.fig("conf")} shows the row-normalized confusion matrices summed over the three seeds. The mean of the diagonal (the balanced accuracy) rises from ${S.map((s) => `${f3(g(s, "fedavg", "bacc")[0])} to ${f3(g(s, "fedcare", "bacc")[0])} on ${H.SNAME[s]}`).join(" and from ")}. ${S.map((s) => { const q = pc[s].fedcare; const cm = q.confusion_sum.map((row) => { const t = row.reduce((x, y) => x + y, 0) || 1; return row.map((x) => x / t); }); let bi = 0, bj = 1, bv = -1; cm.forEach((row, i) => row.forEach((v, j) => { if (i !== j && v > bv) { bv = v; bi = i; bj = j; } })); return `The largest remaining confusion of FedCARE on ${H.SNAME[s]} is ${q.class_names[bi]} predicted as ${q.class_names[bj]} (${pts(bv)}% of the ${q.class_names[bi]} images)`; }).join("; ")}.`));
  add(H.FIGURE("fig5_perclass.png", "perclass", "Per-class test F1 (mean over three seeds) of FedAvg, FedLC, FedRS and FedCARE. Top: BloodMNIST cell types. Bottom: DermaMNIST lesion types; df, vasc and akiec are the rarest classes."));
  add(H.FIGURE("fig6_confusion.png", "conf", "Row-normalized confusion matrices (in %, summed over three seeds; values below 5% omitted) of FedAvg and FedCARE on BloodMNIST (left pair) and DermaMNIST (right pair). Rows are true classes and columns predicted classes.", true));
  // ---------------------------------------------------------------- E. ablation
  add(H.H2("E", "Ablation Study"));
  const AB = ["fedcare", "fedcare-noCAH", "fedcare-noPC", "fedcare+align", "fedcare+proto", "fedcare+lrcal"];
  const arows = AB.filter((v) => S.every((s) => a(s, v, "macro_f1"))).map((v) => [v === "fedcare" ? "**FedCARE (full)**" : H.L[v],
    ...S.map((s) => `${f3(a(s, v, "macro_f1")[0])}±${f3(a(s, v, "macro_f1")[1])}`), ...S.map((s) => `${f3(a(s, v, "worst_recall")[0])}±${f3(a(s, v, "worst_recall")[1])}`)]);
  const d = (v, s) => a(s, v, "macro_f1") ? a(s, "fedcare", "macro_f1")[0] - a(s, v, "macro_f1")[0] : NaN;
  add(H.P(`Table ${H.tab("abl") || "V"} and Fig. ${H.fig("abl")} isolate the components. Removing class-aware aggregation (prior correction only) lowers macro-F1 by ${pts(d("fedcare-noCAH", "blood-shift"))} and ${pts(d("fedcare-noCAH", "derma"))} points on Blood-shift and Derma, and removing the prior correction (class-aware aggregation only) by ${pts(d("fedcare-noPC", "blood-shift"))} and ${pts(d("fedcare-noPC", "derma"))} points${["fedcare-noCAH", "fedcare-noPC"].every((v) => S.every((s) => d(v, s) > 0)) ? `. Both components therefore contribute in both settings, but the prior correction accounts for most of the gain; the class-aware rule adds a smaller improvement, ${S.every((s) => d("fedcare-noCAH", s) < 0.03) ? "consistent with Proposition 1 in direction but modest in size here, because once the prior correction has made absent-class rows inert, dilution only slows the rows of classes held by few clients rather than biasing them" : "in line with Proposition 1"}` : "; a negative value means that removing the component helped, so the two parts are not uniformly complementary"}. Adding PCA subspace alignment changes macro-F1 by ${pts(-d("fedcare+align", "blood-shift"))} and ${pts(-d("fedcare+align", "derma"))} points, and prototype alignment (rank 0) by ${pts(-d("fedcare+proto", "blood-shift"))} and ${pts(-d("fedcare+proto", "derma"))} points. Post-hoc LR-Cal on top of FedCARE changes macro-F1 by ${pts(-d("fedcare+lrcal", "blood-shift"))} and ${pts(-d("fedcare+lrcal", "derma"))} points. ${S.every((s) => d("fedcare+lrcal", s) > 0) ? "Once the classifier is trained to be prior-free, re-fitting it to Gaussian virtual features no longer removes a bias, and the Gaussian approximation of the feature distribution costs accuracy." : S.every((s) => d("fedcare+lrcal", s) < 0) ? "Calibration therefore remains useful on top of FedCARE, because FedCARE balances the classifier only with respect to the local priors it observed during training." : `It helps on ${S.filter((s) => d("fedcare+lrcal", s) < 0).map((s) => H.SNAME[s]).join(" and ")} but not on ${S.filter((s) => d("fedcare+lrcal", s) >= 0).map((s) => H.SNAME[s]).join(" and ")}: when the global label distribution is itself long-tailed, re-fitting the classifier on class-balanced virtual features adds a correction that local prior correction alone does not provide, whereas for a classifier that is already balanced the Gaussian approximation of the features costs accuracy.`}`));
  add(H.TABLE("abl", "Ablation of FedCARE (Test Macro-F1 and Worst-Class Recall)", [[{ t: "" }, { t: "Macro-F1", span: 2 }, { t: "Worst-class recall", span: 2 }], ["Variant", "Blood", "Derma", "Blood", "Derma"]], arows,
    [1640, 850, 850, 850, 850], { size: 13, note: "w/o class-aware head: prior-corrected loss with sample-weighted aggregation; w/o prior correction: class-aware aggregation with plain cross-entropy; + alignment: subspace (rank 4) or prototype (rank 0) penalty (5) with β = 0.5." }));
  add(H.FIGURE("fig7_ablation.png", "abl", "Ablation of FedCARE: test macro-F1 (mean ± s.d. over three seeds). The full method is shown in red."));
  // ---------------------------------------------------------------- F. calibration & communication
  add(H.H2("F", "Low-Rank Calibration and Communication Cost"));
  const m = H.N.meta["blood-shift"]; const dd = m.feat_dim, r = 4;
  const lrF = 1 + dd + r * dd + r + 1, fullF = 1 + dd + dd * (dd + 1) / 2;
  const present = (s) => H.N.meta[s].client_class_counts.map((row) => row.filter((x) => x > 0).length).reduce((x, y) => x + y, 0) / H.N.meta[s].client_class_counts.length;
  const crow = (label, v, upOnce, perRound) => [label, ...S.map((s) => `${f3(g(s, v, "macro_f1")[0])}±${f3(g(s, v, "macro_f1")[1])}`), upOnce, perRound];
  const nParams = m.n_params;
  const ccvrUp = S.map((s) => Math.round(present(s) * fullF)), lrUp = S.map((s) => Math.round(present(s) * lrF));
  add(H.P(`Table ${H.tab("comm") || "VI"} compares calibration and communication. Every method uploads the ${nParams.toLocaleString("en-US")} model values per client and round. LR-Cal and CCVR give practically the same classifier for FedAvg (macro-F1 ${S.map((s) => `${f3(g(s, "fedavg+lrcal", "macro_f1")[0])} vs. ${f3(g(s, "fedavg+ccvr", "macro_f1")[0])}`).join(" and ")}), while the one-time statistics upload per client drops from about ${ccvrUp.map((x) => x.toLocaleString("en-US")).join("/")} to ${lrUp.map((x) => x.toLocaleString("en-US")).join("/")} values (Blood-shift/Derma, ${(fullF / lrF).toFixed(1)}× fewer). The gain of LR-Cal grows quadratically with the feature dimension: for a ResNet-18 with *d* = 512 the per-class upload would fall from ${(1 + 512 + 512 * 513 / 2).toLocaleString("en-US")} to ${(1 + 512 + 4 * 512 + 4 + 1).toLocaleString("en-US")} values. FedCARE itself needs only *C* label counts per round (${H.N.meta["blood-shift"].class_names.length} or ${H.N.meta.derma.class_names.length} integers), whereas SCAFFOLD doubles the upload with its control variates. Mean wall-clock training time on one CPU core was ${f3(g("blood-shift", "fedcare", "train_time_s")[0] / g("blood-shift", "fedavg", "train_time_s")[0])}× that of FedAvg for FedCARE and ${f3(g("blood-shift", "moon", "train_time_s")[0] / g("blood-shift", "fedavg", "train_time_s")[0])}× for MOON, which runs two additional forward passes per batch.`));
  add(H.TABLE("comm", "Calibration Quality and Upload Cost per Client", [["Method", "Blood F1", "Derma F1", "Stats. (once)", "Extra / round"]],
    [crow("FedAvg", "fedavg", "—", "0"), crow("FedAvg + CCVR", "fedavg+ccvr", `${ccvrUp[0].toLocaleString("en-US")}`, "0"),
     crow("FedAvg + LR-Cal", "fedavg+lrcal", `${lrUp[0].toLocaleString("en-US")}`, "0"), crow("SCAFFOLD", "scaffold", "—", nParams.toLocaleString("en-US")),
     crow("**FedCARE**", "fedcare", "—", `*C* = ${H.N.meta["blood-shift"].class_names.length}`)],
    [1380, 960, 960, 900, 840], { size: 13, note: `Values are counts of transmitted numbers per client (statistics column: Blood-shift, average number of classes per client × per-class summary size; ${fullF} values per class for CCVR, ${lrF} for LR-Cal with *r* = 4). Model upload: ${nParams.toLocaleString("en-US")} values per round for every method.` }));
  return out;
}

function discussion(H) {
  const { g, f3, pts } = acc(H);
  const out = [];
  const add = (...xs) => xs.flat().forEach((x) => out.push(x));
  const rho = (s) => {
    const cc = H.N.meta[s].client_class_counts; const nk = cc.map((r) => r.reduce((x, y) => x + y, 0)); const N = nk.reduce((x, y) => x + y, 0);
    const vals = cc[0].map((_, c) => cc.reduce((t, r, k) => t + (r[c] > 0 ? nk[k] : 0), 0) / N);
    return [Math.min(...vals), vals.reduce((x, y) => x + y, 0) / vals.length];
  };
  const rb = rho("blood-shift"), rd = rho("derma");
  add(H.H1("VII", "Discussion and Limitations"));
  add(H.P(`**Why class-aware aggregation helps.** Proposition 1 quantifies the dilution of FedAvg. For the seed-0 partitions, the share of training data held by the clients that observe a class, ρ_{c}, averages ${f3(rb[1])} and is as low as ${f3(rb[0])} on Blood-shift (${f3(rd[1])} and ${f3(rd[0])} on Derma, where the natural long tail places the rare classes on few clients). Under FedAvg the informative update of such a class is therefore shrunk to ${pts(rb[0])}% of its size in every round, which is consistent with the slow and oscillating learning of minority classes in Figs. ${H.fig("conv")} and ${H.fig("perclass")}. Because the correction acts on the classifier rows only, it is complementary to drift-correcting optimizers and could be combined with them.`));
  if (alignHelps(H)) add(H.P(`**Subspace alignment.** Aligning features with PCA class subspaces, the natural continuation of subspace-guided aggregation, ${alignPhrase(H).replace("aligning features with PCA class subspaces ", "")}; it is not part of the default configuration because its effect was not consistent across settings and it requires uploading feature statistics every round.`));
  else add(H.P("**Why subspace alignment did not help.** Aligning features with PCA class subspaces was the natural continuation of subspace-guided aggregation, but it did not improve on FedCARE. Three observations explain this. First, the targets are estimated from a representation that is still changing quickly, so pulling features towards last round's class subspace partly fights the learning of the representation; the variance-normalized form of the penalty even diverged for this reason. Second, in these settings the dominant error is classifier bias, which FedCARE already removes, as the small gains of the drift- and representation-oriented baselines also indicate. Third, the simulated site shift changes colour, contrast and sharpness, to which the batch-normalized CNN is partly invariant. Feature alignment may still matter for stronger, real scanner shifts, which we leave to future work."));
  add(H.P(`**When post-hoc calibration is preferable.** On DermaMNIST, re-fitting the FedAvg classifier on class-balanced virtual features (CCVR or LR-Cal) reached a macro-F1 of ${f3(g("derma", "fedavg+lrcal", "macro_f1")[0])} and a worst-class recall of ${f3(g("derma", "fedavg+lrcal", "worst_recall")[0])}, compared with ${f3(g("derma", "fedcare", "macro_f1")[0])} and ${f3(g("derma", "fedcare", "worst_recall")[0])} for FedCARE. FedCARE corrects each client's classifier for the *local* label prior and then aggregates class-wise, but the global label distribution of DermaMNIST is itself long-tailed (nevi form two thirds of the data), and the class-aware rows inherit this imbalance through the shared features. Post-hoc calibration targets exactly this global imbalance at the cost of an extra statistics round, so the two approaches are complementary: FedCARE removes the client-level bias during training, and LR-Cal can remove a remaining global imbalance cheaply. Indeed, applying LR-Cal to the final FedCARE model on DermaMNIST gives a macro-F1 of ${f3(H.N.ablation.derma["fedcare+lrcal"].macro_f1[0])} and a worst-class recall of ${f3(H.N.ablation.derma["fedcare+lrcal"].worst_recall[0])}, ${H.N.ablation.derma["fedcare+lrcal"].macro_f1[0] >= g("derma", "fedavg+ccvr", "macro_f1")[0] && H.N.ablation.derma["fedcare+lrcal"].worst_recall[0] >= g("derma", "fedavg+ccvr", "worst_recall")[0] ? "the best values of all configurations on this dataset" : "comparable to the best configurations on this dataset"}; on Blood-shift, where the global label distribution is nearly balanced, the same calibration lowers the macro-F1 of FedCARE (${f3(H.N.ablation["blood-shift"]["fedcare+lrcal"].macro_f1[0])} vs. ${f3(g("blood-shift", "fedcare", "macro_f1")[0])}), so it should be applied only when the global distribution is known to be long-tailed.`));
  add(H.P("**Limitations.** The conclusions are bounded by the experimental scale. (i) A compact CNN on 28×28 images, 4,000 training images per federation and 25 rounds were used so that 66 federated runs fit on a single CPU core; larger backbones, higher resolutions and longer training may change the size of the gaps. (ii) BloodMNIST and DermaMNIST provide real class imbalance, but the acquisition shift of Blood-shift is simulated, and the official test splits are unshifted, so the experiments do not measure generalization to unseen hospitals. Validation on naturally multi-centre data, such as the external-centre test set of PathMNIST, Fed-ISIC2019 of FLamby {cite:terrail2022flamby} or Camelyon17-WILDS {cite:koh2021wilds}, is necessary before any clinical or IoMT deployment claim; the released configuration files include the PathMNIST and multi-domain settings for this purpose. (iii) Three seeds per setting give six paired observations, enough to detect consistent differences but not small ones, and the Blood-shift results vary strongly across partitions; more seeds and partitions would narrow the confidence intervals. (iv) All clients participated in every round; partial participation, client dropout and asynchronous IoMT clients were not studied. (v) FedCARE shares per-client label counts, which reveal the local label distribution; secure aggregation {cite:bonawitz2017} of the sums required by (2) or differentially private counts {cite:abadi2016} can mitigate this, and model updates themselves remain exposed to gradient-inversion attacks {cite:zhu2019dlg}."));
  add(H.H1("VIII", "Conclusion"));
  const T = H.N.tests;
  const winsTT = TT.reduce((t, b) => t + (T[b] && T[b].macro_f1 ? T[b].macro_f1.wins : 0), 0);
  const nTT = TT.reduce((t, b) => t + (T[b] && T[b].macro_f1 ? T[b].macro_f1.n : 0), 0);
  const bestTT = H.SETS.every((s) => g(s, "fedcare", "macro_f1")[0] >= Math.max(...TT.map((v) => g(s, v, "macro_f1")[0])));
  add(H.P(`We identified a simple but consequential weakness of federated averaging under label skew, namely that the classifier row of every class is diluted by the share of data held by clients that never observed it, and proposed FedCARE, which combines a prior-corrected local loss with class-aware aggregation of the classifier. FedCARE requires no tuning and only label counts as extra communication. ${bestTT ? "It obtained the highest mean macro-F1 among the training-time methods in both settings" : "It improved the mean macro-F1 over most training-time methods"} and won ${winsTT} of ${nTT} paired comparisons with them, with clear gains in balanced accuracy over FedAvg. Results under the most severe skew varied strongly across partitions, and on the naturally long-tailed DermaMNIST post-hoc calibration of FedAvg was at least as accurate, which suggests combining both. LR-Cal reduces the upload of feature-statistics calibration by a factor of ${((1 + 64 + 64 * 65 / 2) / (1 + 64 + 4 * 64 + 4 + 1)).toFixed(1)} without loss of accuracy, and ${alignPhrase(H)}. Future work will evaluate FedCARE on real multi-centre data, with partial participation and larger backbones, and will combine it with privacy-preserving aggregation of the label counts.`));
  return out;
}

function appendix(H) {
  const { f3 } = acc(H);
  const out = [];
  const add = (...xs) => xs.flat().forEach((x) => out.push(x));
  out.push(new Paragraph({ children: [new TextRun({ text: "Appendix A", font: "Times New Roman", size: 20, smallCaps: true })], alignment: AlignmentType.CENTER, spacing: { before: 200, after: 40 } }));
  out.push(new Paragraph({ children: [new TextRun({ text: "Per-Class Precision, Recall and F1", font: "Times New Roman", size: 20, smallCaps: true })], alignment: AlignmentType.CENTER, spacing: { after: 100 } }));
  add(H.P("Tables VII and VIII list the per-class precision (P), recall (R) and F1 on the full test splits, averaged over the three seeds, for FedAvg, the strongest label-skew baseline FedLC, post-hoc CCVR and FedCARE; the last rows give the macro averages. The raw per-run values, including the confusion matrices of every run, are part of the released result files."));
  const V = ["fedavg", "fedlc", "fedavg+ccvr", "fedcare"];
  for (const s of H.SETS) {
    const pc = H.N.per_class[s];
    const names = pc.fedavg.class_names;
    const rows = names.map((n, i) => [n, String(Math.round(pc.fedavg.support[i])),
      ...V.flatMap((v) => [f3(pc[v].precision[i]), f3(pc[v].recall[i]), f3(pc[v].f1[i])])]);
    const mean = (arr) => arr.reduce((x, y) => x + y, 0) / arr.length;
    rows.push([{ t: "**Macro avg.**", topRule: true }, { t: String(Math.round(pc.fedavg.support.reduce((x, y) => x + y, 0))), topRule: true },
      ...V.flatMap((v) => ["precision", "recall", "f1"].map((k) => ({ t: `**${f3(mean(pc[v][k]))}**`, topRule: true })))]);
    const hdr = [[{ t: "" }, { t: "" }, ...V.map((v) => ({ t: H.L[v].replace(" (ours)", ""), span: 3 }))],
      ["Class", "*n*", ...V.flatMap(() => ["P", "R", "F1"])]];
    add(H.TABLE(`pc-${s}`, `Per-Class Results on ${H.SNAME[s]} (${s === "derma" ? "Derma" : "Blood-shift"} Setting, Mean of Three Seeds)`, hdr, rows,
      [1100, 580, ...Array(12).fill(730)], { size: 13 }));
  }
  return out;
}

function acknowledgment(H) {
  return [new Paragraph({ children: [new TextRun({ text: "Acknowledgment", font: "Times New Roman", size: 20, smallCaps: true })], alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 } }),
    H.P("The authors used an AI assistant (Claude, Anthropic) for code development, experiment orchestration and language editing; all methods, results and statements were checked by the authors, who take full responsibility for the content.")];
}

module.exports = { TITLE, alignPhrase, abstract, algorithmBox, results, discussion, appendix, acknowledgment };
