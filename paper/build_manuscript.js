// Builds the IEEE-style manuscript (.docx) from results/summary/paper_numbers.json
// and results/figures/*.png.   Usage: node paper/build_manuscript.js out.docx
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType,
  WidthType, BorderStyle, SectionType, TabStopType, VerticalAlign, Footer, PageNumber, FootnoteReferenceRun,
} = require("docx");

const ROOT = path.join(__dirname, "..");
const N = JSON.parse(fs.readFileSync(process.env.NUMBERS || path.join(ROOT, "results/summary/paper_numbers.json")));
const FIGNUM = { framework: 1, partition: 2, sites: 3, conv: 4, perclass: 5, conf: 6, abl: 7 };
const TABNUM = { positioning: 1, config: 2, main: 3, tests: 4, abl: 5, comm: 6, "pc-blood-shift": 7, "pc-derma": 8 };
const FIG = (f) => path.join(ROOT, "results/figures", f);
const FONT = "Times New Roman";
const COLW = 5040, FULLW = 10440;            // 3.5 in column, 7.25 in text width (twips)

// ------------------------------------------------------------------ citations
const REFS = require("./references.js");
const order = [];
function cite(...keys) {
  const nums = keys.map((k) => {
    if (!REFS[k]) throw new Error("unknown ref " + k);
    if (!order.includes(k)) order.push(k);
    return order.indexOf(k) + 1;
  }).sort((a, b) => a - b);
  // compress ranges: [3]-[5]
  const parts = []; let i = 0;
  while (i < nums.length) {
    let j = i; while (j + 1 < nums.length && nums[j + 1] === nums[j] + 1) j++;
    parts.push(j - i >= 2 ? `[${nums[i]}]–[${nums[j]}]` : nums.slice(i, j + 1).map((n) => `[${n}]`).join(", "));
    i = j + 1;
  }
  return parts.join(", ");
}

// ------------------------------------------------------------------ rich text
// markup: _{sub} ^{sup} *italic* **bold** {cite:a,b}
function runs(text, base = {}) {
  const out = [];
  text = text.replace(/\{cite:([^}]+)\}/g, (_, k) => cite(...k.split(",").map((s) => s.trim())));
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|_\{[^}]*\}|\^\{[^}]*\})/g;
  let last = 0, m;
  const push = (t, extra = {}) => { if (t) out.push(new TextRun({ text: t, font: FONT, ...base, ...extra })); };
  while ((m = re.exec(text)) !== null) {
    push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("**")) push(tok.slice(2, -2), { bold: true });
    else if (tok.startsWith("*")) push(tok.slice(1, -1), { italics: true });
    else if (tok.startsWith("_{")) push(tok.slice(2, -1), { subScript: true });
    else push(tok.slice(2, -1), { superScript: true });
    last = m.index + tok.length;
  }
  push(text.slice(last));
  return out;
}
const P = (text, o = {}) => new Paragraph({
  children: runs(text, { size: o.size || 20 }), alignment: o.align || AlignmentType.JUSTIFIED,
  indent: o.noIndent ? undefined : { firstLine: 202 }, spacing: { after: o.after || 0, before: o.before || 0, line: 240 },
  keepNext: o.keepNext,
});
const H1 = (num, text) => new Paragraph({
  children: [new TextRun({ text: `${num}. `, font: FONT, size: 20 }), new TextRun({ text, font: FONT, size: 20, smallCaps: true })],
  alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 }, keepNext: true,
});
const H2 = (letter, text) => new Paragraph({
  children: [new TextRun({ text: `${letter}. ${text}`, font: FONT, size: 20, italics: true })],
  spacing: { before: 120, after: 60 }, keepNext: true,
});
let eqNo = 0;
function EQ(text, width = COLW) {
  eqNo += 1;
  return new Paragraph({
    children: [new TextRun({ text: "\t", font: FONT }), ...runs(text, { size: 20 }), new TextRun({ text: `\t(${eqNo})`, font: FONT, size: 20 })],
    tabStops: [{ type: TabStopType.CENTER, position: Math.round(width / 2) }, { type: TabStopType.RIGHT, position: width }],
    spacing: { before: 60, after: 60 },
  });
}
const BULLET = (label, text) => new Paragraph({
  children: [...runs(`${label}\t`, { size: 20 }), ...runs(text, { size: 20 })], alignment: AlignmentType.JUSTIFIED,
  indent: { left: 360, hanging: 360 }, tabStops: [{ type: TabStopType.LEFT, position: 360 }], spacing: { after: 30 },
});

// ------------------------------------------------------------------ figures
let figNo = 0; const figLabel = {};
function pngSize(f) { const b = fs.readFileSync(f); return [b.readUInt32BE(16), b.readUInt32BE(20)]; }
function FIGURE(file, key, caption, wide = false) {
  figNo += 1; figLabel[key] = figNo;
  const [w, h] = pngSize(FIG(file));
  const widthPx = wide ? 680 : 330;
  const items = [
    new Paragraph({ children: [new ImageRun({ type: "png", data: fs.readFileSync(FIG(file)), transformation: { width: widthPx, height: Math.round(widthPx * h / w) } })],
      alignment: AlignmentType.CENTER, spacing: { before: 100, after: 40 }, keepNext: true }),
    new Paragraph({ children: [new TextRun({ text: `Fig. ${figNo}. `, font: FONT, size: 16 }), ...runs(caption, { size: 16 })],
      alignment: AlignmentType.JUSTIFIED, spacing: { after: 140 } }),
  ];
  return wide ? { __wide: true, items } : items;
}

// ------------------------------------------------------------------ tables
const ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"];
let tabNo = 0; const tabLabel = {};
const thin = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const thick = { style: BorderStyle.SINGLE, size: 8, color: "000000" };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function TABLE(key, title, header, rows, widths, opts = {}) {
  tabNo += 1; tabLabel[key] = tabNo;
  const total = widths.reduce((a, b) => a + b, 0);
  const fs8 = opts.size || 15;
  const mk = (cells, i, isHeader) => { let col = 0; return new TableRow({
    tableHeader: isHeader, cantSplit: true,
    children: cells.map((c) => {
      const cell = typeof c === "object" && c !== null && !Array.isArray(c) ? c : { t: String(c) };
      const span = cell.span || 1, j = col;
      const wcell = widths.slice(col, col + span).reduce((x, y) => x + y, 0); col += span;
      return new TableCell({
        width: { size: wcell, type: WidthType.DXA }, columnSpan: span, verticalAlign: VerticalAlign.CENTER,
        margins: { top: 20, bottom: 20, left: 50, right: 50 },
        borders: { left: none, right: none,
          top: (isHeader && i === 0) ? thick : (cell.topRule ? thin : none),
          bottom: isHeader && i === header.length - 1 ? thin : (!isHeader && i === rows.length - 1 ? thick : (cell.rule ? thin : none)) },
        children: [new Paragraph({ children: runs(cell.t, { size: fs8, bold: cell.b || isHeader && opts.boldHeader }),
          keepNext: isHeader || i < rows.length - 1,
          alignment: j === 0 && !opts.centerFirst ? AlignmentType.LEFT : AlignmentType.CENTER })],
      });
    }),
  }); };
  const table = new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, alignment: AlignmentType.CENTER,
    rows: [...header.map((r, i) => mk(r, i, true)), ...rows.map((r, i) => mk(r, i, false))] });
  const items = [
    new Paragraph({ children: [new TextRun({ text: `TABLE ${ROMAN[tabNo]}`, font: FONT, size: 16 })], alignment: AlignmentType.CENTER, spacing: { before: 120 }, keepNext: true }),
    new Paragraph({ children: runs(title, { size: 16, smallCaps: true }), alignment: AlignmentType.CENTER, spacing: { after: 60 }, keepNext: true }),
    table,
    new Paragraph({ children: opts.note ? runs(opts.note, { size: 14 }) : [], alignment: AlignmentType.JUSTIFIED, spacing: { before: 30, after: 120 } }),
  ];
  return opts.wide ? { __wide: true, items } : items;
}

// ------------------------------------------------------------------ numbers
const L = N.labels;
const fmt = (ms, d = 3) => `${ms[0].toFixed(d)} ± ${ms[1].toFixed(d)}`;
const pct = (x, d = 1) => `${(100 * x).toFixed(d)}`;
const SETS = ["blood-shift", "derma"];
const SNAME = { "blood-shift": "BloodMNIST", derma: "DermaMNIST" };
const meta = N.meta;
const H = { P, H1, H2, EQ, BULLET, FIGURE, TABLE, runs, cite, fmt, pct, N, L, SETS, SNAME, COLW, FULLW,
  fig: (k) => FIGNUM[k], tab: (k) => ROMAN[TABNUM[k]], AlignmentType, Paragraph, TextRun };
const R = require("./results_text.js");

// ------------------------------------------------------------------ front matter
const title = new Paragraph({ children: [new TextRun({ text: R.TITLE, font: FONT, size: 44 })],
  alignment: AlignmentType.CENTER, spacing: { before: 0, after: 160 } });
// Authors and affiliations (first-page footnote, IEEE style)
const AUTHORS = [
  "Alireza Fathi — Dept. Computer Engineering, NT.C., Islamic Azad University, Tehran, Iran; alirezafathi84@gmail.com",
  "Reza Boostani — CSE&IT Dept., ECE School, Shiraz University, Shiraz, Iran; rboostani87@gmail.com",
  "Rahil Hosseini — Dept. Computer Engineering, Islamic Azad University, ShQ.C., Tehran, Iran; rahilhosseini@gmail.com",
  "Mazin Abed Mohammed — Dept. Computer Science, University of Anbar, Ramadi, Iraq; mazinalshujeary@uoanbar.edu.iq",
];
const author = [
  new Paragraph({ children: [new TextRun({ text: "Alireza Fathi, Reza Boostani, Rahil Hosseini, and Mazin Abed Mohammed", font: FONT, size: 22 }),
    new FootnoteReferenceRun(1)], alignment: AlignmentType.CENTER, spacing: { after: 240 } }),
];
const footnotes = { 1: { children: AUTHORS.map((t) => new Paragraph({ children: [new TextRun({ text: " " + t, font: FONT, size: 16 })], spacing: { after: 10 } })) } };
const abstract = new Paragraph({ children: [new TextRun({ text: "Abstract", font: FONT, size: 18, bold: true, italics: true }),
  new TextRun({ text: "—", font: FONT, size: 18, bold: true }), ...runs(R.abstract(H), { size: 18, bold: true })],
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 100 } });
const keywords = new Paragraph({ children: [new TextRun({ text: "Index Terms", font: FONT, size: 18, bold: true, italics: true }),
  new TextRun({ text: "—Federated learning, label skew, class imbalance, classifier calibration, acquisition shift, principal component analysis, medical image classification, Internet of Medical Things.", font: FONT, size: 18, bold: true })],
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 160 } });

// ------------------------------------------------------------------ body
const body = [];
const add = (...xs) => xs.flat().forEach((x) => body.push(x));
const bs = meta["blood-shift"], dm = meta.derma;
const d = bs.feat_dim, r = 4, Cb = bs.class_names.length;
const lrFloats = 1 + d + r * d + r + 1, fullFloats = 1 + d + d * (d + 1) / 2;

add(H1("I", "Introduction"));
add(P("Clinical images are generated at many hospitals, laboratories and, increasingly, at point-of-care devices of the Internet of Medical Things (IoMT). Federated learning (FL) trains a shared model on these distributed collections without moving patient data: every round, clients improve the current global model on their own data and a server aggregates the resulting parameters {cite:mcmahan2017,kairouz2021}. FL is therefore regarded as a practical route to multi-institutional medical AI {cite:rieke2020,sheller2020,kaissis2020,guan2024} and to learning on resource-constrained IoT nodes {cite:nguyen2021iot}."));
add(P("Two kinds of heterogeneity make medical FL difficult. First, *label skew*: disease prevalence and referral patterns differ across sites, so each client sees a different, often extremely unbalanced, label distribution, and globally the classes are long-tailed. Second, *acquisition shift*: staining protocols, scanners, illumination and optics change the appearance of the same class from site to site. Under label skew, local training pushes the classifier towards the locally frequent classes and suppresses the classes a client never observes; after averaging, the global classifier is biased and minority classes are recognized poorly or not at all {cite:li2021fedrs,zhang2022fedlc,luo2021ccvr}. In our earlier experiments with subspace-weighted aggregation, several tissue classes kept F1 scores close to zero for exactly this reason, which motivated the present, classifier-centric design."));
add(P("Existing remedies act at different points of the pipeline. Drift-correcting optimizers such as FedProx {cite:li2020fedprox}, SCAFFOLD {cite:karimireddy2020}, FedNova {cite:wang2020fednova} and MOON {cite:li2021moon} constrain or correct the local updates of the whole network but do not target the classifier bias that label skew induces. Label-skew methods modify the local softmax, either by restricting the logits of locally absent classes (FedRS {cite:li2021fedrs}) or by calibrating logits with local class counts (FedLC {cite:zhang2022fedlc}); they still aggregate the classifier with sample-size weights. Post-hoc calibration (CCVR {cite:luo2021ccvr}) re-trains the classifier on virtual features drawn from class-conditional Gaussians, but each client must upload a full *d*×*d* covariance matrix per class. Prototype- and subspace-based methods {cite:tan2022fedproto,huang2023fpl} align feature distributions across clients but rely on statistics that drift while the representation is still being learned."));
add(P("**Research gap.** The classifier is the component that label skew damages most directly, yet the aggregation rule for its rows has received little attention: with sample-size weights, the update of the row of class *c* is averaged together with clients that never observed *c*. We show that this *dilutes* the informative update of every class by the fraction of data held by the clients that observed it, a factor that is small exactly for the clinically important minority classes. Correcting the local objective and the aggregation rule jointly, without extra models, control variates or covariance uploads, has not been studied systematically, and published comparisons in this area are frequently based on a single partition and seed."));
add(P("This paper makes the following contributions:"));
add(BULLET("1)", "We propose **FedCARE** (Federated Class-Aware REcalibration), which combines a prior-corrected (balanced-softmax) local objective with *class-aware aggregation* of the classifier rows. We show analytically that the two parts are complementary: prior correction makes the rows of locally absent classes inert during local training, and class-aware weights then remove the dilution of the informative rows (Proposition 1). FedCARE adds only *C* label counts per client to the upload and has no tuned hyper-parameter."));
add(BULLET("2)", `We introduce **LR-Cal**, a low-rank variant of classifier calibration in which clients upload rank-*r* principal-component summaries of their class-conditional features instead of full covariances; the server merges them exactly with the law of total covariance. With *d* = ${d} and *r* = ${r}, LR-Cal reduces the per-class upload from ${fullFloats} to ${lrFloats} values (${(fullFloats / lrFloats).toFixed(1)}× smaller) while matching CCVR.`));
add(BULLET("3)", "We evaluate against eight baselines (FedAvg, FedProx, SCAFFOLD, FedNova, MOON, FedRS, FedLC and FedAvg+CCVR) on two public medical benchmarks with naturally long-tailed classes and with simulated per-site acquisition shift, using three seeds that re-draw the data subsample, the client partition, the site parameters and the initialization; we report means and standard deviations, paired Wilcoxon signed-rank tests, macro-F1, balanced accuracy, macro-AUC, worst-class recall, full per-class precision/recall/F1 and confusion matrices."));
add(BULLET("4)", `We report the outcome of the subspace-guided direction transparently: ${R.alignPhrase(H)} on top of FedCARE, and we analyse why. All code, configurations, raw results and unit tests are released.`));
add(P(`The remainder of the paper is organized as follows. Section II reviews related work, Section III formalizes the problem, Section IV presents FedCARE and LR-Cal, Section V describes the experimental protocol, Section VI reports and analyses the results, Section VII discusses limitations and Section VIII concludes.`, { before: 40 }));

add(H1("II", "Related Work"));
add(H2("A", "Federated Optimization under Heterogeneity"));
add(P("FedAvg {cite:mcmahan2017} averages locally trained models with sample-size weights. When client distributions differ, local optima diverge (client drift) and convergence slows {cite:li2020fedprox,karimireddy2020,hsieh2020}. FedProx adds a proximal term to the local objective {cite:li2020fedprox}; SCAFFOLD corrects local gradients with control variates {cite:karimireddy2020}; FedNova normalizes local updates to remove the objective inconsistency caused by unequal numbers of local steps {cite:wang2020fednova}; MOON regularizes local representations towards the global model with a contrastive loss {cite:li2021moon}. Large benchmark studies found that none of these methods dominates under label skew {cite:li2022niid}. These methods treat all parameters alike, whereas FedCARE changes only the classifier's loss and aggregation."));
add(H2("B", "Label Skew and Classifier Bias"));
add(P("Long-tail learning showed that a softmax classifier trained on imbalanced labels absorbs the label prior and that adding the log-prior to the logits during training (logit adjustment, balanced softmax) yields a classifier that is consistent for the balanced error {cite:menon2021,ren2020}. In FL, FedRS scales down the logits of classes a client does not hold {cite:li2021fedrs}, FedLC subtracts a count-dependent calibration term {cite:zhang2022fedlc}, FedBABU freezes the classifier during federated training {cite:oh2022fedbabu}, and CCVR {cite:luo2021ccvr} and CReFF {cite:shang2022creff} re-train the classifier on the server with virtual or federated features. FedCARE uses the Bayes-consistent log-prior correction locally, but its distinctive element is the aggregation rule for the classifier rows, which these methods leave unchanged; LR-Cal reduces the communication of CCVR-style calibration."));
add(H2("C", "Feature Shift, Prototypes and PCA in FL"));
add(P("FedBN keeps batch-normalization layers local to absorb feature shift {cite:li2021fedbn}, at the price of client-specific models; FedDG exchanges amplitude spectra for domain generalization {cite:liu2021feddg}. Prototype methods share class means and pull local features towards them {cite:tan2022fedproto,huang2023fpl}, which is related to center-loss regularization {cite:wen2016}. Principal component analysis has been federated directly {cite:grammenos2020}, and probabilistic PCA {cite:tipping1999} provides the low-rank-plus-isotropic covariance model that LR-Cal transmits. Our ablation tests whether aligning features with PCA class subspaces adds value on top of classifier recalibration."));
add(H2("D", "Federated Learning for Medical Imaging and IoMT"));
add(P("Medical FL has been demonstrated across institutions {cite:sheller2020,rieke2020} and surveyed for medical image analysis {cite:guan2024} and for IoT systems {cite:nguyen2021iot}. Realistic cross-silo benchmarks such as FLamby {cite:terrail2022flamby} and distribution-shift benchmarks such as WILDS {cite:koh2021wilds} highlight that conclusions drawn from a single synthetic split are fragile, which motivates the multi-seed protocol and the tempered claims of this study. Table I summarizes how the compared methods differ."));
add(TABLE("positioning", "Where Each Method Acts and What It Adds to the Upload",
  [["Method", "Local objective", "Aggregation", "Post-hoc", "Extra upload / client"]],
  [["FedAvg {cite:mcmahan2017}", "CE", "sample-weighted", "—", "none"],
   ["FedProx {cite:li2020fedprox}", "CE + proximal", "sample-weighted", "—", "none"],
   ["SCAFFOLD {cite:karimireddy2020}", "corrected gradients", "sample-weighted", "—", "control variate (= model size)"],
   ["FedNova {cite:wang2020fednova}", "CE", "normalized updates", "—", "number of local steps"],
   ["MOON {cite:li2021moon}", "CE + contrastive", "sample-weighted", "—", "none"],
   ["FedRS {cite:li2021fedrs}", "restricted softmax", "sample-weighted", "—", "none"],
   ["FedLC {cite:zhang2022fedlc}", "calibrated logits", "sample-weighted", "—", "none"],
   ["CCVR {cite:luo2021ccvr}", "CE", "sample-weighted", "classifier re-fit", "*C*(*d*+*d*(*d*+1)/2+1), once"],
   ["**FedCARE (ours)**", "log-prior corrected", "**class-aware rows**", "optional LR-Cal", "*C* label counts"],
   ["**LR-Cal (ours)**", "—", "—", "classifier re-fit", "*C*(*rd*+*d*+*r*+2), once"]],
  [1500, 1060, 1000, 720, 760], { size: 13 }));

add(H1("III", "Problem Formulation"));
add(P("Consider *K* clients. Client *k* holds *n*_{k} labelled images {(*x*_{i}, *y*_{i})} with *y* ∈ {1, …, *C*}; *n*_{k,c} denotes its number of images of class *c*, *N*_{c} = Σ_{k} *n*_{k,c} and *N* = Σ_{k} *n*_{k}. The local label distribution is π_{k,c} = *n*_{k,c}/*n*_{k}. *Label skew* means π_{k} differs across clients and from the global distribution; *acquisition shift* means that the class-conditional distribution *p*_{k}(*x*|*y*) differs across clients. The model is *f* = *g* ∘ φ, a feature extractor φ: *x* ↦ *h* ∈ ℝ^{d} followed by a linear classifier *g*(*h*) = *Wh* + *b* with rows *w*_{c}. Because clinically rare classes matter as much as frequent ones, we evaluate the global model with class-balanced metrics: balanced accuracy (mean per-class recall), macro-F1, macro-AUC (one-vs-rest) and the worst per-class recall."));

add(H1("IV", "Proposed Method"));
add(H2("A", "Overview"));
add(P(`Fig. 1 summarizes FedCARE. In each round the server broadcasts the global model θ^{t} = (φ^{t}, *W*^{t}, *b*^{t}). Each client trains with a prior-corrected objective (Section IV-B) and uploads its parameters together with its *C* label counts. The server averages the feature extractor with sample-size weights and the classifier rows with class-count weights (Section IV-C). LR-Cal (Section IV-E) is an optional post-hoc step, and subspace alignment (Section IV-F) is the ablation studied in Section VI.`));
add(FIGURE("fig1_framework.png", "framework", "Overview of FedCARE. (1) Clients train with the prior-corrected loss, which keeps the rows of locally absent classes inert. (2) The server averages the feature extractor with sample-size weights and every classifier row only over the clients that observed the corresponding class, weighted by their class counts. (3) Optional low-rank (PCA) classifier calibration (LR-Cal). The dotted box is the subspace-alignment variant evaluated as an ablation.", true));
add(H2("B", "Prior-Corrected Local Objective"));
add(P("Client *k* minimizes the balanced-softmax cross-entropy {cite:ren2020,menon2021}"));
add(EQ("ℓ_{k}(*x*, *y*) = −log [ exp(*z*_{y} + τ log π̃_{k,y}) / Σ_{c} exp(*z*_{c} + τ log π̃_{k,c}) ],"));
add(P("where *z* = *Wh* + *b* are the logits, τ = 1 and π̃_{k,c} = (*n*_{k,c} + 1)/(*n*_{k} + *C*) is the add-one-smoothed local prior. If the logits model log *p*(*x*|*y*) up to a constant, the corrected softmax equals the local posterior *p*_{k}(*y*|*x*), so the minimizer learns prior-free logits that transfer to any label distribution {cite:menon2021}. Two consequences matter for FL. For a class absent from client *k*, π̃_{k,c} ≈ 1/*n*_{k} lowers its logit by ≈ log *n*_{k}, its softmax probability is ≈ 0 and the gradient with respect to *w*_{c}, *p*_{c}(*x*)·*h*, nearly vanishes: the client neither learns nor suppresses that class. For classes that are present but rare, the correction removes the local majority bias. With τ = 1 the correction is Bayes-consistent, so FedCARE has no hyper-parameter to tune."));
add(H2("C", "Class-Aware Aggregation of the Classifier"));
add(P("The feature extractor is averaged as in FedAvg, φ^{t+1} = Σ_{k} (*n*_{k}/*N*) φ_{k}^{t+1}. Each classifier row is averaged only over the clients that observed the class:"));
add(EQ("*w*_{c}^{t+1} = Σ_{k} (*n*_{k,c}/*N*_{c}) *w*_{k,c}^{t+1},    *b*_{c}^{t+1} = Σ_{k} (*n*_{k,c}/*N*_{c}) *b*_{k,c}^{t+1}."));
add(P("If no participating client holds class *c* in a round, the row falls back to sample-size weights. **Proposition 1 (row dilution).** Let Δ_{k,c} = *w*_{k,c}^{t+1} − *w*_{c}^{t} and let 𝒮_{c} be the clients with *n*_{k,c} > 0. If Δ_{k,c} = 0 for *k* ∉ 𝒮_{c} (the inert rows produced by the prior correction, up to weight decay), the FedAvg update of row *c* is Δ_{c}^{avg} = ρ_{c} Δ̄_{c}, where ρ_{c} = Σ_{k∈𝒮_{c}} *n*_{k}/*N* ≤ 1 and Δ̄_{c} is a convex combination of the informative updates Δ_{k,c}, *k* ∈ 𝒮_{c}. The class-aware update (2) is Δ_{c}^{CA} = Σ_{k∈𝒮_{c}} (*n*_{k,c}/*N*_{c}) Δ_{k,c}, i.e., it is not scaled by ρ_{c} and weights every client by its amount of class-*c* evidence. *Proof:* substitute Δ_{k,c} = 0 for *k* ∉ 𝒮_{c} into Σ_{k} (*n*_{k}/*N*) Δ_{k,c} and factor out ρ_{c}. ∎"));
add(P(`FedAvg therefore applies a smaller effective server step to the row of every class held by few clients. Without the prior correction the rows of absent classes are not inert but are actively pushed away from the local features, which is the classical source of classifier bias {cite:li2021fedrs}; with the correction alone, the informative updates are still diluted by ρ_{c}. The two components therefore address different failure modes, which the ablation in Section VI confirms. Class-aware aggregation needs only Σ_{k} *n*_{k,c} *w*_{k,c} and Σ_{k} *n*_{k,c}, both sums over clients, so it is compatible with secure aggregation {cite:bonawitz2017}.`));
add(H2("D", "Algorithm and Cost"));
add(P("Algorithm 1 lists one FedCARE round. Compared with FedAvg, local computation is unchanged (one vector addition to the logits), the upload grows by *C* integers, and the server performs a weighted average per classifier row, which is negligible. No extra model copies (MOON), control variates (SCAFFOLD) or covariance matrices (CCVR) are needed, which suits clients with limited memory and bandwidth such as IoMT gateways."));
add(...R.algorithmBox(H));
add(H2("E", "LR-Cal: Classifier Calibration from Low-Rank Feature Statistics"));
add(P(`CCVR {cite:luo2021ccvr} re-fits the classifier on virtual features sampled from class-conditional Gaussians whose means and covariances are aggregated from clients. LR-Cal keeps this procedure but replaces the covariance upload by a probabilistic-PCA summary {cite:tipping1999}. After training, client *k* computes the features of at most 64 images per class with the final global model and, for each class it holds, uploads the count *n*_{k,c}, the mean μ_{k,c} and the top-*r* eigenpairs (*U*_{k,c}, Λ_{k,c}) of the class covariance *S*_{k,c} together with the mean residual eigenvalue σ^{2}_{k,c}. The server reconstructs`));
add(EQ("Ŝ_{k,c} = *U*_{k,c} Λ_{k,c} *U*_{k,c}^{⊤} + σ^{2}_{k,c} (*I* − *U*_{k,c} *U*_{k,c}^{⊤}),"));
add(P("and merges the summaries with the law of total covariance,"));
add(EQ("Σ_{c} = Σ_{k} (*n*_{k,c}/*N*_{c}) [ Ŝ_{k,c} + (μ_{k,c} − μ_{c})(μ_{k,c} − μ_{c})^{⊤} ],"));
add(P(`where μ_{c} = Σ_{k} (*n*_{k,c}/*N*_{c}) μ_{k,c}. The merge is exact when *r* = *d* (verified by the unit tests of the released code); for *r* < *d* it preserves the trace and the leading spectrum of every client covariance. The server then samples 128 virtual features per class from 𝒩(μ_{c}, Σ_{c}) and fine-tunes the classifier for 100 SGD steps with a class-balanced cross-entropy. The per-class upload is *rd* + *d* + *r* + 2 values (${lrFloats} for *d* = ${d}, *r* = ${r}) instead of *d*(*d*+1)/2 + *d* + 1 (${fullFloats}).`));
add(H2("F", "Ablation: Alignment with PCA Class Subspaces"));
add(P("Subspace-based aggregation was the starting point of this work, so we also test whether aligning features across sites helps on top of FedCARE. Clients summarize ℓ_{2}-normalized features ĥ = *h*/‖*h*‖ with the received global model; the server merges them and keeps, per class, the mean μ_{c} and the top-*r* eigenvectors *U*_{c} of the pooled *within-client* scatter, which deliberately excludes the between-site spread of the class means. Clients then add β·*R* to (1), with"));
add(EQ("*R*(*h*, *y*) = ‖(*I* − *U*_{y}*U*_{y}^{⊤})(ĥ − μ_{y})‖^{2} + γ‖*U*_{y}*U*_{y}^{⊤}(ĥ − μ_{y})‖^{2},"));
add(P("which penalizes site-specific deviations outside the shared class subspace fully and deviations along it by γ = 0.1 (β = 0.5, *r* = 4; *r* = 0 reduces to prototype alignment). A first version that scaled the penalty by the estimated class variances (a probabilistic-PCA Mahalanobis distance) diverged, because the variances of early-round features are tiny and change quickly; the bounded form (5), which never exceeds 4 for γ ≤ 1, trains stably."));

add(H1("V", "Experimental Setup"));
add(H2("A", "Datasets"));
add(P(`We use the official 28×28 releases and splits of two MedMNIST v2 datasets {cite:yang2023medmnist}; every file is verified against its published MD5 checksum before use. **BloodMNIST** contains 17,092 microscopy images of eight peripheral-blood cell types acquired at the Hospital Clinic of Barcelona {cite:acevedo2020}. **DermaMNIST** contains 10,015 dermatoscopic images of seven pigmented-lesion classes from the multi-source HAM10000 collection {cite:tschandl2018,codella2019}; it is naturally long-tailed, with melanocytic nevi forming ${pct(dm.train_class_counts[5] / dm.train_class_counts.reduce((a, b) => a + b, 0), 0)}% of the training images and dermatofibroma ${pct(dm.train_class_counts[3] / dm.train_class_counts.reduce((a, b) => a + b, 0))}%.`));
add(H2("B", "Federated Settings"));
add(P(`Each setting has *K* = 10 clients with full participation. From the official training split we draw a class-stratified subsample of 4,000 images and partition it with the Dirichlet label-skew protocol {cite:hsu2019,li2022niid}: for every class, client proportions are drawn from Dir(α), and partitions are re-drawn until every client holds at least ten images. **Blood-shift** uses α = ${bs.setting.alpha} (severe skew) and adds a static, per-site acquisition shift: each client's images are transformed once by a site-specific stain-intensity jitter in optical-density space (per-channel scale in [0.65, 1.35] and offset in [−0.4, 0.4]; the same model as the HED jitter of Tellez *et al.* {cite:tellez2019} without restricting it to H&E stain vectors {cite:ruifrok2001}), brightness and contrast changes of up to ±10% and ±15%, Gaussian blur (σ ≤ 0.9 px) and sensor noise (σ ≤ 0.03). **Derma** uses α = ${dm.setting.alpha} on top of the natural long tail. Fig. 2 shows the resulting client label distributions for seed 0, and Fig. 3 shows how the same blood cells look at five simulated sites. Validation uses the official validation split (1,000 images for BloodMNIST, all 1,003 for DermaMNIST), and all final results are computed on the complete official test splits (3,421 and 2,005 images), which are not shifted and therefore act as a reference site.`));
add(FIGURE("fig2_partition.png", "partition", "Client-by-class sample counts for seed 0 (cells without a number are empty). In BloodMNIST (α = 0.1) most clients hold two to four of the eight cell types; in DermaMNIST (α = 0.3) the natural dominance of nevi (nv) is compounded by label skew, and several clients hold no dermatofibroma (df) or vascular lesion (vasc) images."));
add(FIGURE("fig3_sites.png", "sites", "Simulated acquisition shift in the Blood-shift setting: the same six test cells (top row) rendered with the stain-intensity, exposure, blur and noise parameters of sites 1–5 (seed 0). Each client's training images are transformed once with its own site parameters; the test set is left unshifted."));
add(H2("C", "Model and Training Protocol"));
add(P(`All methods share the same compact CNN (three 3×3 convolution blocks with batch normalization and ReLU, 16/32/64 channels, global average pooling to a *d* = ${d} feature vector and a linear classifier; ${bs.n_params.toLocaleString("en-US")} parameters including normalization statistics), the same initialization for a given seed and the same local optimizer (SGD, learning rate ${bs.train_cfg.lr}, momentum ${bs.train_cfg.momentum}, weight decay ${bs.train_cfg.weight_decay}, batch size ${bs.train_cfg.batch_size}, ${bs.train_cfg.local_epochs} local epoch per round, ${bs.train_cfg.rounds} rounds). Flips and 90° rotations are used as augmentation. The small model reflects the memory budget of edge clients and keeps the 66 federated runs of this study tractable on a single CPU core; Table II lists the configuration.`));
add(TABLE("config", "Experimental Configuration",
  [["Item", "Value"]],
  [["Clients / participation", "10 / all clients every round"],
   ["Partition", "Dirichlet α = 0.1 (Blood-shift), α = 0.3 (Derma)"],
   ["Training images (per seed)", "4,000 (class-stratified subsample)"],
   ["Rounds × local epochs", `${bs.train_cfg.rounds} × ${bs.train_cfg.local_epochs}`],
   ["Optimizer", `SGD, lr ${bs.train_cfg.lr}, momentum ${bs.train_cfg.momentum}, wd ${bs.train_cfg.weight_decay}, batch ${bs.train_cfg.batch_size}`],
   ["Model", `compact CNN, ${bs.n_params.toLocaleString("en-US")} parameters, *d* = ${d}`],
   ["Seeds", "0, 1, 2 (re-draw subsample, partition, site shift, init.)"],
   ["Evaluation", "final round, full official test split"],
   ["Hardware", "1 CPU core (Intel Xeon), PyTorch 2"]],
  [1900, 3140]));
add(H2("D", "Baselines and Hyper-Parameters"));
add(P("We compare with FedAvg {cite:mcmahan2017}, FedProx (μ = 0.01) {cite:li2020fedprox}, SCAFFOLD {cite:karimireddy2020}, FedNova {cite:wang2020fednova}, MOON (μ = 1, temperature 0.5, applied to the penultimate features) {cite:li2021moon}, FedRS (α = 0.5) {cite:li2021fedrs}, FedLC (τ = 1) {cite:zhang2022fedlc} and post-hoc CCVR {cite:luo2021ccvr} applied to the final FedAvg model; hyper-parameters are the defaults recommended by the original papers. Because all methods use momentum SGD, the SCAFFOLD control variates and the FedNova normalizer both use the effective number of local steps of heavy-ball SGD, [τ − ρ(1 − ρ^{τ})/(1 − ρ)]/(1 − ρ) {cite:wang2020fednova}; with the plain-SGD rule SCAFFOLD diverged in our pilot runs. FedCARE uses τ = 1 and has no tuned parameter; LR-Cal uses *r* = 4, the same virtual-sample budget and the same optimizer as CCVR."));
add(H2("E", "Metrics and Statistical Protocol"));
add(P("We report accuracy, balanced accuracy, macro-F1, macro-AUC (one-vs-rest), worst-class recall and per-class precision, recall and F1 of the final global model on the full test split; no checkpoint is selected on test data. Every configuration is run with seeds 0, 1 and 2, and each seed re-draws the training subsample, the Dirichlet partition, the site parameters and the initialization, giving three independent federations per setting. We report mean ± standard deviation and compare FedCARE with every baseline using the two-sided exact Wilcoxon signed-rank test {cite:wilcoxon1945,demsar2006} and a paired *t*-test over the six (setting, seed) pairs; with six pairs the smallest attainable Wilcoxon *p*-value is 0.031. Design decisions made during development (for example, removing subspace alignment from the default configuration) used only the validation split of seed 0 of Blood-shift."));
add(H2("F", "Implementation and Reproducibility"));
add(P("The framework is implemented in PyTorch and released with configuration files for every run, the raw per-run result files, scripts that regenerate all tables and figures, and a unit-test suite (exact statistic merging, aggregation rules, partitioning and an end-to-end run of every method) at https://github.com/alirezafathi84/FedCARE. A full run of one method takes 30 to 50 s on one CPU core.", { align: AlignmentType.LEFT }));

add(...R.results(H));
add(...R.discussion(H));

// ------------------------------------------------------------------ references
const refParas = [new Paragraph({ children: [new TextRun({ text: "References", font: FONT, size: 20, smallCaps: true })], alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 } })];
const appendix = R.appendix(H);
// appendix may cite nothing new; build reference list last
order.forEach((k, i) => refParas.push(new Paragraph({
  children: [new TextRun({ text: `[${i + 1}]\t`, font: FONT, size: 16 }), ...runs(REFS[k], { size: 16 })],
  alignment: AlignmentType.JUSTIFIED, indent: { left: 400, hanging: 400 }, tabStops: [{ type: TabStopType.LEFT, position: 400 }], spacing: { after: 30 },
})));

// ------------------------------------------------------------------ sections
const page = { size: { width: 12240, height: 15840 }, margin: { top: 1080, bottom: 1440, left: 900, right: 900, footer: 600 } };
const footer = { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16 })] })] }) };
const two = { count: 2, space: 360, equalWidth: true };
const one = { count: 1, space: 0 };
// split body at "wide" markers produced by R/H (paragraph objects tagged __wide)
const sections = [{ properties: { page, column: one }, footers: footer, children: [title, ...author] }];
let cur = [abstract, keywords];
const flush = (cols, type = SectionType.CONTINUOUS) => { if (cur.length) sections.push({ properties: { type, page, column: cols }, footers: footer, children: cur }); cur = []; };
for (const el of body) {
  if (el && el.__wide) { flush(two); cur = el.items; flush(one); }
  else cur.push(el);
}
flush(two);
cur = appendix; flush(one, SectionType.NEXT_PAGE);
cur = [...R.acknowledgment(H), ...refParas]; flush(two);

const doc = new Document({
  footnotes,
  styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  sections,
});
for (const [k, v] of Object.entries(FIGNUM)) if (figLabel[k] !== v) throw new Error(`figure ${k}: ${figLabel[k]} != ${v}`);
for (const [k, v] of Object.entries(TABNUM)) if (tabLabel[k] !== v) throw new Error(`table ${k}: ${tabLabel[k]} != ${v}`);
Packer.toBuffer(doc).then((buf) => {
  const out = process.argv[2] || path.join(ROOT, "paper", "FedCARE_manuscript.docx");
  fs.writeFileSync(out, buf);
  console.log("written", out, "figures", figNo, "tables", tabNo, "equations", eqNo, "refs", order.length);
});
