// Builds report/drbm25_results.pptx from report/data/*.json (scripts/report_data.py) and the chart images in
// report/figures/deck_*.png (scripts/deck_figures.py). Run: cd report/deck && npm ci && node build.js
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const REPO = path.resolve(__dirname, "../..");
const T = JSON.parse(fs.readFileSync(path.join(REPO, "report/data/training.json")));
const S = JSON.parse(fs.readFileSync(path.join(REPO, "report/data/sampling.json")));
const OUT = path.join(REPO, "report/drbm25_results.pptx");

const HEX = { normal: "1F6F8B", once: "E07B39", untrained: "B07AA1", raw: "6B6B6B", ink: "1B2631", grid: "D9E1E6", soft: "EEF3F5" };
const THEME = {
  name: "Rewrite study",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: HEX.ink, lt1: "FFFFFF", dk2: "16485A", lt2: HEX.soft,
    accent1: HEX.normal, accent2: HEX.once, accent3: HEX.untrained, accent4: HEX.raw, accent5: "2E8B57", accent6: "C0392B",
    hlink: HEX.normal, folHlink: HEX.untrained,
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.title = "Same score, different behavior";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "DARK",
  background: { color: HEX.ink },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.3, w: 11.7, h: 1.6, fontSize: 40, bold: true, color: "FFFFFF", fontFace: "Cambria", valign: "middle" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.0, w: 11.7, h: 1.4, fontSize: 18, color: "CFD8DC", fontFace: "Calibri", valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: "FFFFFF" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.9, fontSize: 30, bold: true, color: HEX.ink, fontFace: "Cambria", valign: "middle" }, text: "" } },
  ],
  slideNumber: { x: 12.5, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: "8A9AA5", fontFace: "Calibri" },
});

const fmt = (x, d = 3) => Number(x).toFixed(d);
const pct = (x, d = 1) => (100 * x).toFixed(d) + "%";
const body = (slide, text, opts) => slide.addText(text, { isTextBox: true, fontFace: "Calibri", fontSize: 16, color: HEX.ink, valign: "top", margin: 0, ...opts });
const caption = (slide, text, opts) => slide.addText(text, { isTextBox: true, fontFace: "Calibri", fontSize: 11, color: "5F6B73", margin: 0, ...opts });
const bullets = (slide, items, opts) =>
  slide.addText(
    items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 8 } })),
    { isTextBox: true, fontFace: "Calibri", fontSize: 16, color: HEX.ink, valign: "top", margin: 0, ...opts },
  );
const stat = (slide, big, label, x, y, w, color) => {
  slide.addText(big, { isTextBox: true, x, y, w, h: 0.9, fontFace: "Cambria", fontSize: 40, bold: true, color, margin: 0 });
  slide.addText(label, { isTextBox: true, x, y: y + 0.9, w, h: 0.7, fontFace: "Calibri", fontSize: 13, color: "5F6B73", margin: 0, valign: "top" });
};
const card = (slide, x, y, w, h, name) =>
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: HEX.soft }, line: { color: HEX.soft }, rectRadius: 0.12, objectName: name });
const chartBase = (title) => ({
  showTitle: true, title, titleFontSize: 13, titleColor: HEX.ink, titleFontFace: "+mj-lt",
  catAxisLabelColor: "5F6B73", valAxisLabelColor: "5F6B73", catAxisLabelFontSize: 10, valAxisLabelFontSize: 10,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  valGridLine: { color: HEX.grid, size: 0.5 }, catGridLine: { style: "none" },
  legendFontSize: 11, legendFontFace: "+mn-lt", legendColor: HEX.ink,
});
const img = (slide, file, x, y, w, h, alt) => slide.addImage({ path: path.join(REPO, "report/figures", file), x, y, w, h, altText: alt });
const curve = (mode, key) => T.per_run[mode].curves[key];
const lineData = (key, keyOnce) => {
  const a = curve("normal", key), b = curve("count_once", keyOnce || key);
  return [
    { name: "normal reward", labels: a.map((p) => String(p[0])), values: a.map((p) => p[1]) },
    { name: "count-once reward", labels: b.map((p) => String(p[0])), values: b.map((p) => p[1]) },
  ];
};
const lineOpts = (title, extra) => ({
  ...chartBase(title), chartColors: [HEX.normal, HEX.once], lineSize: 2, lineDataSymbol: "none",
  catAxisLabelFrequency: 25, showLegend: true, legendPos: "b", catAxisTitle: "PPO step", showCatAxisTitle: true,
  catAxisTitleFontSize: 10, catAxisTitleColor: "5F6B73", ...extra,
});

const m = S.means;
const P = Object.fromEntries(S.pair_comparisons.map((c) => [c.metric, c]));
const G = Object.fromEntries(S.greedy_eval.ndcg.map((r) => [r.source, r]));
const ci = S.greedy_eval.paired_ci[0];
const pat = S.patterns;
const nr = T.per_run.normal, co = T.per_run.count_once;

// 1 Title
{
  const s = pres.addSlide({ masterName: "DARK" });
  s.addText("DeepRetrieval revisited: same score, different behavior", { placeholder: "title" });
  s.addText("Reproducing and extending DeepRetrieval (Jiang et al., 2025) on MS MARCO-Science with BM25\nDoes counting repeated query words in the reward change what the rewriter learns?", { placeholder: "body" });
  s.addNotes("Two copies of the same model, trained the same way, except for one detail of the reward. The answer to the headline question: retrieval quality is the same within about ±0.005 nDCG (95% CI), but the two trained models behave differently in ways the score does not show (with one seed each, some of that may be run-to-run variation).");
}

// 1b Paper
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Starting point: the DeepRetrieval paper", { placeholder: "title" });
  card(s, 0.6, 1.45, 5.9, 4.6, "paper");
  s.addText("What the paper did", { isTextBox: true, x: 0.85, y: 1.6, w: 5.4, h: 0.4, fontFace: "Cambria", fontSize: 18, bold: true, color: HEX.ink, margin: 0 });
  bullets(s, [
    "Trains Qwen2.5-3B-Instruct with reinforcement learning (PPO) to rewrite search queries; the reward is how well the right documents rank",
    "Reports beating prompted GPT-4o and Claude-3.5-Sonnet; with BM25 it is best on 7 of 8 datasets (Table 2)",
    "The setting we reproduce: MS MARCO-Science with BM25, nDCG@10 41.1 vs 38.8 for the original query (Table 2)",
  ], { x: 0.85, y: 2.1, w: 5.4, h: 3.8, fontSize: 15 });
  card(s, 6.8, 1.45, 5.9, 4.6, "ours");
  s.addText("What we add", { isTextBox: true, x: 7.05, y: 1.6, w: 5.4, h: 0.4, fontFace: "Cambria", fontSize: 18, bold: true, color: HEX.once, margin: 0 });
  bullets(s, [
    "An independent re-implementation of the training, rerun on that setting",
    "A reward ablation: does the reward let the model game BM25 by repeating words?",
    "Training diagnostics the paper does not report: entropy, KL, critic, validation noise",
    "A behavior analysis: 32 samples per test question, compared with confidence intervals",
    "A check of the paper's text against its released code",
  ], { x: 7.05, y: 2.1, w: 5.4, h: 3.8, fontSize: 15 });
  caption(s, "Jiang, Lin, Cao, Tian, Kang, Wang, Sun, Han. DeepRetrieval: Hacking Real Search Engines and Retrievers with Large Language Models via Reinforcement Learning. arXiv:2503.00223 (v3).", { x: 0.6, y: 6.3, w: 12.1, h: 0.5 });
  s.addNotes("This is a reproduction and extension, not a new method. We picked one cell of the paper's Table 2, MS-S with BM25, and asked what is behind the number.");
}

// 2 Task
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("The task: rewrite a question so keyword search finds the answer", { placeholder: "title" });
  const steps = [
    ["Question", "“3 characteristics of protozoa replication”"],
    ["Model rewrites it", "reasons in <think>, then writes a search query"],
    ["BM25 search", "ranks 800,000 science passages; keep the top 10"],
    ["Score: nDCG@10", "searched as written, this question finds its passage at rank 6 → 0.36"],
  ];
  steps.forEach(([h, t], i) => {
    const x = 0.6 + i * 3.1;
    card(s, x, 1.5, 2.8, 1.75, `task${i}`);
    s.addText(`${i + 1}  ${h}`, { isTextBox: true, x: x + 0.2, y: 1.65, w: 2.4, h: 0.4, fontFace: "Cambria", fontSize: 16, bold: true, color: HEX.normal, margin: 0 });
    s.addText(t, { isTextBox: true, x: x + 0.2, y: 2.1, w: 2.4, h: 1.05, fontFace: "Calibri", fontSize: 14, color: HEX.ink, margin: 0, valign: "top" });
  });
  s.addText("nDCG@10: how high the relevant passage ranks", { isTextBox: true, x: 0.6, y: 3.6, w: 6.0, h: 0.4, fontFace: "Cambria", fontSize: 18, bold: true, color: HEX.ink, margin: 0 });
  const nd = [["Rank of relevant passage", "nDCG@10"], ["1", "1.00"], ["2", "0.63"], ["6", "0.36"], ["not in top 10", "0"]];
  s.addTable(nd.map((r, i) => r.map((c) => ({ text: c, options: { bold: i === 0, color: HEX.ink, fill: { color: i === 0 ? HEX.soft : "FFFFFF" } } }))), {
    x: 0.6, y: 4.1, w: 5.6, colW: [3.6, 2.0], fontFace: "Calibri", fontSize: 14, border: { type: "solid", color: HEX.grid, pt: 0.5 }, rowH: 0.4,
  });
  s.addText("Data: MS MARCO-Science", { isTextBox: true, x: 7.0, y: 3.6, w: 5.7, h: 0.4, fontFace: "Cambria", fontSize: 18, bold: true, color: HEX.ink, margin: 0 });
  bullets(s, [
    "22,416 training questions; 500 held out for validation",
    "2,305 test questions",
    "About 94% of questions have exactly one relevant passage",
    "Score reported: mean nDCG@10 over questions",
  ], { x: 7.0, y: 4.15, w: 5.7, h: 2.4, fontSize: 16 });
  s.addNotes("This is the DeepRetrieval setup we reproduced (Jiang et al., 2025, arXiv:2503.00223). Keyword search misses passages when the question's wording differs from the passage's, so a model rewrites the question first. nDCG@10 with one relevant passage is 1 over log2 of rank plus 1. In this example from the test set the raw question already finds its passage at rank 6.");
}

// 3 Reward difference
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("The one difference between the two runs: whether repeated words count", { placeholder: "title" });
  card(s, 0.6, 1.45, 12.1, 1.75, "bm25");
  s.addText("How BM25 scores", { isTextBox: true, x: 0.85, y: 1.6, w: 11.6, h: 0.4, fontFace: "Cambria", fontSize: 17, bold: true, color: HEX.ink, margin: 0 });
  body(s, "Words are lowercased, common words (“the”, “is”, “and”) dropped, and the rest stemmed (“fibre” → “fibr”). A passage scores higher for each query word it contains, more for rare words. Word order and AND / OR are ignored.", { x: 0.85, y: 2.05, w: 11.6, h: 1.1, fontSize: 15 });
  card(s, 0.6, 3.45, 5.9, 1.9, "normal");
  s.addText("Normal reward", { isTextBox: true, x: 0.85, y: 3.6, w: 5.4, h: 0.4, fontFace: "Cambria", fontSize: 18, bold: true, color: HEX.normal, margin: 0 });
  body(s, "Standard BM25: a word written twice in the query counts twice. Repeating words can raise the score.", { x: 0.85, y: 4.1, w: 5.4, h: 1.1 });
  card(s, 6.8, 3.45, 5.9, 1.9, "once");
  s.addText("Count-once reward", { isTextBox: true, x: 7.05, y: 3.6, w: 5.4, h: 0.4, fontFace: "Cambria", fontSize: 18, bold: true, color: HEX.once, margin: 0 });
  body(s, "Duplicate words are removed before searching. Repetition cannot help.", { x: 7.05, y: 4.1, w: 5.4, h: 1.1 });
  body(s, "Why ask: the untrained model repeats words in 42% of its rewrites, and the paper reports a model trained without reasoning that learned to repeat query terms (Appendix D.1). Would a rewriter trained on standard BM25 game it this way, and would removing that option change what it learns?", { x: 0.6, y: 5.65, w: 12.1, h: 1.0, fontSize: 16 });
  s.addNotes("Everything else is held fixed: same model, data order, settings and random seed. The DeepRetrieval prompt also tells the model to use Boolean operators and parentheses, which this BM25 ignores; that is why the untrained model writes queries full of AND and OR.");
}

// 4 Training procedure
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Training: reinforcement learning (PPO) with the search engine as judge", { placeholder: "title" });
  const loop = [
    ["Sample", "64 training questions"],
    ["Generate", "one response each at temperature 0.6: reasoning, then a query"],
    ["Reward", "0.1 + nDCG@10 if the passage is in the top 10; 0 if not; −2 if malformed"],
    ["Update", "a critic predicts the reward; PPO shifts the model toward responses that beat the prediction"],
  ];
  loop.forEach(([h, t], i) => {
    const x = 0.6 + i * 3.1;
    card(s, x, 1.5, 2.8, 2.1, `loop${i}`);
    s.addText(`${i + 1}  ${h}`, { isTextBox: true, x: x + 0.2, y: 1.65, w: 2.4, h: 0.4, fontFace: "Cambria", fontSize: 16, bold: true, color: HEX.normal, margin: 0 });
    s.addText(t, { isTextBox: true, x: x + 0.2, y: 2.1, w: 2.4, h: 1.4, fontFace: "Calibri", fontSize: 14, color: HEX.ink, margin: 0, valign: "top" });
  });
  bullets(s, [
    "Repeated for 1,751 steps per run: 112,064 responses, about five passes over the 21,916 training questions",
    "A small penalty (KL 0.001) keeps the model near where it started; a small entropy bonus (0.001) discourages it from becoming fully deterministic",
    "Every 50 steps: greedy rewrites (always the most likely word) of the 500 validation questions; the best checkpoint is kept (normal step 1549, count-once step 1349)",
    "Model: Qwen2.5-3B-Instruct, all weights trained; same seed and data order for both runs",
  ], { x: 0.6, y: 3.95, w: 12.1, h: 2.7, fontSize: 16 });
  caption(s, "Actor learning rate 1e-6, critic 1e-5, PPO clip 0.2, one epoch per batch, gradient clip 1.0. 2× RTX PRO 6000; both runs together took about 15.6 hours.", { x: 0.6, y: 6.75, w: 12.1, h: 0.4 });
  s.addNotes("PPO in one sentence: sample responses, score them, and make better-than-expected responses more likely, in small steps. The critic is a second network that learns the expected reward so the update knows what counts as better than expected. The two runs were trained alternately on the same two GPUs in 50-step chunks.");
}

// 5 What we analyzed
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("What we analyzed", { placeholder: "title" });
  const cols = [
    ["Training logs", "Every step of both runs: reward, nDCG under both scorers, how often rewrites repeat words, response length, and training-health signals. Validation every 50 steps.", "Did training work, and did the rewards differ?"],
    ["Greedy test", "One rewrite per test question (2,305) from four sources: the raw question, the untrained model, and the two trained models.", "Which retrieves better?"],
    ["32 samples per question", "The untrained and both trained models answered every test question 32 times (73,760 responses each): reasoning, rewrite, retrieved passages, scores.", "How do the models behave beyond one answer?"],
  ];
  cols.forEach(([h, t, q], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 1.5, 3.8, 4.1, `src${i}`);
    s.addText(h, { isTextBox: true, x: x + 0.25, y: 1.7, w: 3.3, h: 0.45, fontFace: "Cambria", fontSize: 19, bold: true, color: [HEX.normal, HEX.raw, HEX.once][i], margin: 0 });
    s.addText(t, { isTextBox: true, x: x + 0.25, y: 2.25, w: 3.3, h: 2.2, fontFace: "Calibri", fontSize: 15, color: HEX.ink, margin: 0, valign: "top" });
    s.addText(q, { isTextBox: true, x: x + 0.25, y: 4.6, w: 3.3, h: 0.85, fontFace: "Calibri", fontSize: 15, italic: true, color: "5F6B73", margin: 0, valign: "top" });
  });
  body(s, "Comparisons pair the two models question by question: average each question's samples, then a bootstrap over questions gives 95% confidence intervals. With 33 behavior metrics tested, p-values are corrected for multiple comparisons.", { x: 0.6, y: 5.85, w: 12.1, h: 0.9, fontSize: 15 });
  s.addNotes("The untrained model is the same Qwen model with the same prompt, before any training. All samples use the training temperature and the same random seed per question for every model, so differences come from the models, not the sampling.");
}

// 3 Headline
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Retrieval quality is the same for both rewards", { placeholder: "title" });
  img(s, "deck_headline.png", 0.6, 1.4, 7.4, 5.3, "Bar chart of greedy nDCG@10: raw query 0.388, untrained 0.258, normal 0.397, count-once 0.396");
  stat(s, fmt(ci.mean_diff, 4), `count-once − normal; 95% CI [${fmt(ci.ci_low, 4)}, ${fmt(ci.ci_high, 4)}]`, 8.5, 1.6, 4.2, HEX.ink);
  stat(s, "+" + fmt(G["trained-normal"].ndcg10_normal - G.original.ndcg10_normal, 3), "trained (normal) vs raw query, greedy; over 32 samples +0.008 [0.001, 0.015]", 8.5, 3.3, 4.2, HEX.normal);
  stat(s, "+" + fmt(G["trained-normal"].ndcg10_normal - G.untrained.ndcg10_normal, 3), "trained (normal) vs untrained model", 8.5, 5.0, 4.2, HEX.untrained);
  s.addNotes(`Greedy decoding on all 2,305 test questions. Count-once minus normal is ${fmt(ci.mean_diff, 4)} with a paired bootstrap CI that straddles zero. With 32 samples per question the CI is [${fmt(P.ndcg_normal_mean.ci_low, 4)}, ${fmt(P.ndcg_normal_mean.ci_high, 4)}]. The trained models score about 14 points above the untrained model and about 1 point above the raw question.`);
}

// 6b Paper comparison
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Compared with the paper: baselines match, the trained gain is smaller", { placeholder: "title" });
  const rows = [
    ["nDCG@10 (×100), MS-S, BM25", "Paper", "Ours"],
    ["Original query", "38.8", "38.8"],
    ["Qwen2.5-3B, zero-shot with reasoning (untrained)", "25.9", "25.8"],
    ["DeepRetrieval-3B (trained)", "41.1", "39.7 normal / 39.6 count-once"],
  ];
  s.addTable(rows.map((r, i) => r.map((c) => ({ text: c, options: { bold: i === 0, color: HEX.ink, fill: { color: i === 0 ? HEX.soft : "FFFFFF" } } }))), {
    x: 0.6, y: 1.4, w: 12.1, colW: [6.1, 2.5, 3.5], fontFace: "Calibri", fontSize: 15, border: { type: "solid", color: HEX.grid, pt: 0.5 }, rowH: 0.42,
  });
  s.addText("Where we followed the released code instead of the paper's text", { isTextBox: true, x: 0.6, y: 3.35, w: 12.1, h: 0.4, fontFace: "Cambria", fontSize: 17, bold: true, color: HEX.ink, margin: 0 });
  const d = [
    ["", "Paper text", "Released code", "Ours"],
    ["nDCG cutoff in the training reward", "k = 3000", "k = 10", "k = 10"],
    ["Maximum response length", "512 tokens", "350", "350"],
    ["Validation during training", "not described", "test set, every 50 steps", "500 held-out training questions"],
  ];
  s.addTable(d.map((r, i) => r.map((c) => ({ text: c, options: { bold: i === 0, color: HEX.ink, fill: { color: i === 0 ? HEX.soft : "FFFFFF" } } }))), {
    x: 0.6, y: 3.85, w: 12.1, colW: [4.3, 2.4, 2.9, 2.5], fontFace: "Calibri", fontSize: 14, border: { type: "solid", color: HEX.grid, pt: 0.5 }, rowH: 0.4,
  });
  body(s, "None of these is tested as a cause of the 1.4-point gap. With k = 10, 44% of training responses got no retrieval reward, the sparse signal the paper says k = 3000 avoids. Training length and checkpoint choice are other candidates.", { x: 0.6, y: 5.75, w: 12.1, h: 0.9, fontSize: 15 });
  s.addNotes("Matches the paper's text: model, batch 64, mini-batch 16, learning rates, KL coefficient, temperature. Matches only the released code, which the paper does not specify: the reward constants, the entropy coefficient and the exact prompt. The paper is also inconsistent with itself: Table 8 gives the sparse reward as the evaluation metric, Appendix B.2 says k = 3000. The only released MS MARCO script is for the technology subset and resumes from step 450 of a run named k_10, so the paper's MS-S training length is unknown. The paper reports single numbers without confidence intervals and we do not have its per-question results, so we cannot say whether the gap exceeds run-to-run variation. The paper does not say how its checkpoint was chosen; the released script evaluates on the test set every 50 steps.");
}

// 6c Prompt and Boolean operators
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("The paper's prompt asks for Boolean operators that BM25 ignores here", { placeholder: "title" });
  card(s, 0.6, 1.45, 6.3, 2.35, "promptcard");
  s.addText("From the prompt (released code; paper Figure 15)", { isTextBox: true, x: 0.85, y: 1.6, w: 5.8, h: 0.4, fontFace: "Cambria", fontSize: 16, bold: true, color: HEX.ink, margin: 0 });
  s.addText("“You are a query rewriting expert. Your task is to augment the user query to find relevant literature in a corpus with sparse retrieval. … Note: The query should use Boolean operators (AND, OR) and parentheses for grouping terms appropriately.”", { isTextBox: true, x: 0.85, y: 2.05, w: 5.8, h: 1.65, fontFace: "Calibri", fontSize: 14, italic: true, color: HEX.ink, margin: 0, valign: "top" });
  card(s, 7.1, 1.45, 5.6, 2.35, "analyzecard");
  s.addText("What BM25 searches", { isTextBox: true, x: 7.35, y: 1.6, w: 5.1, h: 0.4, fontFace: "Cambria", fontSize: 16, bold: true, color: HEX.ink, margin: 0 });
  s.addText([
    { text: "Pascal's Law AND (definition OR applications OR examples) IN CONTEXT", options: { breakLine: true, fontFace: "Courier New", fontSize: 12 } },
    { text: "↓ analyzed as a bag of words", options: { breakLine: true, fontSize: 13, color: "5F6B73" } },
    { text: "pascal law definit applic exampl context", options: { fontFace: "Courier New", fontSize: 12 } },
  ], { isTextBox: true, x: 7.35, y: 2.05, w: 5.1, h: 1.65, fontFace: "Calibri", color: HEX.ink, margin: 0, valign: "top", paraSpaceAfter: 6 });
  const rows = [
    ["", "Rewrites using AND / OR / ( )", "Effect"],
    ["Untrained", "98%", "Within a question, Boolean rewrites score 0.027 lower [−0.046, −0.009] and carry 1.4 more terms"],
    ["Normal", "11.6% (AND only)", "“Sea Otters AND endangered AND Alaska”: same search as without AND"],
    ["Count-once", "0.3%", "Learned to drop them almost entirely"],
  ];
  s.addTable(rows.map((r, i) => r.map((c) => ({ text: c, options: { bold: i === 0, color: HEX.ink, fill: { color: i === 0 ? HEX.soft : "FFFFFF" } } }))), {
    x: 0.6, y: 4.05, w: 12.1, colW: [1.9, 3.0, 7.2], fontFace: "Calibri", fontSize: 14, border: { type: "solid", color: HEX.grid, pt: 0.5 }, rowH: 0.45,
  });
  body(s, "The paper describes sparse retrieval as “keyword matching and boolean operations with BM25” (Table 8). In its released code, as in ours, the rewrite goes to pyserini as plain text, so the operators have no effect; an OR clause just adds words.", { x: 0.6, y: 6.0, w: 12.1, h: 0.85, fontSize: 14 });
  s.addNotes("The paper motivates Boolean syntax with real search engines (PubMed, ClinicalTrials.gov) used for literature search, where operators do work. Our guess, untested, is that the instruction was carried over into the BM25 prompt. RL taught both models to ignore an instruction in their own prompt. The within-question comparison uses the 630 test questions where the untrained model's 32 samples include both Boolean and plain rewrites.");
}

// 4 Training reward
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Both runs climbed to the same training reward", { placeholder: "title" });
  img(s, "deck_reward.png", 0.6, 1.4, 8.2, 5.4, "Training reward curves for both runs");
  bullets(s, [
    `Normal: ${fmt(nr["reward/mean"].first50)} → ${fmt(nr["reward/mean"].last50)} (first vs last 50 steps)`,
    `Count-once: ${fmt(co["reward/mean"].first50)} → ${fmt(co["reward/mean"].last50)}; most of the rise came in the first 150 steps`,
    "The two scorers' batch nDCG differs by 0.003 to 0.005 in the first 50 steps and 0.002 to 0.004 on average: the rewards sent almost the same signal",
  ], { x: 9.1, y: 1.6, w: 3.7, h: 5.0, fontSize: 15 });
  s.addNotes("The reward includes a 0.1 bonus for any nonzero nDCG, which is why it sits above nDCG. Most of the climb happens in the first 150 steps. The two scorers agree on almost every rewrite, so the two runs were optimizing nearly the same objective.");
}

// 5 Validation
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Validation: normal stayed flat, count-once jumped after step 900", { placeholder: "title" });
  const vn = nr.val.points, vc = co.val.points;
  img(s, "deck_val.png", 0.6, 1.4, 8.2, 5.4, "Validation nDCG every 50 steps for both runs");
  bullets(s, [
    "Greedy nDCG on 500 held-out questions every 50 steps; the best value picks the checkpoint",
    "Normal: mean 0.363 (steps 49 to 299) vs 0.366 (1049 to 1750); no trend",
    "Count-once: 0.347 to 0.355 from step 149 to 899, then mostly 0.366 to 0.372 (dips to 0.346 at 1049 and 0.356 at 1649)",
    "Single points swing by about 0.02; one 500-question mean has a standard error of about 0.017. The y-axis starts at 0.32",
  ], { x: 9.1, y: 1.6, w: 3.7, h: 5.0, fontSize: 15 });
  s.addNotes("Per-question nDCG has a standard deviation of about 0.39, so one 500-question mean has a standard error of about 0.017, the size of the point-to-point swings. Checkpoints are scored on the same 500 questions, so differences between them are less noisy than that, but the normal run's best step is still hard to separate from its neighbours. Count-once's rise of about 0.02 after step 900 holds from step 1099 to the end apart from two dips, so it is not one noisy value.");
}

// 6 Health
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Training was stable, but entropy collapsed early", { placeholder: "title" });
  img(s, "deck_entropy.png", 0.6, 1.35, 6.0, 3.0, "Policy entropy over training");
  img(s, "deck_kl.png", 6.8, 1.35, 6.0, 3.0, "KL divergence from the starting model over training");
  const rows = [
    ["Signal", "Normal", "Count-once"],
    ["Entropy, first → last 50 steps", `${fmt(nr["ppo/entropy"].first50)} → ${fmt(nr["ppo/entropy"].last50)}`, `${fmt(co["ppo/entropy"].first50)} → ${fmt(co["ppo/entropy"].last50)}`],
    ["KL from start model, last 50 steps", fmt(nr["ppo/kl_ref_mean"].last50, 2), fmt(co["ppo/kl_ref_mean"].last50, 2)],
    ["Response length (tokens), first → last 50 steps", `${fmt(nr["rollout/response_len_mean"].first50, 0)} → ${fmt(nr["rollout/response_len_mean"].last50, 0)}`, `${fmt(co["rollout/response_len_mean"].first50, 0)} → ${fmt(co["rollout/response_len_mean"].last50, 0)}`],
    ["Optimizer updates with actor gradient clipped to 1.0", "84%", "88%"],
    ["Largest actor gradient norm", fmt(nr["optim/actor_grad_norm"].max, 0), fmt(co["optim/actor_grad_norm"].max, 0)],
    ["Non-finite values / skipped steps", "0 / 0", "0 / 0"],
  ];
  s.addTable(rows.map((r, i) => r.map((c) => ({ text: c, options: { bold: i === 0, color: HEX.ink, fill: { color: i === 0 ? HEX.soft : "FFFFFF" } } }))), {
    x: 0.6, y: 4.55, w: 12.1, colW: [5.3, 3.4, 3.4], fontFace: "Calibri", fontSize: 13, border: { type: "solid", color: HEX.grid, pt: 0.5 }, rowH: 0.33,
  });
  s.addNotes("Entropy measures how spread out the next-token choices are; 0 means the policy always picks the same token. KL measures how far the policy's token probabilities moved from the untrained model. Response length fell in both runs, which matches the paper's observation on literature search that reasoning shortens during training. Entropy fell about 96 percent by step 300, which is where the low diversity in the sampling analysis comes from. The normal run drifted about twice as far from the starting model. Count-once had five gradient spikes above 100; clipping held every update to norm 1, and reward showed no lasting drop.");
}

// 7 Critic
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("The normal run's critic stayed weak", { placeholder: "title" });
  img(s, "deck_explained_var.png", 0.6, 1.4, 8.2, 5.4, "Critic explained variance over training");
  bullets(s, [
    "Explained variance: the share of reward variation the critic predicts (1 perfect, 0 no better than the average)",
    `At the end: ${fmt(nr["ppo/explained_var"].last50, 2)} (normal) vs ${fmt(co["ppo/explained_var"].last50, 2)} (count-once)`,
    `Value loss: ${fmt(nr["ppo/value_loss"].first50)} → ${fmt(nr["ppo/value_loss"].last50)} (normal), ${fmt(co["ppo/value_loss"].first50)} → ${fmt(co["ppo/value_loss"].last50)} (count-once)`,
    "Step 1000: the normal critic fell from 0.26 to 0 and recovered slowly; cause unknown",
    "Normal steps 1500 to 1640: responses halved (about 113 to 57 tokens), KL rose 1.6 to 2.4; the selected checkpoint (1549) is mid-shift",
  ], { x: 9.1, y: 1.5, w: 3.7, h: 5.3, fontSize: 14 });
  s.addNotes("A weaker critic makes the better-than-expected signal noisier, yet the normal run reached the same reward. The count-once critic may predict returns better because its policy is more deterministic; that is an inference we did not test. PPO clip fraction, the share of tokens whose update PPO capped, stayed below 2 percent: updates were small.");
}

// 8 Repetition
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Repetition stopped paying early, so the two rewards trained the same thing", { placeholder: "title" });
  const qs = Object.fromEntries(S.greedy_eval.query_stats.map((r) => [r.source, r]));
  img(s, "deck_repeat_train.png", 0.6, 1.4, 8.2, 5.4, "Share of training rewrites with a repeated term, both runs");
  bullets(s, [
    "Both runs started near 50% and were down to 5 to 7% by steps 100 to 150",
    "The normal run drifted back three times (25-step average up to 37%). Each time standard BM25 scored those rewrites 0.003 to 0.010 lower than count-once BM25, and reward did not rise",
    `Final greedy rewrites with a repeat: ${pct(qs["trained-normal"].any_repeat_rate)} (normal), ${pct(qs["trained-count_once"].any_repeat_rate)} (count-once), ${pct(qs.original.any_repeat_rate)} (raw query), ${pct(qs.untrained.any_repeat_rate)} (untrained)`,
  ], { x: 9.1, y: 1.6, w: 3.7, h: 5.2, fontSize: 14 });
  s.addNotes("This explains the identical scores. In the first 50 steps repetition gave standard BM25 a small edge of 0.003 to 0.005 nDCG; after that, whenever the normal policy repeated terms, standard BM25 scored it lower than count-once BM25 would have. We do not know what started the episodes, or why repetition stopped helping; both selected checkpoints fall outside them.");
}

// 9 Determinism
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Count-once is more deterministic and gains less from resampling", { placeholder: "title" });
  const ks = ["1", "2", "4", "8", "16"];
  const bk = (src) => [m[src].ndcg_normal_mean, m[src].best_of_2, m[src].best_of_4, m[src].best_of_8, m[src].best_of_16];
  img(s, "deck_best_of_k.png", 0.6, 1.4, 7.0, 5.4, "Expected best-of-k nDCG for k = 1 to 16");
  stat(s, "3.6 vs 1.6", "distinct rewrites out of 32 samples, normal vs count-once (untrained 31.3)", 8.0, 1.5, 4.8, HEX.ink);
  stat(s, `${fmt(m["trained-normal"].ndcg_normal_std)} vs ${fmt(m["trained-count_once"].ndcg_normal_std)}`, "within-query std of nDCG across 32 samples", 8.0, 3.2, 4.8, HEX.ink);
  stat(s, fmt(m.untrained.best_of_16, 2), "untrained best-of-16, above both trained policies (0.42, 0.40)", 8.0, 4.9, 4.8, HEX.untrained);
  s.addNotes(`All differences shown are significant after correction. Best-of-16: count-once minus normal ${fmt(P.best_of_16.mean_diff, 3)} [${fmt(P.best_of_16.ci_low, 3)}, ${fmt(P.best_of_16.ci_high, 3)}]. Training traded diversity for a better single sample; the untrained model's 32 samples contain far more good rewrites, but its average single sample is poor (0.263).`);
}

// 10 Rewrite style
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Both policies learned to delete words from the query", { placeholder: "title" });
  const labels = ["compression", "same", "substitution", "mixed", "expansion"];
  const rc = S.rewrite_class;
  img(s, "deck_rewrite_class.png", 0.6, 1.4, 7.6, 5.4, "Share of rewrites by type of term change");
  bullets(s, [
    `Only removing terms: ${pct(rc["trained-normal"].compression, 0)} (normal), ${pct(rc["trained-count_once"].compression, 0)} (count-once), ${pct(rc.untrained.compression, 0)} (untrained)`,
    `Trained policies keep about 72% of the query's terms and add almost none; the untrained model keeps ${pct(m.untrained.mean_retention, 0)} and adds about 3`,
  ], { x: 8.5, y: 1.7, w: 4.3, h: 3.6, fontSize: 15 });
  caption(s, "Types compare the rewrite's analyzed terms with the query's. Compression: only removes terms. Same: same terms. Expansion: only adds. Substitution: adds as many as it removes. Mixed: adds and removes different numbers.", { x: 8.5, y: 5.5, w: 4.3, h: 1.2 });
  s.addNotes("The learned strategy is to strip question words and filler. Our guess, not tested, is that most of the gain over the untrained model comes from no longer expanding the query with Boolean clauses full of extra terms.");
}

// 11 Examples
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("What the rewrites look like", { placeholder: "title" });
  const hdr = ["Query (raw nDCG)", "untrained (one sample)", "normal", "count-once"];
  const rows = [
    ["how many chromosomes do carp have (1.00)", "(chromosome count OR number of chromosomes) AND (carp OR fish) → 1.00", "how many chromosomes do Carp have (9/32) → 1.00", "number of chromosomes carp (30/32) → 1.00"],
    ["how far are the crust and mantle (0.00)", "how far are the crust and mantle AND located → 0.00", "distance between Crust and Mantle (31/32) → 1.00", "distance are the crust and mantle (18/32) → 1.00\nfar are the crust and mantle (10/32) → 0.00"],
    ["what food is fibre found in (1.00)", "fiber AND food AND (whole grains OR fruits OR vegetables OR legumes) → 0.00", "Food is Fiber found in (15/32) → 0.00", "food is fiber found (32/32) → 0.00"],
    ["a _______ is a pictorial representation of the chromosomes in a cell (1.00)", "chromosome AND (map OR diagram) → 0.00", "karyotype is a pictorial representation of the chromosomes in a cell (5/32) → 1.00", "a pictorial representation of the chromosomes in a cell (32/32) → 1.00"],
  ];
  const colors = [HEX.ink, HEX.untrained, HEX.normal, HEX.once];
  s.addTable([hdr.map((h, i) => ({ text: h, options: { bold: true, color: colors[i], fill: { color: HEX.soft } } })), ...rows.map((r) => r.map((c) => ({ text: c, options: { color: HEX.ink } })))], {
    x: 0.6, y: 1.35, w: 12.1, colW: [3.1, 3.3, 2.9, 2.8], fontFace: "Calibri", fontSize: 13, rowH: 0.9, border: { type: "solid", color: HEX.grid, pt: 0.5 }, valign: "middle",
  });
  caption(s, "Each trained cell: the most common rewrite, how many of 32 samples produced it, and that rewrite's nDCG@10. All 32 untrained samples differ, so one is shown (carp: this one scored 1.00, but the untrained mean is 0.64). Fibre: both trained policies drop 'what' and turn 'fibre' into 'fiber'; the relevant passage leaves the top 10. Karyotype: the normal policy writes the answer into the query.", { x: 0.6, y: 6.45, w: 11.8, h: 0.8, fontSize: 12 });
  s.addNotes("Pick one example to walk through. Carp shows the 'how many' versus 'number' habit. Crust and mantle is a big gain from rephrasing. Fibre is one of the largest losses; we did not test whether dropping 'what' or the spelling change causes it. The karyotype row shows the normal policy inserting the answer.");
}

// 12 Reasoning
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("The two policies reason in very different templates", { placeholder: "title" });
  card(s, 0.6, 1.45, 6.0, 2.4, "normalquote");
  s.addText("Normal", { isTextBox: true, x: 0.85, y: 1.6, w: 5.5, h: 0.4, fontFace: "Cambria", fontSize: 18, bold: true, color: HEX.normal, margin: 0 });
  s.addText("“The user query can be broken down into key terms: \"Chromosomes\" and \"Carp have X Chromosomes\". The number of chromosomes in Carp is 20.”", { isTextBox: true, x: 0.85, y: 2.05, w: 5.5, h: 2.1, fontFace: "Calibri", fontSize: 15, italic: true, color: HEX.ink, margin: 0, valign: "top" });
  card(s, 6.8, 1.45, 6.0, 2.4, "oncequote");
  s.addText("Count-once", { isTextBox: true, x: 7.05, y: 1.6, w: 5.5, h: 0.4, fontFace: "Cambria", fontSize: 18, bold: true, color: HEX.once, margin: 0 });
  s.addText("“The query is looking for the number of chromosomes that are found in a carp. This is a straightforward query that does not require the use of Boolean operators or parentheses for grouping terms, as the relationship between the terms is clear and unambiguous.”", { isTextBox: true, x: 7.05, y: 2.05, w: 5.5, h: 2.1, fontFace: "Calibri", fontSize: 15, italic: true, color: HEX.ink, margin: 0, valign: "top" });
  stat(s, pct(pat["trained-normal"].think_starts_broken_down, 0), "of normal samples open with “The user query can be broken down into key terms”", 0.6, 4.2, 3.9, HEX.normal);
  stat(s, pct(pat["trained-count_once"].think_starts_query_is_looking_for, 1), "of count-once samples open with “The query is looking for”", 4.75, 4.2, 3.9, HEX.once);
  stat(s, ">99.9%", "accuracy of a text classifier telling the two policies' reasoning apart on held-out questions (p ≤ 0.005)", 8.9, 4.2, 3.9, HEX.ink);
  body(s, `The normal policy states a number in its reasoning in ${pct(pat["trained-normal"].think_has_digit)} of samples, often answering the question (count-once ${pct(pat["trained-count_once"].think_has_digit)}, untrained ${pct(pat.untrained.think_has_digit)}).`, { x: 0.6, y: 6.0, w: 12.1, h: 0.6, fontSize: 15 });
  s.addNotes(`Same query, first sample from each policy. The normal policy often answers the question in its reasoning: ${pct(pat["trained-normal"].think_has_digit)} of its samples state a number, against ${pct(pat["trained-count_once"].think_has_digit)} for count-once and ${pct(pat.untrained.think_has_digit)} for the untrained model. Its answer here, 20 chromosomes, is not something we checked and should not be read as correct. The classifier is logistic regression on word and word-pair frequencies, tested on questions it was not trained on. The result shows the outputs differ; it picks up stock phrasing, so it does not show a difference in reasoning quality.`);
}

// 13 Same documents
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Different words, mostly the same documents", { placeholder: "title" });
  stat(s, "25%", "of queries where the two policies' most common rewrites have different BM25 terms (92% differ as exact text)", 0.8, 1.7, 3.7, HEX.ink);
  stat(s, pct(S.cross_policy.rbo_modal_eq1, 0), "of queries where those rewrites retrieve the identical top 10 (rank-biased overlap = 1)", 4.8, 1.7, 3.7, HEX.normal);
  stat(s, pct(S.c2st.rewrite.accuracy, 0), "accuracy telling the rewrites apart (chance 50%, p ≤ 0.005)", 8.8, 1.7, 3.7, HEX.once);
  bullets(s, [
    "Most text differences vanish in BM25's analyzer: normal capitalizes (“Xenon come from” vs “xenon come from”), and stopwords and stems map to the same terms",
    "Normal keeps question words (“how many”, “why”, “what”); count-once often swaps “how many” for “number”",
    `Count-once rewrites are slightly shorter (${fmt(m["trained-count_once"].mean_n_terms, 2)} vs ${fmt(m["trained-normal"].mean_n_terms, 2)} terms) and add fewer new terms (${fmt(m["trained-count_once"].mean_n_added, 2)} vs ${fmt(m["trained-normal"].mean_n_added, 2)})`,
  ], { x: 0.8, y: 3.8, w: 11.8, h: 3.1, fontSize: 17 });
  s.addNotes("Most wording differences are invisible to BM25 because they are stopwords, casing, or stems that map to the same terms. That is why behavior can differ without moving the score.");
}

// 14 Takeaways
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("What this means", { placeholder: "title" });
  bullets(s, [
    "Reproduction: the paper's baselines reproduce; our trained models are 1.4 points below its 41.1. Untested candidates: the k = 10 vs 3000 reward, training length, checkpoint selection.",
    "Extension: the count-once variant changed nothing in the score; repetition did not pay under standard BM25 after the first steps, consistent with the paper's claim that reasoning avoids it.",
    "The prompt's Boolean operators do nothing in this BM25 setup; training taught both models to drop them.",
    "Equal scores hid measurable differences: how deterministic each policy is, its reasoning template, whether it answers the question while reasoning, and its wording. With one seed per reward, some of this may be run-to-run variation.",
    "Limits: one seed per reward; a 500-question validation set too small to rank late checkpoints.",
    "Next: train with k = 3000 to test the gap; a second seed per reward; a larger entropy bonus to keep diversity, since best-of-16 shows the headroom.",
  ], { x: 0.8, y: 1.6, w: 11.7, h: 5.3, fontSize: 20 });
  s.addNotes("Close on the main point: measure behavior, not only the score.");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  console.log("wrote", OUT);
})();
