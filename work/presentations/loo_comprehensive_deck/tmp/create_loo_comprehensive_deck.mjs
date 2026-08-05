import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT = "C:/Users/A8327/OneDrive/Documents/OI/outputs/loo_comprehensive_self_contained_deck.pptx";
const TMP = "C:/Users/A8327/OneDrive/Documents/OI/work/presentations/loo_comprehensive_deck/tmp";
const PREVIEW_DIR = path.join(TMP, "preview_connected");
const MATH_DIR = path.join(TMP, "math");

async function writeBlob(filePath, blob) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

const mathDefs = {
  ladder: `<math><mrow><msub><mi>m</mi><mrow><mi>i</mi><mi>k</mi></mrow></msub><mo>-</mo><msub><mi>m</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo><msub><mi>ψ</mi><mi>k</mi></msub><mo>-</mo><msub><mi>ψ</mi><mi>j</mi></msub><mspace width="1em"/><mtext>for every </mtext><mi>i</mi><mo>,</mo><mi>j</mi><mo>,</mo><mi>k</mi></mrow></math>`,
  primitives: `<math><mrow><mi>m</mi><mo>:</mo><mi>𝒯</mi><mo>→</mo><mi>ℝ</mi><mspace width="2em"/><msubsup><mi>P</mi><mrow><mi>I</mi><mi>J</mi></mrow><mtext>obs</mtext></msubsup><mspace width="1.3em"/><mtext>versus</mtext><mspace width="1.3em"/><msub><mi>P</mi><mi>I</mi></msub><msub><mi>P</mi><mi>J</mi></msub></mrow></math>`,
  estimands: `<math><mtable columnalign="left"><mtr><mtd><msub><mi>Q</mi><mi>F</mi></msub><mo>=</mo><msub><mi>𝔼</mi><mi>I</mi></msub><mo>[</mo><msub><mi>Var</mi><mi>J</mi></msub><mo>(</mo><msub><mi>m</mi><mrow><mi>I</mi><mi>J</mi></mrow></msub><mo>)</mo><mo>]</mo></mtd><mtd><mspace width="2em"/><msub><mi>H</mi><mi>F</mi></msub><mo>=</mo><msub><mi>𝔼</mi><mrow><mi>J</mi><mo>,</mo><mi>K</mi></mrow></msub><mo>[</mo><msub><mi>Var</mi><mi>I</mi></msub><mo>(</mo><msub><mi>m</mi><mrow><mi>I</mi><mi>K</mi></mrow></msub><mo>-</mo><msub><mi>m</mi><mrow><mi>I</mi><mi>J</mi></mrow></msub><mo>)</mo><mo>]</mo></mtd></mtr><mtr><mtd><msubsup><mi>C</mi><mtext>assign</mtext><mi>w</mi></msubsup><mo>=</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><mo>{</mo><msub><mi>Var</mi><mtext>obs</mtext></msub><mo>(</mo><msub><mi>m</mi><mrow><mi>I</mi><mi>J</mi></mrow></msub><mo>)</mo><mo>-</mo><msub><mi>Var</mi><mtext>prod</mtext></msub><mo>(</mo><msub><mi>m</mi><mrow><mi>I</mi><mi>J</mi></mrow></msub><mo>)</mo><mo>}</mo></mtd><mtd><mspace width="2em"/><msub><mi>A</mi><mi>h</mi></msub><mo>=</mo><msub><mi>𝔼</mi><mtext>obs</mtext></msub><mo>[</mo><msub><mi>h</mi><mrow><mi>I</mi><mi>J</mi></mrow></msub><mo>]</mo></mtd></mtr></mtable></math>`,
  decomposition: `<math><mtable columnalign="left"><mtr><mtd><msub><mi>m</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo><mi>μ</mi><mo>+</mo><msub><mi>a</mi><mi>i</mi></msub><mo>+</mo><msub><mi>b</mi><mi>j</mi></msub><mo>+</mo><msub><mi>h</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub></mtd></mtr><mtr><mtd><msub><mi>𝔼</mi><mi>I</mi></msub><mo>[</mo><msub><mi>a</mi><mi>I</mi></msub><mo>]</mo><mo>=</mo><msub><mi>𝔼</mi><mi>J</mi></msub><mo>[</mo><msub><mi>b</mi><mi>J</mi></msub><mo>]</mo><mo>=</mo><mn>0</mn><mo>,</mo><mspace width="1em"/><msub><mi>𝔼</mi><mi>J</mi></msub><mo>[</mo><msub><mi>h</mi><mrow><mi>i</mi><mi>J</mi></mrow></msub><mo>]</mo><mo>=</mo><msub><mi>𝔼</mi><mi>I</mi></msub><mo>[</mo><msub><mi>h</mi><mrow><mi>I</mi><mi>j</mi></mrow></msub><mo>]</mo><mo>=</mo><mn>0</mn></mtd></mtr></mtable></math>`,
  orthogonality: `<math><mrow><mo>⟨</mo><mi>a</mi><mo>,</mo><mi>b</mi><mo>⟩</mo><mo>=</mo><mo>⟨</mo><mi>a</mi><mo>,</mo><mi>h</mi><mo>⟩</mo><mo>=</mo><mo>⟨</mo><mi>b</mi><mo>,</mo><mi>h</mi><mo>⟩</mo><mo>=</mo><mn>0</mn></mrow></math>`,
  firmSplit: `<math><mrow><msub><mi>Q</mi><mi>F</mi></msub><mo>=</mo><mi>Var</mi><mo>(</mo><msub><mi>b</mi><mi>J</mi></msub><mo>)</mo><mo>+</mo><msub><mi>Q</mi><mi>h</mi></msub><mo>=</mo><mi>Var</mi><mo>(</mo><msub><mi>b</mi><mi>J</mi></msub><mo>)</mo><mo>+</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><msub><mi>H</mi><mi>F</mi></msub></mrow></math>`,
  assignment: `<math><mtable columnalign="left"><mtr><mtd><msubsup><mi>C</mi><mtext>assign</mtext><mi>w</mi></msubsup><mo>=</mo><msub><mi>Cov</mi><mtext>obs</mtext></msub><mo>(</mo><msub><mi>a</mi><mi>I</mi></msub><mo>,</mo><msub><mi>b</mi><mi>J</mi></msub><mo>)</mo><mo>+</mo><msub><mi>Cov</mi><mtext>obs</mtext></msub><mo>(</mo><msub><mi>a</mi><mi>I</mi></msub><mo>,</mo><msub><mi>h</mi><mrow><mi>I</mi><mi>J</mi></mrow></msub><mo>)</mo></mtd></mtr><mtr><mtd><mspace width="3.2em"/><mo>+</mo><msub><mi>Cov</mi><mtext>obs</mtext></msub><mo>(</mo><msub><mi>b</mi><mi>J</mi></msub><mo>,</mo><msub><mi>h</mi><mrow><mi>I</mi><mi>J</mi></mrow></msub><mo>)</mo><mo>+</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><mo>[</mo><msub><mi>Var</mi><mtext>obs</mtext></msub><mo>(</mo><msub><mi>h</mi><mrow><mi>I</mi><mi>J</mi></mrow></msub><mo>)</mo><mo>-</mo><msub><mi>Var</mi><mtext>prod</mtext></msub><mo>(</mo><msub><mi>h</mi><mrow><mi>I</mi><mi>J</mi></mrow></msub><mo>)</mo><mo>]</mo></mtd></mtr></mtable></math>`,
  meanAssignment: `<math><mrow><msub><mi>𝔼</mi><mtext>obs</mtext></msub><mo>[</mo><msub><mi>m</mi><mrow><mi>I</mi><mi>J</mi></mrow></msub><mo>]</mo><mo>-</mo><msub><mi>𝔼</mi><mtext>prod</mtext></msub><mo>[</mo><msub><mi>m</mi><mrow><mi>I</mi><mi>J</mi></mrow></msub><mo>]</mo><mo>=</mo><msub><mi>𝔼</mi><mtext>obs</mtext></msub><mo>[</mo><msub><mi>h</mi><mrow><mi>I</mi><mi>J</mi></mrow></msub><mo>]</mo><mo>=</mo><msub><mi>A</mi><mi>h</mi></msub></mrow></math>`,
  cases: `<math><mtable columnalign="left"><mtr><mtd><mtext>AKM:</mtext><mspace width="1em"/><msub><mi>h</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo><mn>0</mn></mtd></mtr><mtr><mtd><mtext>Centered Tukey:</mtext><mspace width="1em"/><msub><mi>h</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo><msub><mi>β</mi><mn>0</mn></msub><msub><mi>α</mi><mi>i</mi></msub><msub><mi>ψ</mi><mi>j</mi></msub></mtd></mtr><mtr><mtd><mtext>Free-factor rank one:</mtext><mspace width="1em"/><msub><mi>h</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo><msub><mi>u</mi><mi>i</mi></msub><msub><mi>v</mi><mi>j</mi></msub></mtd></mtr></mtable></math>`,
  support: `<math><mrow><msub><mi>R</mi><mi>E</mi></msub><mi>m</mi><mspace width="2em"/><mtext>is observed, while</mtext><mspace width="2em"/><msub><mi>𝒦</mi><mi>E</mi></msub><mo>=</mo><mi>ker</mi><mo>(</mo><msub><mi>R</mi><mi>E</mi></msub><mo>)</mo><mspace width="1em"/><mtext>contains missing-cell perturbations.</mtext></mrow></math>`,
  identification: `<math><mtable columnalign="left"><mtr><mtd><msup><mi>ℓ</mi><mo>′</mo></msup><mi>m</mi><mspace width="1em"/><mtext>identified</mtext><mspace width=".8em"/><mo>⇔</mo><mspace width=".8em"/><msup><mi>ℓ</mi><mo>′</mo></msup><mi>d</mi><mo>=</mo><mn>0</mn><mspace width=".5em"/><mo>∀</mo><mi>d</mi><mo>∈</mo><msub><mi>𝒦</mi><mi>E</mi></msub></mtd></mtr><mtr><mtd><mi>q</mi><mo>(</mo><mi>m</mi><mo>)</mo><mo>=</mo><msup><mi>m</mi><mo>′</mo></msup><mi>A</mi><mi>m</mi><mspace width="1em"/><mtext>identified</mtext><mspace width=".8em"/><mo>⇔</mo><mspace width=".8em"/><mi>q</mi><mo>(</mo><mi>m</mi><mo>+</mo><mi>d</mi><mo>)</mo><mo>=</mo><mi>q</mi><mo>(</mo><mi>m</mi><mo>)</mo><mspace width=".5em"/><mo>∀</mo><mi>d</mi><mo>∈</mo><msub><mi>𝒦</mi><mi>E</mi></msub></mtd></mtr></mtable></math>`,
  lowrank: `<math><mrow><msub><mi>Y</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo><msubsup><mi>X</mi><mrow><mi>i</mi><mi>j</mi></mrow><mo>′</mo></msubsup><mi>β</mi><mo>+</mo><msub><mi>α</mi><mi>i</mi></msub><mo>+</mo><msub><mi>ψ</mi><mi>j</mi></msub><mo>+</mo><msubsup><mi>u</mi><mi>i</mi><mo>′</mo></msubsup><mi>Λ</mi><msub><mi>v</mi><mi>j</mi></msub><mo>+</mo><msub><mi>ε</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mspace width="1.5em"/><mi>rank</mi><mo>(</mo><mi>Λ</mi><mo>)</mo><mo>=</mo><mi>r</mi></mrow></math>`,
  rectangles: `<math><mrow><msub><mi>D</mi><mrow><mi>i</mi><msup><mi>i</mi><mo>′</mo></msup><mo>;</mo><mi>j</mi><mi>k</mi></mrow></msub><mo>=</mo><msub><mi>m</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>-</mo><msub><mi>m</mi><mrow><mi>i</mi><mi>k</mi></mrow></msub><mo>-</mo><msub><mi>m</mi><mrow><msup><mi>i</mi><mo>′</mo></msup><mi>j</mi></mrow></msub><mo>+</mo><msub><mi>m</mi><mrow><msup><mi>i</mi><mo>′</mo></msup><mi>k</mi></mrow></msub><mo>=</mo><mo>(</mo><msub><mi>u</mi><mi>i</mi></msub><mo>-</mo><msub><mi>u</mi><msup><mi>i</mi><mo>′</mo></msup></msub><mo>)</mo><mo>(</mo><msub><mi>v</mi><mi>j</mi></msub><mo>-</mo><msub><mi>v</mi><mi>k</mi></msub><mo>)</mo></mrow></math>`,
  functional: `<math><mrow><msub><mi>T</mi><mi>A</mi></msub><mo>(</mo><mi>m</mi><mo>+</mo><mi>Δ</mi><mo>)</mo><mo>=</mo><msub><mi>T</mi><mi>A</mi></msub><mo>(</mo><mi>m</mi><mo>)</mo><mspace width="1em"/><mtext>for every observationally equivalent direction</mtext><mspace width=".6em"/><mi>Δ</mi></mrow></math>`,
  objective: `<math><mrow><munder><mi>min</mi><mrow><mi>α</mi><mo>,</mo><mi>ψ</mi><mo>,</mo><mi>U</mi><mo>,</mo><mi>V</mi><mo>,</mo><mi>Λ</mi></mrow></munder><mspace width=".5em"/><munder><mo>∑</mo><mrow><mo>(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo>)</mo><mo>∈</mo><mi>E</mi></mrow></munder><msub><mi>n</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><msup><mrow><mo>(</mo><msub><mover><mi>Y</mi><mo>¯</mo></mover><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>-</mo><msub><mi>α</mi><mi>i</mi></msub><mo>-</mo><msub><mi>ψ</mi><mi>j</mi></msub><mo>-</mo><msubsup><mi>u</mi><mi>i</mi><mo>′</mo></msubsup><mi>Λ</mi><msub><mi>v</mi><mi>j</mi></msub><mo>)</mo></mrow><mn>2</mn></msup></mrow></math>`,
  bias: `<math><mrow><mi>q</mi><mo>(</mo><mover><mi>m</mi><mo>^</mo></mover><mo>)</mo><mo>-</mo><mi>q</mi><mo>(</mo><mi>m</mi><mo>)</mo><mo>=</mo><mn>2</mn><msup><mi>m</mi><mo>′</mo></msup><mi>A</mi><mi>η</mi><mo>+</mo><msup><mi>η</mi><mo>′</mo></msup><mi>A</mi><mi>η</mi><mo>,</mo><mspace width="1em"/><mover><mi>m</mi><mo>^</mo></mover><mo>=</mo><mi>m</mi><mo>+</mo><mi>η</mi></mrow></math>`,
  asymptotics: `<math><mrow><mi>I</mi><mo>,</mo><mi>J</mi><mo>,</mo><mo>|</mo><mi>E</mi><mo>|</mo><mo>→</mo><mi>∞</mi><mspace width="1.5em"/><mfrac><mi>J</mi><mi>I</mi></mfrac><mo>→</mo><mi>κ</mi><mo>∈</mo><mo>(</mo><mn>0</mn><mo>,</mo><mi>∞</mi><mo>)</mo><mspace width="1.5em"/><msubsup><mover><mi>T</mi><mo>^</mo></mover><mi>A</mi><mtext>bc</mtext></msubsup><mo>-</mo><msub><mi>T</mi><mi>A</mi></msub><mover><mo>→</mo><mi>p</mi></mover><mn>0</mn></mrow></math>`,
};

async function renderMath() {
  await fs.mkdir(MATH_DIR, { recursive: true });
  const out = {};
  for (const [id, markup] of Object.entries(mathDefs)) {
    const htmlFile = path.join(MATH_DIR, `${id}.html`);
    const file = path.join(MATH_DIR, `${id}.png`);
    await fs.writeFile(htmlFile, `<style>html,body{margin:0;width:2400px;height:300px;background:transparent;overflow:hidden}.eq{width:2400px;height:300px;display:flex;align-items:center;justify-content:center;color:#000;font-family:"Cambria Math","STIX Two Math",serif;font-size:68px;white-space:nowrap}math{font-family:"Cambria Math","STIX Two Math",serif}</style><div class="eq">${markup}</div>`);
    out[id] = { file, width: 2400, height: 300 };
  }
  return out;
}

const mathAssets = await renderMath();
const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const C = { bg: "#ffffff", ink: "#000000", muted: "#666666", rule: "#bdbdbd" };

function addText(slide, text, x, y, w, h, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = { fontSize: opts.size ?? 21, bold: opts.bold ?? false, color: opts.color ?? C.ink, alignment: opts.align ?? "left", fontFamily: opts.fontFamily ?? "Aptos" };
  return shape;
}

function addRule(slide, y) {
  slide.shapes.add({ geometry: "rect", position: { left: 72, top: y, width: 1136, height: 1 }, fill: C.rule, line: { style: "solid", fill: "none", width: 0 } });
}

async function addMath(slide, id, x, y, w, h) {
  const a = mathAssets[id];
  const bytes = await fs.readFile(a.file);
  slide.images.add({ blob: bytes, contentType: "image/png", alt: `Typeset equation: ${id}`, fit: "contain", position: { left: x, top: y, width: w, height: h } });
}

function footer(slide, source, n) {
  addText(slide, `Source: ${source}`, 72, 676, 1040, 18, { size: 10.5, color: C.muted });
  addText(slide, `${n}`, 1150, 676, 58, 18, { size: 10.5, color: C.muted, align: "right" });
}

async function argumentSlide({ step, title, equation, body, bridge, source, bodySize = 20, equationHeight = 92 }) {
  const s = deck.slides.add();
  s.background.fill = C.bg;
  addText(s, step, 72, 38, 1080, 20, { size: 12, bold: true, color: C.muted });
  addText(s, title, 72, 72, 1136, 70, { size: 34, bold: true });
  let bodyTop = 170;
  if (equation) {
    const displayHeight = equationHeight + 14;
    await addMath(s, equation, 82, 154, 1116, displayHeight);
    bodyTop = 164 + displayHeight + 18;
  }
  addText(s, body.trim(), 86, bodyTop, 1100, 575 - bodyTop, { size: bodySize });
  addRule(s, 602);
  addText(s, bridge, 86, 615, 1090, 34, { size: 16, color: C.muted });
  footer(s, source, deck.slides.items.length);
  return s;
}

{
  const s = deck.slides.add();
  s.background.fill = C.bg;
  addText(s, "Firm Wage Effects and Assignment Beyond Additivity", 72, 120, 1136, 120, { size: 50, bold: true });
  addText(s, "A technical reading of the LOO note", 76, 270, 900, 38, { size: 26 });
  addText(s, "One argument: define the right wage-schedule functionals, identify them on sparse worker-firm graphs, and estimate them without quadratic plug-in bias.", 76, 390, 1000, 90, { size: 22 });
  footer(s, "LOO note; merged TeX Part II", 1);
}

await argumentSlide({ step: "STEP 1 | WHY THE OBJECT MUST CHANGE", title: "AKM is complete only when every worker faces the same firm ladder", equation: "ladder", equationHeight: 82, body: `AKM summarizes the firm side with a common premium and sorting with a worker-firm covariance. The displayed restriction is exactly what makes those two objects complete: every firm comparison is invariant across workers.\n\nOnce comparative firm advantages vary by worker, the coefficient vector is no longer the economic object. The primitive must be the full pairwise wage schedule.`, bridge: "Therefore the analysis must first declare the wage schedule and the assignment distribution used to average it.", source: "LOO note, pp. 6-10" });

await argumentSlide({ step: "STEP 2 | DECLARE THE PRIMITIVES", title: "Observed matches and the target wage schedule are different objects", equation: "primitives", equationHeight: 76, body: `The target schedule assigns a maintained systematic wage to worker-firm pairs in a declared target set. The observed assignment distribution places mass only on realized edges; the product distribution breaks matching while preserving worker and firm marginals.\n\nThis distinction prevents an estimand from silently depending on missing cells or on a counterfactual assignment measure that was never stated.`, bridge: "With the primitives fixed, the paper can ask two precise questions: how much firms matter and how assignment matters.", source: "LOO note, pp. 9-10, 19" });

await argumentSlide({ step: "STEP 3 | DEFINE THE FOUR TARGETS", title: "Four functionals replace the two AKM firm-side summaries", equation: "estimands", equationHeight: 132, body: `The first pair measures firm wage relevance: average cross-firm dispersion for a worker, and heterogeneity across workers in firm wage gaps. The second pair measures assignment: the change in wage-schedule variance under observed rather than independent matching, and the mean interaction selected by observed matches.\n\nUnder additivity, the gap-heterogeneity and mean-interaction targets vanish, while the remaining two collapse to the familiar AKM objects.`, bridge: "To interpret these four scalars, the schedule needs a unique common-firm component and a unique interaction component.", source: "LOO note, pp. 6-7, 12-17", bodySize: 19 });

await argumentSlide({ step: "STEP 4 | PROJECT THE SCHEDULE", title: "A weighted two-way projection separates common levels from comparative advantage", equation: "decomposition", equationHeight: 120, body: `The worker component is the worker's product-weighted average wage; the firm component is the common firm ladder; the interaction is what remains after both margins are removed. Row and column centering make this representation unique for the declared reference distributions.\n\nThis is an accounting projection of the wage schedule. It is not yet a causal or production decomposition.`, bridge: "The centering restrictions do more than normalize labels: they make the three nonconstant components orthogonal.", source: "LOO note, p. 11; TeX Part II, Proposition 1", bodySize: 19 });

await argumentSlide({ step: "STEP 5 | USE PROPOSITION 1", title: "Uniqueness and orthogonality remove the cross terms", equation: "orthogonality", equationHeight: 76, body: `Suppose two centered decompositions represented the same schedule. Averaging their difference over firms identifies the worker component; averaging over workers identifies the firm component; subtraction then identifies the interaction. This proves uniqueness.\n\nThe same conditional mean-zero restrictions imply the displayed inner products are zero under product weights. Every variance identity that follows uses this orthogonality.`, bridge: "The first payoff is a sharp decomposition of firm wage relevance into a common ladder and heterogeneous firm gaps.", source: "LOO note, p. 11; TeX Part II, Proposition 1 proof" });

await argumentSlide({ step: "STEP 6 | DECOMPOSE FIRM RELEVANCE", title: "Firm wage dispersion equals common-premium dispersion plus half of gap heterogeneity", equation: "firmSplit", equationHeight: 86, body: `For a fixed worker, only the common firm component and the interaction vary across firms. Orthogonality eliminates their covariance after averaging over workers.\n\nA difference between two independently drawn firms doubles the interaction variance, so the gap-heterogeneity statistic is twice the interaction contribution. The ratio of half the gap statistic to total firm relevance is therefore the share that a common firm premium cannot summarize.`, bridge: "The same projected components also reveal why assignment has more than the single AKM sorting channel.", source: "LOO note, pp. 13-14; Proposition 2", bodySize: 19 });

await argumentSlide({ step: "STEP 7 | DECOMPOSE ASSIGNMENT", title: "Nonadditivity opens three assignment channels beyond AKM sorting", equation: "assignment", equationHeight: 132, body: `Observed and product assignments have the same worker and firm marginals, so the marginal variances of the worker and firm components cancel. What remains is covariance between the common components, two covariances involving comparative advantage, and a change in interaction dispersion.\n\nThe first term is AKM-style sorting. The other three distinguish sorting on favorable cells from sorting on a common firm ranking.`, bridge: "A separate mean comparison then isolates the cleanest scalar measure of selection into comparative advantage.", source: "LOO note, pp. 15-17; Proposition 3", bodySize: 18.5 });

await argumentSlide({ step: "STEP 8 | ISOLATE MEAN INTERACTION SELECTION", title: "Breaking assignment leaves only the selected interaction mean", equation: "meanAssignment", equationHeight: 82, body: `Because observed and product assignments share both marginals, the worker and common-firm means are identical under the two measures. Their difference therefore equals the observed mean of the centered interaction.\n\nA positive value says realized matches occupy cells above what the two marginal components predict. It is a wage-schedule statement, not automatically a statement about output, surplus, or welfare.`, bridge: "The nested special cases show exactly which restrictions turn these new objects off.", source: "LOO note, pp. 16-17, 40-41" });

await argumentSlide({ step: "STEP 9 | CHECK THE NESTED MODELS", title: "Additivity, Tukey interactions, and free factors impose distinct restrictions", equation: "cases", equationHeight: 142, body: `Additivity eliminates interaction-based firm relevance and assignment. A centered Tukey term ties comparative advantage to the same worker and firm indices that generate the main effects. A free-factor interaction permits a rank-one surface without forcing that economic interpretation.\n\nThus algebraic rank one does not by itself deliver a low-dimensional parameterization: unrestricted worker and firm factors can still be numerous incidental objects.`, bridge: "These definitions are coherent on a complete schedule; the next issue is whether sparse observed matches identify them.", source: "LOO note, pp. 17-23; Appendix E", bodySize: 18.5 });

await argumentSlide({ step: "STEP 10 | STATE THE SUPPORT PROBLEM", title: "The data reveal observed edges, while the targets generally use unobserved cells", equation: "support", equationHeight: 78, body: `The restriction operator selects the worker-firm cells present in the bipartite graph. Any perturbation in its kernel changes only missing cells and is observationally invisible without further structure.\n\nOrdinary connectedness links worker and firm main effects in an additive model. It does not determine arbitrary cardinal levels or contrasts in missing pair-specific wage cells.`, bridge: "Identification therefore reduces to a precise invariance question: can an invisible perturbation change the target?", source: "LOO note, pp. 19-21" });

await argumentSlide({ step: "STEP 11 | APPLY PROPOSITION 4", title: "A target is identified only when every missing-cell perturbation leaves it unchanged", equation: "identification", equationHeight: 118, body: `For a linear target, the target loading must annihilate every direction hidden by the graph. For a quadratic target, both the linear cross term and the perturbation's own quadratic contribution must vanish over the admissible model class.\n\nThe paper's central firm and assignment statistics generally fail this test on an unrestricted incomplete schedule. With unbounded missing wages, some identified sets are unbounded.`, bridge: "The model must therefore restrict the missing interaction surface, while preserving unrestricted worker and firm main effects.", source: "LOO note, pp. 19-21; Proposition 4 and Corollary 1", bodySize: 18.5 });

await argumentSlide({ step: "STEP 12 | ADD THE MINIMAL STRUCTURE", title: "Low rank is imposed on comparative advantage, not on worker or firm main effects", equation: "lowrank", equationHeight: 82, body: `The additive components remain unrestricted; only the centered interaction is factorized. Setting the interaction matrix to zero nests AKM. Grouped BLM mean layers imply a low-rank double-demeaned interaction after type cells are lifted to workers and firms.\n\nKline-style mover restrictions concern edge differences rather than a complete level schedule, so they are diagnostics implied by this model, not a substitute for its level structure.`, bridge: "Low rank becomes empirically useful because observed rectangles cancel the unrestricted main effects and expose interaction contrasts.", source: "LOO note, pp. 21-23, 41-44; merged TeX Part I", bodySize: 18.5 });

await argumentSlide({ step: "STEP 13 | EXPLOIT RECTANGLES", title: "Double differences identify factor contrasts after main effects cancel", equation: "rectangles", equationHeight: 86, body: `Four observed cells forming a worker-by-firm rectangle eliminate the worker and firm components. Under rank one, the surviving double difference factors into a worker contrast times a firm contrast.\n\nOverlapping rectangles transmit these relative contrasts across the graph. Anchors and normalizations fix the usual factor scale, sign, and location indeterminacies; sparse or rectangle-poor regions remain weakly identified.`, bridge: "But the paper does not need every factor or missing cell: it needs only invariance of the target functional.", source: "LOO note, pp. 25-28, 44", bodySize: 18.5 });

await argumentSlide({ step: "STEP 14 | IDENTIFY THE FUNCTIONAL", title: "Target identification can hold even when full matrix completion fails", equation: "functional", equationHeight: 76, body: `Full completion asks whether every missing wage cell is unique. Functional identification asks the weaker and economically relevant question displayed above.\n\nLocally, unidentified tangent directions must lie in null directions of the target's derivative and curvature. This reframes the graph condition around the scalar being reported rather than around arbitrary factor normalizations.`, bridge: "Once the target is identified, estimation should be designed directly for that target rather than only for fitted cell means.", source: "LOO note, p. 26; TeX Part II, low-rank identification" });

await argumentSlide({ step: "STEP 15 | ESTIMATE THE SCHEDULE", title: "Weighted low-rank least squares supplies fitted cells, but the estimands are quadratic operators", equation: "objective", equationHeight: 94, body: `Cell means are fit on observed edges using exposure weights and a fixed interaction rank. The four economic targets are then linear or quadratic functionals of the fitted schedule under the declared reference measures.\n\nThis distinction matters: an estimator with acceptable prediction error can still have material bias for a quadratic variance or covariance target.`, bridge: "The source of that bias appears immediately when the fitted schedule is substituted into a quadratic form.", source: "LOO note, pp. 28-31; Appendix J", bodySize: 19 });

await argumentSlide({ step: "STEP 16 | CORRECT QUADRATIC PLUG-IN BIAS", title: "Estimation noise contributes its own positive quadratic term", equation: "bias", equationHeight: 84, body: `Sparse graphs create many noisy local worker, firm, and factor estimates. Even if the first-order term averages out, the noise quadratic need not vanish and can mimic genuine dispersion. This is the nonadditive analogue of limited-mobility bias.\n\nA leave-worker-out or cavity construction estimates the relevant firm-side object without worker i and evaluates that worker against an object less mechanically correlated with the worker's noise.`, bridge: "A complete theorem must show that this correction works for aggregate targets under a growing sparse graph.", source: "LOO note, pp. 28-34, 46-47", bodySize: 18.5 });

await argumentSlide({ step: "STEP 17 | FORMULATE THE INFERENCE TARGET", title: "Consistency is required for aggregate functionals, not for every latent effect", equation: "asymptotics", equationHeight: 84, body: `The relevant asymptotic sequence grows workers, firms, and observed edges together. Individual factors may remain noisy; the bias-corrected aggregate can still be consistent if leverage is controlled and support propagates enough information for the target.\n\nWeak anchors, scarce rectangles, and high-leverage cells are therefore part of the inferential assumptions, not merely descriptive graph statistics. Rank uncertainty must also enter sensitivity analysis.`, bridge: "The simulation and empirical design should directly stress these theorem conditions and report the four economic targets.", source: "LOO note, pp. 31-37", bodySize: 18.5 });

await argumentSlide({ step: "STEP 18 | MAKE THE DESIGNS DIAGNOSTIC", title: "Monte Carlo and empirical work should test recoverability, not only fit", body: `Monte Carlo varies three ingredients jointly: the true interaction structure, assignment on common versus match-specific components, and graph support ranging from rectangle-rich to weakly anchored. It reports bias, RMSE, and coverage for each economic functional under rank misspecification.\n\nThe empirical application reports the firm-dispersion split, all four assignment channels, and support diagnostics including edge density, rectangle density, anchor strength, and target leverage.`, bridge: "These reporting choices also clarify how the contribution differs from adjacent literatures that restrict different objects.", source: "LOO note, pp. 37-41", bodySize: 20 });

await argumentSlide({ step: "STEP 19 | LOCATE THE CONTRIBUTION", title: "The adjacent literatures differ in the object they restrict", body: `Kline and KSS characterize restrictions and bias for additive mover-edge effects; this note studies level functionals of a pairwise wage schedule. BLM's grouped mean layer motivates low-rank double demeaning, but complete BLM also models type probabilities and wage distributions.\n\nEvent-study diagnostics can validate restrictions on mover dynamics without eliminating nonadditive wage levels. Search and Roy models explain selected matches, but wage-schedule sorting alone does not identify production complementarity or surplus.`, bridge: "Taken together, these comparisons leave a precise first-paper contribution and a precise remaining theorem agenda.", source: "LOO note, pp. 41-44; merged TeX Parts I-II", bodySize: 19 });

await argumentSlide({ step: "CONCLUSION | WHAT THE PAPER ADDS", title: "The paper replaces coefficient interpretation with identified wage-schedule functionals", equation: "firmSplit", equationHeight: 84, body: `The conceptual result is the displayed identity: total firm wage relevance separates into a common firm ladder and worker-specific firm-gap dispersion. The assignment decomposition then shows exactly how matching can operate through both components.\n\nThe econometric result is a roadmap: unrestricted sparse support cannot identify the cardinal quadratic targets; low-rank interaction structure can make the relevant functionals invariant; leave-out methods must then remove graph-induced quadratic bias. The remaining burden is a target-specific identification and inference theorem.`, bridge: "The proofs and intermediate algebra are developed in the consolidated TeX notes; this deck preserves the argument and the estimands.", source: "LOO note, teaching summary; merged TeX Part II", bodySize: 18.5 });

await fs.mkdir(PREVIEW_DIR, { recursive: true });
for (const [i, s] of deck.slides.items.entries()) {
  await writeBlob(path.join(PREVIEW_DIR, `slide-${String(i + 1).padStart(2, "0")}.png`), await deck.export({ slide: s, format: "png", scale: 1 }));
  const layout = await s.export({ format: "layout" });
  await fs.writeFile(path.join(PREVIEW_DIR, `slide-${String(i + 1).padStart(2, "0")}.layout.json`), await layout.text());
}

await writeBlob(path.join(PREVIEW_DIR, "montage.webp"), await deck.export({ format: "webp", montage: true, scale: 1 }));
await fs.mkdir(path.dirname(OUT), { recursive: true });
await (await PresentationFile.exportPptx(deck)).save(OUT);
console.log(OUT);
