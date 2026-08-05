import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

const workspace = "C:\\Users\\A8327\\OneDrive\\Documents\\OI\\work\\presentations\\prompt3\\tmp";
const outputDir = "C:\\Users\\A8327\\OneDrive\\Documents\\OI\\outputs";
await fs.mkdir(outputDir, { recursive: true });

const deck = Presentation.create({
  slideSize: { width: 1280, height: 720 },
});

const slide = deck.slides.add();
slide.background.fill = "white";

function textbox(name, text, position, style) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name,
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = style;
  return shape;
}

function rule(name, left, top, width) {
  slide.shapes.add({
    geometry: "rect",
    name,
    position: { left, top, width, height: 2 },
    fill: "111111",
    line: { style: "solid", fill: "111111", width: 0 },
  });
}

textbox(
  "title",
  "Firm Wage Effects Beyond Additivity",
  { left: 62, top: 42, width: 1040, height: 64 },
  { fontSize: 50, bold: true, color: "111111" },
);

textbox(
  "subtitle",
  "Keep AKM's questions; replace scalar firm effects with invariant functionals of the pairwise wage schedule m_ij.",
  { left: 64, top: 112, width: 1060, height: 42 },
  { fontSize: 20, color: "333333" },
);

rule("top-rule", 64, 168, 1152);

textbox(
  "left-heading",
  "Objects",
  { left: 72, top: 198, width: 420, height: 42 },
  { fontSize: 30, bold: true, color: "111111" },
);

textbox(
  "right-heading",
  "Lecture Takeaways",
  { left: 680, top: 198, width: 480, height: 42 },
  { fontSize: 30, bold: true, color: "111111" },
);

const objectLines = [
  "Schedule:  m_ij = mu + a_i + b_j + h_ij",
  "Firm dispersion:  Q_F = E_i Var_j(m_ij)",
  "Contrast heterogeneity:  H_F = E_jk Var_i(m_ik - m_ij)",
  "Assignment:  C_assign^w = 1/2[Var_obs(m) - Var_prod(m)]",
].join("\n\n");

textbox(
  "objects-body",
  objectLines,
  { left: 76, top: 252, width: 520, height: 220 },
  { fontSize: 21, color: "111111" },
);

const takeawayLines = [
  "AKM reduction:  (Q_F, H_F, C_assign^w) = (Var(psi_j), 0, Cov(alpha_i, psi_j))",
  "Key identity:  Q_F = Var(b_j) + 1/2 H_F",
  "Assignment beyond AKM: common assortativity + worker-interaction + firm-interaction + concentration",
].join("\n\n");

textbox(
  "takeaways-body",
  takeawayLines,
  { left: 684, top: 252, width: 520, height: 222 },
  { fontSize: 21, color: "111111" },
);

rule("middle-rule", 64, 504, 1152);

textbox(
  "board-flow-heading",
  "Board Flow",
  { left: 72, top: 532, width: 220, height: 36 },
  { fontSize: 26, bold: true, color: "111111" },
);

textbox(
  "board-flow",
  "Decomposition -> Proposition 2 -> Proposition 3 -> sparse-graph nonidentification -> low-rank completion and leave-out inference",
  { left: 72, top: 578, width: 1110, height: 58 },
  { fontSize: 22, color: "111111" },
);

textbox(
  "footer",
  "Interpretation boundary: wage schedule first; causal effects, surplus, and welfare require extra assumptions.",
  { left: 72, top: 660, width: 1100, height: 28 },
  { fontSize: 16, color: "555555" },
);

const png = await deck.export({ slide, format: "png", scale: 2 });
await writeBlob(path.join(workspace, "loo_summary_slide.png"), png);

const layout = await slide.export({ format: "layout" });
await fs.writeFile(path.join(workspace, "loo_summary_slide.layout.json"), await layout.text());

const inspect = await deck.inspect({
  kind: "slide,textbox,shape,layout",
  maxChars: 8000,
});
await fs.writeFile(path.join(workspace, "loo_summary_slide.inspect.ndjson"), inspect.ndjson);

const pptx = await PresentationFile.exportPptx(deck);
await pptx.save(path.join(outputDir, "loo_prompt3_summary_slide.pptx"));
