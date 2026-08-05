# Handoff: LOO/BLM/Kline/BS/Econ-AI notes and slides

Date: 2026-07-24  
Workspace: `C:\Users\A8327\OneDrive\Documents\OI`

## 1. User constraints and preferences

- Do not edit files from earlier chats.
- All substantive TeX produced in this chat was consolidated into
  `todo_chat_merged_notes.tex`.
- Later, the user explicitly requested the BLM-to-project proof and dummy
  example as a **separate** standalone TeX note. That authorized exception is
  `blm_to_project_linear_proof_dummy_example.tex`.
- Slides are only for the attached LOO note, not for every paper in the merged
  notes.
- Slide style must be close to default Microsoft PowerPoint: little color,
  little formatting, few split text boxes, technical enough for a professor,
  readable mathematics, and a connected argument across slides.
- Mathematical explanations should be linear, self-contained, and
  undergraduate-readable. Do not skip selector identities, centering steps,
  rank arguments, or substitutions.
- Keep notation stable. In the standalone BLM proof:
  - `\alpha_i^B` is the BLM worker type.
  - `\alpha_i` is the project additive worker effect.
  - `M` is the worker-type/firm-class interaction matrix.
  - `H` is the individual worker/firm interaction matrix.

## 2. Primary artifacts

### Consolidated notes

`C:\Users\A8327\OneDrive\Documents\OI\todo_chat_merged_notes.tex`

This is the main TeX artifact from the chat. It includes:

- BLM static model, corrected double-demeaning proof, weighted centering, and
  lifting to workers and firms.
- Kline edge model, cycle restrictions, distinction between firm-mobility
  cycles and worker-firm rectangles, anchor/gauge/propagation discussion, and
  the limits of identification.
- Extended LOO-v2 note with proofs, notation, flow, and semistructural
  comparison.
- Econ-AI proposal, PPI, multi-task PPI, empirical Bayes, and hybrid
  recommendations.
- Borovickova-Shimer (2020) sorting note and its connection to the LOO-v2
  primitives.

The latest standalone BLM proof refinements were not merged back into this
file because the user requested them as a separate note.

### Standalone BLM proof

`C:\Users\A8327\OneDrive\Documents\OI\blm_to_project_linear_proof_dummy_example.tex`

This file contains:

- A bare-bones, hand-typed presentation.
- A notation table covering all indices, histories, coefficients, maps,
  weights, means, matrices, and project factors.
- A complete forward proof from BLM's static regression to the project model.
- The full selector proof of `H=ZMW'`.
- The noncircular rank proof: derive `M` first, then lift it to `H`.
- The Stefan/Michael and GS/Meta/BoA numerical example.
- A dependency audit and a scope statement.

Latest style changes:

- Removed boxes.
- Removed booktabs and decorative rules.
- Replaced large step headings with compact paragraph headings.
- Added the iterated-expectations step from BLM's full-history mean-zero
  restriction to the current-cell conditional mean.

### Slides

Current final deck:

`C:\Users\A8327\OneDrive\Documents\OI\outputs\loo_slides.pptx`

Other older slide artifacts exist in `outputs`; do not assume they are the
latest:

- `outputs\looslides.pptx`
- `outputs\loo_prompt3_summary_slide.pptx`

The rendered slide images and source work are under:

- `outputs\loo_connected_render_final`
- `work\presentations\loo_comprehensive_deck`

## 3. Source files used

- BLM:
  `papers\bonhomme_lamadon_manresa_2019_distributional_framework.pdf`
- Kline:
  `papers\kline_2024_review_akm_firm_wage_effects_w33084.pdf`
- Borovickova-Shimer (2020):
  `C:\Users\A8327\Downloads\borovickova shimer 2020 high wage workers work for high wage firms nonparametric_sorting.pdf`
- Highlighted meeting note:
  `C:\Users\A8327\Downloads\7_22_Meeting_highlighted.pdf`
- Original LOO note:
  `C:\Users\A8327\Downloads\loo_full_note_v2.pdf`
- Econ-AI proposal:
  `openai_exchange_proposal_v4_for_ext.pdf`
- PPI:
  `prediction_powered_inference_2301_09633.pdf`
- Multi-task PPI:
  `multi_task_ppi_2605_29249.pdf`
- Prior context files:
  - `handoff_2026-07-22_search_lowrank_memo.md`
  - `semistructural_project_handoff.md`

## 4. Correct BLM-to-project logic

The displayed BLM static regression is

```tex
Y_{ij}
=X_{ij}'c+a_{h(j)}+b_{h(j)}\alpha_i^B+\varepsilon_{ij}.
```

After using the mean-zero restriction,

```tex
m_{ij}
=X_{ij}'c+a_{h(j)}+b_{h(j)}\alpha_i^B.
```

For worker type `k` and firm class `\ell`,

```tex
\mu_{k\ell}=a_\ell+b_\ell\alpha_k^B.
```

With worker-type weights `p_k` and firm-class weights `q_\ell`,

```tex
\bar\alpha=\sum_k p_k\alpha_k^B,\qquad
\bar a=\sum_\ell q_\ell a_\ell,\qquad
\bar b=\sum_\ell q_\ell b_\ell.
```

The row, column, and grand means are

```tex
\mu_{k\cdot}=\bar a+\bar b\alpha_k^B,
```

```tex
\mu_{\cdot\ell}=a_\ell+b_\ell\bar\alpha,
```

```tex
\mu_{\cdot\cdot}=\bar a+\bar\alpha\bar b.
```

Double demeaning gives

```tex
M_{k\ell}
=\mu_{k\ell}-\mu_{k\cdot}-\mu_{\cdot\ell}+\mu_{\cdot\cdot}
=(\alpha_k^B-\bar\alpha)(b_\ell-\bar b).
```

Therefore

```tex
M=
\begin{pmatrix}
\alpha_1^B-\bar\alpha\\
\vdots\\
\alpha_K^B-\bar\alpha
\end{pmatrix}
\begin{pmatrix}
b_1-\bar b&\cdots&b_L-\bar b
\end{pmatrix},
```

so `rank(M)<=1`. This conclusion is established before `H` is introduced.

Weighted centering also gives

```tex
p'M=0,\qquad Mq=0,
```

so

```tex
\rank(M)\leq\min\{1,K-1,L-1\}.
```

Define membership matrices

```tex
Z_{ik}=1\{g(i)=k\},\qquad
W_{j\ell}=1\{h(j)=\ell\}.
```

Define the individual interaction by

```tex
H_{ij}=M_{g(i),h(j)}.
```

The full selector calculation is

```tex
(ZMW')_{ij}
=\sum_{\ell=1}^L\sum_{k=1}^K
Z_{ik}M_{k\ell}W_{j\ell}
=M_{g(i),h(j)}
=H_{ij}.
```

The middle equality requires the one-hot argument: only `k=g(i)` and
`\ell=h(j)` have nonzero indicators.

The previously established project interaction factors are

```tex
u_i=\alpha_i^B-\bar\alpha,\qquad
v_j=b_{h(j)}-\bar b.
```

Hence

```tex
H_{ij}
=M_{g(i),h(j)}
=u_iv_j,
```

and therefore

```tex
H=ZMW'=uv'.
```

The additive project components are

```tex
\mu=\bar a+\bar\alpha\bar b,
```

```tex
\alpha_i=\bar b(\alpha_i^B-\bar\alpha),
```

```tex
\psi_j
=(a_{h(j)}-\bar a)
+\bar\alpha(b_{h(j)}-\bar b).
```

With `\delta=c` and `\Lambda=[1]`, the project model is

```tex
m_{ij}
=X_{ij}'\delta+\mu+\alpha_i+\psi_j+u_i'\Lambda v_j.
```

Scope warning:

- The exact rank-one result is for BLM's displayed interactive regression.
- A general finite `K x L` BLM conditional-mean table yields
  `rank(M)<=min(K-1,L-1)` after double demeaning, but need not be rank one.
- The project model nests BLM's static conditional-mean layer, not the full
  conditional earnings distribution, assignment process, or dynamic model.

## 5. Numerical dummy example

Workers:

- Stefan: worker type 1, `\alpha_{\text{Stefan}}^B=1`.
- Michael: worker type 2, `\alpha_{\text{Michael}}^B=-1`.

Firms:

- GS: firm class 1.
- Meta: firm class 2.
- BoA: firm class 1.

Parameters:

```tex
a_1=4,\quad a_2=5,\quad b_1=2,\quad b_2=-1,
```

```tex
p=(1/2,1/2)',\qquad q=(2/3,1/3)'.
```

Means:

```tex
\bar\alpha=0,\qquad
\bar a=13/3,\qquad
\bar b=1.
```

Type-class wage table:

```tex
(\mu_{k\ell})
=
\begin{pmatrix}
6&4\\
2&6
\end{pmatrix}.
```

Project components:

```tex
\mu=13/3,
```

```tex
\alpha=
\begin{pmatrix}1\\-1\end{pmatrix},
\qquad
\psi=
\begin{pmatrix}-1/3\\2/3\\-1/3\end{pmatrix},
```

```tex
M=
\begin{pmatrix}
1&-2\\
-1&2
\end{pmatrix},
\qquad
u=
\begin{pmatrix}1\\-1\end{pmatrix},
\qquad
v=
\begin{pmatrix}1\\-2\\1\end{pmatrix}.
```

Membership matrices:

```tex
Z=
\begin{pmatrix}
1&0\\
0&1
\end{pmatrix},
\qquad
W=
\begin{pmatrix}
1&0\\
0&1\\
1&0
\end{pmatrix}.
```

Individual interaction and wage matrices:

```tex
H=ZMW'=uv'
=
\begin{pmatrix}
1&-2&1\\
-1&2&-1
\end{pmatrix},
```

```tex
(m_{ij})
=
\begin{pmatrix}
6&4&6\\
2&6&2
\end{pmatrix}.
```

Rows are Stefan and Michael; columns are GS, Meta, and BoA.

## 6. Borovickova-Shimer (2020): measurement-error interpretation

This latest explanation currently exists in the chat/handoff, not as a new
section in the standalone BLM proof.

The estimator has repeated-measurement-error logic:

```tex
w^w_{im}=\lambda_i+\varepsilon^w_{im}.
```

One squared wage is contaminated:

```tex
E[(w^w_{im})^2\mid\lambda_i]
=\lambda_i^2+\operatorname{Var}(\varepsilon^w_{im}\mid\lambda_i).
```

For two distinct conditionally independent observations,

```tex
E[w^w_{im}w^w_{im'}\mid\lambda_i]=\lambda_i^2,
\qquad m\neq m'.
```

This motivates

```tex
\widehat{\lambda_i^2}
=
\frac{1}{M_i(M_i-1)}
\sum_m\sum_{m'\neq m}w^w_{im}w^w_{im'}.
```

The firm-side argument is identical for `\mu_j^2`.

For a focal match `m` between worker `i` and firm `j`, BS estimate the type
product using the worker's and firm's other matches:

```tex
\widehat c_{im}
=
\left[
\frac{1}{M_i-1}\sum_{m'\neq m}w^w_{im'}
\right]
\left[
\frac{1}{N_j-1}\sum_{n'\neq n(i,m)}w^f_{jn'}
\right].
```

Under their cross-side independence assumption,

```tex
E[\widehat c_{im}\mid\lambda_i,\mu_j]=\lambda_i\mu_j.
```

They leave out the focal match because its wage error may be correlated with
the partner's type or with a shared match shock. At least two observations per
worker and firm ensure that one other observation remains after the focal
match is excluded.

Best characterization:

- It is a repeated-measurement latent-variable estimator.
- It is analogous to measurement-error or reliability logic.
- It is not primarily a classical errors-in-variables regression because
  there is no regression coefficient whose attenuation is being corrected.
- Conditional independence is essential. Repeated annual wages in the same
  job may fail it, which is why BS use distinct matches and, in preferred
  specifications, jobs separated by unemployment.

Source: Borovickova and Shimer (2020), Section 4.2, equations (18)--(25).

## 7. Other major conclusions already in the merged notes

### BLM

- Double demeaning is the bridge from a finite type-class wage table to the
  project's low-rank interaction restriction.
- The phrase "the missing point" refers to an omission in an earlier project
  note, not an omission by BLM.
- If every type and class is represented, lifting by membership matrices does
  not change rank: `rank(ZMW')=rank(M)`.

### Kline

- Kline's directed firm-mobility cycles concern path independence of edge wage
  changes.
- Worker-firm rectangles concern alternating sums of wage levels on the
  bipartite support graph.
- A rich projected firm graph need not provide worker-firm rectangles.
- One rectangle identifies only a product of factor contrasts, not all factor
  coordinates.
- Full rank-one identification needs an anchor subgraph, gauge choices,
  propagation, and nonzero denominators.

### LOO-v2 dispersion

Under the product reference measure:

```tex
Q_F=\operatorname{Var}(b_J)+Q_h,
\qquad
H_F=2Q_h,
```

so

```tex
Q_F=\operatorname{Var}(b_J)+\frac12H_F.
```

The proof in the merged notes includes the full finite-weight expansion and
identifies exactly where row centering, column centering, and independent firm
draws are used.

### PPI versus empirical Bayes

- PPI is a design-based control-variate correction for scalar or
  low-dimensional aggregates.
- Empirical Bayes is model-based shrinkage for a high-dimensional latent
  vector.
- Recommended hybrid: use cross-fitted EB/multi-task predictions inside a PPI
  residual correction for headline aggregates, while using EB posteriors for
  cell estimates, ranks, dispersion, and top-k claims.

### Borovickova-Shimer structural nesting

- The BS sorting statistic is a functional of the selected wage schedule and
  observed assignment distribution.
- In their discrete-choice laboratory,

```tex
w(x,y)=x-\frac{(x-y)^2}{a}
```

has double-centered interaction

```tex
h(x,y)=\frac{2}{a}(x-\bar x)(y-\bar y),
```

which is exactly rank one under nondegenerate worker and firm distributions.
- The wage layer is exactly nested by the project rank-one model.
- Full structural nesting additionally requires the logit/random-utility
  assignment layer and the firm-offer distribution.

## 8. Validation status

### `todo_chat_merged_notes.tex`

Previously checked:

- Balanced braces.
- Matching `\begin`/`\end` environments.
- No duplicate equation tags.
- No TODO/FIXME/TBD placeholders.

### `blm_to_project_linear_proof_dummy_example.tex`

Latest checks:

- Bare-bones presentation: no `\boxed`, booktabs rules, or large subsection
  step headings.
- Balanced braces and matching environments.
- 105 opening and 105 closing standalone display delimiters.
- No duplicate equation tags.
- 60 required notation entries present in the table.
- Numerical checks:
  - `rank(M)=1`.
  - `rank(H)=1`.
  - `p'M=0` and `Mq=0`.
  - `H=ZMW'=uv'`.
  - The project reconstruction equals the BLM wage matrix exactly.

No LaTeX engine (`pdflatex`, `xelatex`, `lualatex`, `latexmk`, or `tectonic`)
is installed or available on `PATH`, so neither TeX file was compiled in the
latest pass.

## 9. Recommended next actions

1. If the user asks to incorporate the measurement-error interpretation,
   add it to `todo_chat_merged_notes.tex` or create a separate BS estimator
   note only after confirming the desired destination.
2. If a LaTeX engine becomes available, compile both TeX files and visually
   inspect longtable page breaks and equation overflow.
3. Treat `outputs\loo_slides.pptx` as the current deck unless the user
   explicitly identifies another slide file.
4. Preserve the noncircular proof order:
   derive `M`, prove its rank, define/lift `H`, then derive `H=uv'`.

