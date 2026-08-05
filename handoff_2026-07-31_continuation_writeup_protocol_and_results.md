# Handoff: write-up protocol and project results

Date: 2026-07-31  
Workspace: `C:\Users\A8327\OneDrive\Documents\OI`

## 1. Purpose of this handoff

This conversation continued earlier work on the AI productivity project. The
latest request was to create a **new, separate, self-contained lecture-style
write-up** covering:

1. the user's AI-productivity proposal;
2. Angelopoulos et al., *Prediction-Powered Inference*,
   arXiv:2301.09633; and
3. Emmenegger, Stahler, and Podimata,
   *Prediction-Powered Inference Across Many Tasks for AI Evaluation &
   Social Science Research*, arXiv:2605.29249.

The user then required the notation in the new write-up to match the earlier
AI-project note exactly. That reconciliation has been completed.

This handoff records:

- the standing instructions for how every future write-up must be constructed;
- the notation authority and reserved symbols;
- the files created and source files used;
- the substantive conclusions reached in this conversation;
- the mathematical qualifications that must not be lost; and
- the remaining technical/production caveats.

## 2. Standing instruction: how to create a write-up

These instructions are persistent. Apply them to every future note unless the
user explicitly overrides them.

### 2.1 Start from the existing project record

1. Search the workspace for files containing `handoff` or `continuation`.
2. Read the relevant handoff files before drafting.
3. Identify the existing note whose notation governs the new task.
4. Read the source documents themselves. Do not rely only on an earlier
   summary.
5. If the user requests a separate write-up, create a new file. Do not
   overwrite or silently merge it into an earlier note.

### 2.2 Use one notation system

1. Keep notation identical across handoffs, notes, and revisions.
2. Do not replace an established symbol merely because a source paper uses a
   different symbol.
3. If a source paper differs, place an explicit source-to-project notation
   conversion immediately after the notation table.
4. Do not reuse one symbol for two different objects.
5. If a collision is unavoidable, introduce one new symbol, explain why, and
   record the mapping.
6. Put a notation table at the beginning of every substantive section.
7. Include **every variable used in that section**:
   - indices and index sets;
   - population and sample sizes;
   - observables and latent variables;
   - parameters and estimands;
   - estimators and confidence sets;
   - errors and variances;
   - tuning parameters;
   - auxiliary functions;
   - source-paper aliases that appear in the text;
   - constants appearing in stated expansions or theorems.
8. After revision, search for obsolete aliases and verify that none remain.

### 2.3 Write as a linear whiteboard lecture

The order should be:

1. define the economic or scientific question;
2. define every object;
3. state the estimand;
4. state what is observed;
5. state what is random and what is fixed;
6. state the assumptions;
7. define the estimator;
8. derive the estimator or identification result one equality at a time;
9. state the guarantee;
10. state what is not identified or not proved;
11. connect the result to the project.

The reader should be able to copy the argument onto a whiteboard in the order
presented. Do not require the reader to look ahead for a definition or return
to an earlier section to supply a missing step.

### 2.4 Make every proof explicit

1. Do not skip conditioning, expectation, covariance, substitution, centering,
   selector, rank, or finite-population-correction steps.
2. State exactly which equality uses which assumption.
3. Distinguish:
   - estimand from estimator;
   - point estimator from confidence set;
   - identification from estimation;
   - exact finite-sample validity from asymptotic validity;
   - design-based validity from model-based extrapolation;
   - a theorem proved in a source from a claim asserted in a proposal;
   - a source result from a derivation made in the note;
   - a fixed finite population from a superpopulation.
4. Check dimensions, denominators, signs, conditioning sets, independence
   requirements, and whether a fitted object is fixed relative to the sampling
   design.
5. If a proof needs an assumption not stated in the source, add the assumption
   explicitly and say that it is additional.

### 2.5 Use citations as part of the argument

1. Cite each sourced claim near the sentence or equation it supports.
2. Give exact PDF page or paper section when possible.
3. Mark algebra supplied by the note as `derived`.
4. Mark unproved statements from the proposal as `proposal claim`.
5. Do not present a simulation calibration as a theorem.
6. Do not claim that a paper proves an extension it only mentions as possible.
7. Quote verbatim only when exact wording matters; otherwise paraphrase and
   cite.

### 2.6 Style

1. The note must be understandable to an undergraduate.
2. The mathematics and qualifications must remain suitable for a PhD adviser.
3. Use ordinary lecture language, not unexplained methodological jargon.
4. Delete auxiliary sentences that do not define, derive, qualify, or connect
   a result.
5. Prefer a short displayed derivation to a dense prose description.
6. Use headings only to preserve the linear argument.
7. Do not use praise, decorative boxes, or excessive formatting.
8. Concision must come from removing repetition, not from skipping steps.

### 2.7 Final quality-control checklist

Before delivering a write-up, verify all of the following:

- Every variable is in the notation table at the beginning of its section.
- No symbol changes meaning across sections without an explicit mapping.
- Every proof proceeds linearly.
- Every nontrivial equality is justified.
- Estimand, estimator, and inferential object are separate.
- Observables and latent objects are separate.
- Exact, asymptotic, design-based, and model-based claims are labeled
  correctly.
- Every source claim has a precise citation.
- Derived statements are labeled as derived.
- Limitations and unsupported extrapolations are stated.
- The answer to the adviser-level question is explicit, not left implicit in
  the derivation.
- Existing user files were not overwritten unless specifically authorized.
- LaTeX braces and environments balance.
- Compile and inspect the output when a compiler is available.
- If compilation is unavailable, report that limitation rather than claiming
  visual verification.

## 3. Primary artifact from this conversation

### New standalone note

`C:\Users\A8327\OneDrive\Documents\OI\ai_project_three_papers_whiteboard_note.tex`

This file is separate from all prior notes. It is approximately 4,390
whitespace-delimited source words and contains:

1. a source convention;
2. a complete section on the project proposal;
3. a complete section on ordinary PPI;
4. a complete section on multi-task PPI;
5. a final section giving one coherent design for the AI productivity project;
6. notation tables at the start of all four substantive sections;
7. source-to-project notation mappings;
8. linear derivations of the main identification and variance results; and
9. a method-by-question conclusion table.

The earlier note was not modified:

`C:\Users\A8327\OneDrive\Documents\OI\todo_chat_merged_notes.tex`

Its AI-project material begins in Part III and remains useful as historical
context, but the new standalone file is the latest self-contained treatment.

### Earlier relevant handoff

`C:\Users\A8327\OneDrive\Documents\OI\handoff_2026-07-24_todo_chat_blm_bs_notes.md`

That handoff covers the earlier BLM, Kline, LOO, BS, and Econ-AI work. Read it
if the next task connects the AI project back to those models.

## 4. Source files used

- User's proposal:
  `C:\Users\A8327\OneDrive\Documents\OI\openai_exchange_proposal_v4_for_ext.pdf`
- Ordinary PPI:
  `C:\Users\A8327\OneDrive\Documents\OI\prediction_powered_inference_2301_09633.pdf`
- Multi-task PPI:
  `C:\Users\A8327\OneDrive\Documents\OI\multi_task_ppi_2605_29249.pdf`
- Earlier combined note:
  `C:\Users\A8327\OneDrive\Documents\OI\todo_chat_merged_notes.tex`

The two arXiv sources are:

- <https://arxiv.org/pdf/2301.09633>
- <https://arxiv.org/pdf/2605.29249>

## 5. Notation authority

The new standalone note follows the notation already used in the earlier
AI-project note, not the notation of the proposal PDF or MT-PPI PDF when those
conflict.

### 5.1 Project proposal

Use:

- `j=1,...,J`: occupation-task cell;
- `\theta_j`: true cell gain;
- `\thetav=(\theta_1,...,\theta_J)'`: full latent map;
- `p_j`: economic weight;
- `\bar\theta=\sum_j p_j\theta_j`: aggregate gain;
- `p_j^0,p_j^A`: pre-AI and actual-use basket weights;
- `\bar\theta^0,\bar\theta^A`: the two basket means;
- `S_j`: broad score;
- `R_j`: validation indicator;
- `\pi_j`: validation inclusion probability;
- `m`: number of validated cells;
- `\widehat v_j`: randomized validation estimate;
- `s_j^2`: known validation-estimate variance;
- `\tau^2`: latent cross-cell variance;
- `b`: score-channel slope/compression parameter;
- `\eta_j`: score error;
- `\varepsilon_j`: validation error;
- `\sigma_S^2`: score-error variance;
- `\lambda=b^2\tau^2/(b^2\tau^2+\sigma_S^2)`: score reliability ratio;
- `\psi=(\mu,\tau^2,a,b,\sigma_S^2)`: channel parameter vector;
- `m_j,\omega^2`: posterior mean and variance given the score;
- `P_j,m_j^{\mathrm{val}}`: posterior precision and posterior mean for a
  validated cell;
- `g(S_j)`: calibrated score prediction.

The proposal PDF instead writes
`\sigma_\theta^2,\beta,\nu_j,\widehat\theta_j^v`. These are source aliases for
`\tau^2,b,\eta_j,\widehat v_j`. Do not use the source aliases as the working
notation in project notes.

### 5.2 Ordinary PPI

Use:

- `(X_i,Y_i)`: labeled observation;
- `(\widetilde X_i,\widetilde Y_i)`: unlabeled observation with unobserved
  outcome;
- `n,N`: labeled and unlabeled sample sizes;
- `f`: prediction rule;
- `\theta^\star`: superpopulation estimand;
- `\widehat\theta_{\mathrm{PP}}`: ordinary PPI point estimator;
- `\Delta_\theta`: rectifier;
- `R_\delta(\theta)`: rectifier confidence set;
- `T_{\alpha-\delta}(\theta)`: imputed-gradient confidence set;
- `C_{\mathrm{PP}}`: prediction-powered confidence set;
- `L`: labeled subset of a fixed population;
- `\theta_N=N^{-1}\sum_iY_i`: fixed-population mean;
- `\widehat\theta_{\mathrm{PP},N}`: fixed-population PPI estimator.

### 5.3 Multi-task PPI

Use:

- `t=1,...,T`: task;
- `\theta^{(t)}`: task mean;
- `Y_i^{(t)}`: task-specific ground truth;
- `\widehat Y_i^{(t)}`: task-specific proxy;
- `L^{(t)}`: task-specific labeled subset;
- `s^{(-t)}`: recalibrator trained on other tasks;
- `\gamma_t`: task-specific power-tuning coefficient;
- `\widehat\gamma_t`: estimated power coefficient;
- `u_i^{(t)}`: AREPPI adaptive proxy;
- `S_Y^2,S_s^2,S_{Ys}`: finite-population outcome variance, surrogate
  variance, and covariance;
- `\rho_N^2(Y,s)`: squared finite-population correlation;
- `\kappa_t`: AREPPI local/global mixing weight.

MT-PPI itself calls the power coefficient `\lambda` and the adaptive mixing
weight `\gamma`. The project notation reserves `\lambda` for the proposal's
score reliability ratio and `\gamma_t` for power tuning. Therefore the new
note uses `\kappa_t` for the AREPPI mixing weight. Do not undo this conversion.

## 6. Main substantive results

### 6.1 The proposal's estimands

The primitive object is the full map

```tex
\thetav=(\theta_1,\ldots,\theta_J)'.
```

The aggregate is

```tex
\bar\theta=\sum_{j=1}^Jp_j\theta_j.
```

The two task-basket aggregates are

```tex
\bar\theta^0=\sum_{j=1}^Jp_j^0\theta_j,
\qquad
\bar\theta^A=\sum_{j=1}^Jp_j^A\theta_j.
```

Recovering the full map is sufficient for every summary, but it is not
necessary for the aggregate mean. Random validation can identify an aggregate
directly without identifying every `\theta_j`. The full map is needed for
cell-level claims, ranks, and top-`k` conclusions.

### 6.2 Scores alone do not identify the channel or latent map

The established project notation is

```tex
\theta_j\sim N(\mu,\tau^2),
```

```tex
S_j=a+b\theta_j+\eta_j,
\qquad
\eta_j\sim N(0,\sigma_S^2).
```

The score law reveals only

```tex
\E[S_j]=a+b\mu
```

and

```tex
\Var(S_j)=b^2\tau^2+\sigma_S^2.
```

Therefore the five channel parameters

```tex
\psi=(\mu,\tau^2,a,b,\sigma_S^2)
```

are not separately identified by the marginal score distribution.

An additional correction made in this conversation: with nonzero score noise,
the observed score ranking does not identify the realized ranking of the true
cell gains. The scores identify only their own ranking unless added assumptions
discipline the noise.

### 6.3 Randomized validation identifies the Gaussian channel

For validated cells:

```tex
\widehat v_j=\theta_j+\varepsilon_j,
\qquad
\E[\varepsilon_j\mid\theta_j,S_j]=0,
\qquad
\Var(\varepsilon_j\mid\theta_j,S_j)=s_j^2.
```

Under random validation, known `s_j^2`, independence of the measurement
errors, and `\tau^2>0`:

```tex
\mu=\E[\widehat v_j],
```

```tex
\tau^2=\Var(\widehat v_j)-\E[s_j^2],
```

```tex
b=\frac{\Cov(S_j,\widehat v_j)}{\tau^2},
```

```tex
a=\E[S_j]-b\mu,
```

```tex
\sigma_S^2=\Var(S_j)-b^2\tau^2.
```

The proposal states identification but does not display these five steps. The
new note derives them.

### 6.4 Empirical-Bayes map

For an unvalidated cell:

```tex
m_j
=\E[\theta_j\mid S_j]
=\mu+
\frac{b\tau^2}{b^2\tau^2+\sigma_S^2}
(S_j-a-b\mu).
```

The posterior variance and reliability ratio are

```tex
\omega^2
=
\frac{\tau^2\sigma_S^2}{b^2\tau^2+\sigma_S^2},
\qquad
\lambda
=
\frac{b^2\tau^2}{b^2\tau^2+\sigma_S^2}.
```

For a validated cell:

```tex
P_j
=
\frac{1}{\tau^2}
+\frac{b^2}{\sigma_S^2}
+\frac{1}{s_j^2},
```

```tex
m_j^{\mathrm{val}}
=
P_j^{-1}
\left(
\frac{\mu}{\tau^2}
+\frac{b(S_j-a)}{\sigma_S^2}
+\frac{\widehat v_j}{s_j^2}
\right).
```

These cell estimates are model-based. Random validation identifies the shared
channel; the Gaussian/exchangeability assumptions extrapolate to unvalidated
cells.

### 6.5 Proposal claims that remain unproved

Do not silently promote the following to theorems:

1. The proposed excess-risk expansion
   `\Lambda_j/m+o(m^{-1})` is asserted but not proved in the five-page
   proposal.
2. The aggregate rate `m^{-1/2}` is also asserted without the complete
   regularity conditions.
3. The numerical ``knee'' near `m=40` is a simulation result.
4. The claim that true value gain lies between the pre-AI and actual-use
   basket values requires an ordering or behavioral assumption not supplied in
   the proposal.

### 6.6 Ordinary PPI: estimand, estimator, and correction

For the mean:

```tex
\theta^\star=\E[Y_i].
```

The estimator is

```tex
\widehat\theta_{\mathrm{PP}}
=
\frac{1}{N}\sum_{i=1}^Nf(\widetilde X_i)
+
\frac{1}{n}\sum_{i=1}^n\{Y_i-f(X_i)\}.
```

Its expectation is

```tex
\E[f(X_i)]+\E[Y_i-f(X_i)]=\E[Y_i]=\theta^\star.
```

The model prediction need not be unbiased. Prediction bias cancels because the
same prediction enters the large prediction mean and the labeled residual.

Under independent labeled and unlabeled samples:

```tex
\Var(\widehat\theta_{\mathrm{PP}})
=
\frac{\sigma_f^2}{N}
+
\frac{\sigma_{f-Y}^2}{n}.
```

Prediction quality supplies precision. The residual correction supplies
validity.

### 6.7 Finite-population PPI

For a fixed population:

```tex
\theta_N=\frac{1}{N}\sum_{i=1}^NY_i.
```

If `L` is a simple random sample without replacement:

```tex
\widehat\theta_{\mathrm{PP},N}
=
\frac{1}{N}\sum_{i=1}^Nf(X_i)
+
\frac{1}{n}\sum_{i\in L}\{Y_i-f(X_i)\}.
```

Let `e_i=Y_i-f(X_i)`. Then:

```tex
\widehat\theta_{\mathrm{PP},N}-\theta_N
=
\frac{1}{n}\sum_{i\in L}e_i
-
\frac{1}{N}\sum_{i=1}^Ne_i.
```

Thus the estimator is exactly design-unbiased and:

```tex
\Var(\widehat\theta_{\mathrm{PP},N})
=
\frac{1}{n}\left(1-\frac{n}{N}\right)S_e^2.
```

PPI can identify a mean or another supported functional without recovering
every latent unit outcome.

### 6.8 Multi-task PPI

For task `t`:

```tex
\theta^{(t)}
=
\frac{1}{N}\sum_{i=1}^NY_i^{(t)}.
```

For a fixed recalibrator `s` and fixed power coefficient `\gamma_t`:

```tex
\widehat\theta^{(t)}
=
\overline Y_L+\gamma_t(\overline s_N-\overline s_L).
```

Let:

```tex
Z_i=Y_i-\gamma_ts_i.
```

Then:

```tex
\widehat\theta^{(t)}-\theta^{(t)}
=
\overline Z_L-\overline Z_N.
```

The exact finite-population variance is:

```tex
V(s,\gamma_t)
=
\frac{1}{n_t}\left(1-\frac{n_t}{N}\right)
\left[
S_Y^2-2\gamma_tS_{Ys}+\gamma_t^2S_s^2
\right].
```

The oracle coefficient is:

```tex
\gamma^\star(s)
=
\frac{S_{Ys}}{S_s^2}
=
\frac{\Cov_N(Y,s)}{\Var_N(s)}.
```

The minimized variance is:

```tex
V^\star(s)
=
\frac{1}{n_t}\left(1-\frac{n_t}{N}\right)
S_Y^2\{1-\rho_N^2(Y,s)\}.
```

### 6.9 GREPPI and AREPPI

GREPPI:

1. learns `s^{(-t)}` from labels in tasks other than `t`;
2. applies it to every proxy in task `t`;
3. retains a within-task residual correction using task-`t` labels; and
4. optionally estimates `\gamma_t` from task-`t` labels.

The other tasks learn a better proxy. They do **not** provide the target
task's bias correction. A target task still needs its own labels.

AREPPI combines a global and a local recalibrator:

```tex
u_i^{(t)}
=
\kappa_t s_{\mathrm{local}}^{(t)}(\widehat Y_i^{(t)})
+
(1-\kappa_t)s^{(-t)}(\widehat Y_i^{(t)}).
```

The local prediction must be out-of-fold for labeled observations. AREPPI is a
protection against heterogeneous proxy-truth relationships across tasks; it is
not a method for a task with no target labels.

### 6.10 Nonlinear necessity

For an affine recalibrator:

```tex
s(\widehat Y)=a\widehat Y+b,
```

squared correlation is unchanged:

```tex
\rho_N^2(Y,a\widehat Y+b)=\rho_N^2(Y,\widehat Y).
```

Therefore affine cross-task recalibration cannot improve oracle variance
beyond power-tuned PPI.

Let:

```tex
m(z)=\E_N[Y\mid\widehat Y=z].
```

Residual orthogonality gives:

```tex
\Cov_N\{Y-m(\widehat Y),\phi(\widehat Y)\}=0.
```

Hence:

```tex
\rho_N^2\{Y,\phi(\widehat Y)\}
\leq
\frac{\Var_N\{m(\widehat Y)\}}{\Var_N(Y)}
=
R_{Y\sim\widehat Y}^2.
```

The bound is attained by `\phi=m`. A strict improvement over the identity
proxy is possible if and only if `m(z)` is non-affine on the finite-population
proxy support, apart from degenerate zero-variance cases.

### 6.11 Small-sample power-tuning qualification

If `s` and `\gamma_t` are fixed relative to the target label draw, the
finite-population estimator is exactly design-unbiased.

If `\widehat\gamma_t` is estimated using the same target labels that estimate
the residual correction, exact finite-sample unbiasedness and the simple
variance argument no longer follow. MT-PPI reports small-sample undercoverage
from this reuse and sets the source paper's power coefficient to one in its
human-data application. Preserve this qualification.

## 7. The two possible data designs for the AI project

This distinction is the most important project-level conclusion.

### 7.1 Design A: one broad score per cell

Observed:

- `S_j` for every cell;
- `\widehat v_j` only for randomly selected validation cells.

Consequences:

- a design-based aggregate is possible;
- unvalidated cell effects require empirical Bayes or another explicit
  cross-cell model;
- ordinary MT-PPI does not directly apply because each cell does not contain a
  population of proxy observations plus target-cell labels.

With inclusion probability `\pi_j` and a calibrated prediction `g(S_j)` fixed
relative to cell `j`'s validation indicator and experimental noise:

```tex
\widehat{\bar\theta}_{\mathrm{PPI}}
=
\sum_{j=1}^Jp_jg(S_j)
+
\sum_{j:R_j=1}
\frac{p_j}{\pi_j}
\{\widehat v_j-g(S_j)\}.
```

Then:

```tex
\E[\widehat{\bar\theta}_{\mathrm{PPI}}]
=
\sum_{j=1}^Jp_j\theta_j
=
\bar\theta.
```

If `g` is trained on the same validation observations used for the residual
correction, use an independent calibration split or design-compatible
cross-fitting. Do not treat a same-sample fitted `g` as fixed without
justification.

For equal weights and a simple random sample of `m` cells:

```tex
\widehat{\bar\theta}_{\mathrm{PPI}}
=
\frac{1}{J}\sum_{j=1}^Jg(S_j)
+
\frac{1}{m}\sum_{j:R_j=1}
\{\widehat v_j-g(S_j)\}.
```

This identifies the aggregate while individual unvalidated `\theta_j` remain
unidentified without the channel model.

### 7.2 Design B: many instances inside every cell

Observed after fixing one basket:

- `\widehat Y_{ij}` for every instance in every cell;
- `Y_{ij}` for a random subset `L_j` inside every reported cell.

Then:

```tex
\theta_j=\frac{1}{N_j}\sum_{i=1}^{N_j}Y_{ij}
```

is a task mean, and GREPPI can be written:

```tex
\widehat\theta_j
=
\frac{1}{N_j}\sum_{i=1}^{N_j}s^{(-j)}(\widehat Y_{ij})
+
\frac{1}{n_j}\sum_{i\in L_j}
\left[
Y_{ij}-s^{(-j)}(\widehat Y_{ij})
\right].
```

Conditional on the leave-cell-out recalibrator, random sampling inside cell
`j` gives:

```tex
\E[\widehat\theta_j]=\theta_j.
```

The aggregate is:

```tex
\widehat{\bar\theta}
=
\sum_{j=1}^Jp_j\widehat\theta_j.
```

This design supports design-based cell means only for cells with target-cell
labels.

## 8. Final method-by-question conclusion

- **Weighted mean gain:** use finite-population PPI/GREG with random
  validation. The full map is not required.
- **Every cell mean with labels inside every cell:** use GREPPI or AREPPI,
  with fixed or separately tuned `\gamma_t` when target samples are very small.
- **Every cell mean when many cells have no labels:** use the proposal's
  empirical-Bayes channel or another explicit cross-cell model. PPI does not
  identify a label-free target cell.
- **Rankings and top-`k` cells:** use a posterior distribution for the full
  vector and report ranking uncertainty. Aggregate PPI does not identify
  ranks.
- **Cross-cell dispersion:** use an explicit second-moment procedure or a
  full-vector model. The mean estimator alone does not provide a valid
  dispersion estimator.

The clean synthesis is:

1. PPI supplies design-based correction for supported aggregates or
   cell means.
2. Multi-task PPI can improve precision by learning shared nonlinear
   proxy-truth structure.
3. Empirical Bayes supplies model-based shrinkage, extrapolation to
   unvalidated cells, and the joint vector needed for ranks and top-`k`
   statements.
4. These methods are complements, not substitutes.

## 9. Quality checks completed

For `ai_project_three_papers_whiteboard_note.tex`:

- The proposal, PPI, MT-PPI, and synthesis sections each begin with a notation
  table.
- The working notation was reconciled to the earlier note.
- Obsolete working aliases were removed:
  - no `w_j` or `\Theta` for project weights/means;
  - no proposal `\beta,\nu_j,\widehat\theta_j^v` outside the explicit source
    alias mapping;
  - no `\theta_\star^{(t)}` as the MT-PPI working target;
  - no MT-PPI `\lambda` as the working power coefficient.
- `\lambda` remains the proposal's score-reliability ratio.
- `\gamma_t` remains the MT-PPI power coefficient.
- `\kappa_t` is explicitly documented as AREPPI's mixing weight.
- LaTeX braces balance.
- All `\begin{...}` and `\end{...}` environments balance.
- All display-math delimiters balance.
- The earlier merged note was not modified.

## 10. Remaining production caveat

A LaTeX compiler was not available in the environment used for this
conversation. Therefore:

- the TeX source was structurally checked;
- no compiled PDF was created; and
- no claim of visual PDF verification was made.

If the next environment has a LaTeX compiler, compile the standalone note and
inspect:

1. longtable widths and page breaks;
2. equation overflow;
3. placement of notation-conversion paragraphs;
4. the three-column method comparison table; and
5. hyperlinks and bibliography formatting.

Do not change mathematical notation merely to address layout.

## 11. Broader artifact inventory from this conversation

The July 24 handoff is not the final state of the broader BS/BLM/Kline work.
The following later files also matter.

### Advisor-facing master note

`C:\Users\A8327\OneDrive\Documents\OI\four_tasks_master_note.tex`

Despite the filename, the current version contains five adviser questions:

1. identification of `C^w_{\mathrm{assign}}` and the product/missing-cell
   problem;
2. BS20 estimand, estimator, consistency, and comparison with BS24;
3. the exact BS--project connection;
4. extension to Kline, BLM, GKLP, and other nested models; and
5. exogenous mobility, additive separability, and the relation to OLS
   exogeneity.

This is the current concise master note for the original adviser-level task.
Its notation tables were moved to the beginning of each question. It was
quality-checked for linearity, explicitness, and notation coverage after the
user found missing variables in earlier tables.

### Latest BS20/Kline/Sorkin/Volpe lecture note

`C:\Users\A8327\OneDrive\Documents\OI\bs2020_kline_sorkin_volpe_lecture_note.tex`

This is the later, lecture-style rewrite prompted by the user's concern that
the earlier document compared objects to discussions and used too much
jargon. Treat this as the current standalone note for:

- the BS20 large-economy multiplier `\tau`;
- the meaning of `I_x`;
- baseline versus replicated economies;
- Kline's worker-specific wage-difference object;
- Sorkin--Warwar's and Volpe's definitions of additive separability and
  exogenous mobility; and
- the status of an average edge effect.

The earlier file

`C:\Users\A8327\OneDrive\Documents\OI\bs2020_lln_kline_wage_difference_note.tex`

contains useful derivations but predates the lecture-language correction. Do
not treat its wording as final when the two notes differ in presentation.

### Standalone BLM-to-project proof

`C:\Users\A8327\OneDrive\Documents\OI\blm_to_project_linear_proof_dummy_example.tex`

This remains the canonical standalone derivation of:

- BLM double demeaning;
- the rank-one type/class interaction;
- the selector identity `H=ZMW'`;
- the connection to `uv'`; and
- the worked Stefan/Michael and GS/Meta/BoA example.

### Other related note

`C:\Users\A8327\OneDrive\Documents\OI\model_acceptance_stochasticity_writeup.tex`

This older note compares deterministic and stochastic matching/acceptance
mechanisms in Shimer--Smith, later Borovickova--Shimer, Eeckhout--Kircher, and
GKLP. Use it as supporting context, not as a replacement for the later master
note.

## 12. Broader substantive results from this conversation

### 12.1 Product benchmark and missing cells

- The product law `P_I\otimes P_J` is observable from the observed worker and
  firm marginals.
- Observing many static matches identifies empirical frequencies and hence
  the marginals; it does not reveal the counterfactual conditional mean wage
  `m_{ij}` for a worker--firm pair that never occurs.
- Therefore
  `\Var_{P_I\otimes P_J}(m_{IJ})` is generally not identified when the product
  support contains unobserved cells.
- Consequently `C^w_{\mathrm{assign}}(m)` is not point identified without a
  valid completion restriction for those cells.
- BS20 does not solve this missing-cell problem. Its statistic uses observed
  matched quantities and identifies a different observed-match sorting
  object.

### 12.2 Why the product-law covariance is zero in the additive BS score

Under the product law, the worker draw and firm draw are independent. If the
score is additive,

```tex
s_{IJ}=\lambda_I^{\mathrm{BS}}+\mu_J^{\mathrm{BS}}+\bar w,
```

then:

```tex
\Cov_{P_I\otimes P_J}
(\lambda_I^{\mathrm{BS}},\mu_J^{\mathrm{BS}})=0.
```

The constant `\bar w` contributes no covariance or variance. It is included
to reproduce wage levels; it is irrelevant for the covariance-based sorting
measure.

### 12.3 BS20 estimand, estimator, and consistency insight

- **Estimand:** a population correlation/covariance between a worker's
  expected-wage type and a firm's expected-wage type under the observed match
  distribution.
- **Estimator:** a repeated-match cross-product statistic constructed from
  observed wages.
- **Key consistency insight:** multiply outcomes from distinct matches of the
  same worker or firm. Distinct-match products remove the own-observation
  noise term before aggregation.
- The repeated observations identify expected-wage types without requiring
  the unobserved product-support wage schedule.

### 12.4 BS20 versus BS24

- BS20 supplies an observed-quantities sorting measure and estimator.
- BS24 is not a replacement estimator for the BS20 estimand.
- BS24 supplies a structural search, meeting, acceptance, and bargaining
  mechanism.
- BS24 distinguishes accepted-match effects from meeting effects.
- Its Pareto result is exactly additive for wage **levels** under its
  assumptions.
- The project's log-wage surface needs a separate finite-rank or
  approximation bridge; level additivity must not be silently treated as log
  additivity.

### 12.5 BS and the project

- BS and the project can share the observed matched-wage interface
  `(m,P^{\mathrm{obs}}_{IJ})`.
- They apply different functionals to that interface.
- BS20's observed-match sorting covariance is not the same as the project's
  assignment contribution relative to a product counterfactual.
- BS20 is attractive for observed-support measurement because it does not
  require completing missing worker--firm wage cells.
- That advantage does not make it an estimator of
  `C^w_{\mathrm{assign}}`.

### 12.6 Extension to other models

Separate three questions:

1. Does the model generate an observed matched-wage distribution to which the
   BS population covariance can be applied?
2. Does the data structure contain the repeated matches needed for the BS20
   estimator?
3. Does the model possess the search/meeting/acceptance structure needed for
   the later BS interpretation?

The answers differ:

- The population covariance extends broadly when finite, nondegenerate
  worker- and firm-type wage moments exist.
- The BS20 estimator additionally needs repeated-match sampling conditions.
- The BS24 search interpretation additionally needs model-specific meeting,
  acceptance, and wage-setting assumptions.
- Kline, BLM, GKLP, AKM/KSS, Tukey/Crippa, Shimer--Smith, and
  Eeckhout--Kircher must therefore be compared layer by layer, not declared
  nested wholesale.

### 12.7 BS20 large-economy argument

- `\tau` is a positive-integer replication multiplier in a theoretical
  sequence of economies.
- It is not an observed worker characteristic and not an economic treatment.
- If the baseline economy has `I_x` workers of characteristic `x`, the
  replicated economy has `\tau I_x`.
- Workers may share the same characteristic while remaining distinct worker
  identities with distinct match histories.
- The number of observations per worker may remain bounded.
- Consistency comes from more distinct workers and firms of every type, not
  from observing one worker infinitely often.
- The baseline economy supplies the type composition and economic rules; the
  replicated economies scale the counts while preserving those rules.

### 12.8 Additive separability and exogenous mobility

Keep the following categories separate:

- an **outcome restriction** limits the wage equation;
- a **mobility/selection restriction** limits how workers move across firms;
- a **treatment-effect object** is a wage contrast;
- an **average edge effect** is an average of such contrasts, not an
  assumption.

The clean comparison is:

- Pure additive separability and exogenous mobility are logically separate;
  neither implies the other.
- Sorkin--Warwar's named additive-separability condition is stronger because
  it bundles the additive wage restriction with a transitory-error exogeneity
  condition. This is why it can look as if exogenous mobility is ``built in.''
- Volpe presents additive separability and exogenous mobility as separate
  pillars.
- Kline separates:
  1. the statistical additive AKM wage restriction;
  2. assumptions for causal identification of edge effects; and
  3. no selection on worker-specific treatment gains.
- Therefore Kline's organization is closer to Volpe's two-pillar
  organization.
- Kline's worker-specific wage contrast has the same mathematical form as the
  potential-wage contrast obtained by subtracting Sorkin--Warwar's wage
  equation across two firms.
- Kline's no-selection-on-treatment-effects condition is related to, but not
  equivalent to, Sorkin--Warwar's exogenous-mobility condition.
- An average edge effect can have an additive representation even when
  individual wage potential outcomes are not additively separable. The
  average object does not itself impose additive separability.

### 12.9 Relation to ordinary OLS exogeneity

For:

```tex
Y=X'\beta+\varepsilon,
```

the outcome equation specifies a conditional-mean representation, while:

```tex
\E[\varepsilon\mid X]=0
```

is the selection/exogeneity condition that supports the usual causal or
regression interpretation. The same outcome-versus-selection distinction
organizes additive wage restrictions and mobility exogeneity in matched
worker--firm models.

## 13. Instruction to the next conversation

Read this handoff and
`handoff_2026-07-24_todo_chat_blm_bs_notes.md` before modifying project notes.
Treat `ai_project_three_papers_whiteboard_note.tex` as the latest standalone
AI-project write-up and `todo_chat_merged_notes.tex` as historical context.
Preserve the write-up protocol in Section 2 throughout the next conversation.
