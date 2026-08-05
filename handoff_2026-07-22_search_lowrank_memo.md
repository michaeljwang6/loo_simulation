# Handoff: Search, Selection, Low-Rank Wage Interfaces, and Edge Diagnostics

Date packaged: 2026-07-22

Primary working memo:

- `C:\Users\A8327\OneDrive\Documents\OI\7_17.tex`

This handoff is meant to let a future conversation continue in the same style
without needing to reconstruct the whole prior chat. It summarizes the content,
the working logic, the style rules that emerged, and the remaining verification
caveats. It is a high-level reasoning record, not a transcript of hidden
chain-of-thought.

## Current Output State

The merged TeX memo is `7_17.tex`. It combines and extends:

- `ss_to_bs_memo.tex`
- `bilal_modeling_memo.tex`
- `misc_eventstudy_kline_memo.tex`
- prior source TeX files, especially `shimer_smith_writeup.tex` and
  `model_acceptance_stochasticity_writeup.tex`
- the prior PDF `C:\Users\A8327\Downloads\7_15_Meeting (1).pdf`

The memo format was adjusted to match the 7/15 notes:

- title and date at top
- no table of contents
- numbered sections
- source convention paragraph near the top of each major section
- notation tables before substantive derivations
- every substantial claim either cited to a paper/old memo or labeled as derived
- notation tables use `longtable` so they can break across pages

The local environment did not have a TeX engine available (`pdflatex`,
`xelatex`, `lualatex`, `tectonic`, `latexmk` were not found), so final PDF
rendering was not verified locally. Source-level checks were repeatedly run:

- balanced `\begin{document}` / `\end{document}`
- balanced `longtable`, `enumerate`, and `array` environments
- no smart quotes or Unicode dash punctuation

## Source Files and Papers

Use these as the core source set:

- Shimer--Smith: cited in the memo as printed Econometrica pages.
- Borovickova--Shimer draft:
  `C:\Users\A8327\Downloads\borovickova shimer 2025 assortative matching and wages the role of selection 20240215 - selection.pdf`
- Bilal lecture slides:
  `C:\Users\A8327\Downloads\bilal_2024_slides_search-and-matching_Lecture2.pdf`
- Bonhomme--Lamadon--Manresa:
  `C:\Users\A8327\OneDrive\Documents\OI\papers\bonhomme_lamadon_manresa_2019_distributional_framework.pdf`
- Sorkin--Warwar:
  `C:\Users\A8327\Downloads\sorkin warwar 2026 sw_logadditivity.pdf`
- Kline review:
  `C:\Users\A8327\OneDrive\Documents\OI\papers\kline_2024_review_akm_firm_wage_effects_w33084.pdf`
- Prior notes:
  `C:\Users\A8327\Downloads\7_15_Meeting (1).pdf`
  `C:\Users\A8327\OneDrive\Documents\OI\shimer_smith_writeup.tex`
  `C:\Users\A8327\OneDrive\Documents\OI\model_acceptance_stochasticity_writeup.tex`

## Content Map of `7_17.tex`

### Section 1: From Shimer--Smith to Borovickova--Shimer

Main point:

Shimer--Smith and BS share Bellman/search logic, but differ in the timing of
match quality. Shimer--Smith has deterministic type-pair output `f(x,y)`. BS has
stochastic realized output `z f_{x,y}` and therefore selection into accepted
matches.

Important resolved points:

- Shimer--Smith only appears to have a "worker" Bellman because it is a
  symmetric-agent model. The other side is the same equation with `x` and `y`
  swapped.
- BS writes worker and firm Bellmans separately because workers and firms are
  economically distinct sides: worker payoff is wage `w`; firm payoff is
  `z f_{x,y}-w`.
- The explicit map is:
  - `W(x|y) <-> V^e_{x,y}(z,w)`
  - `W(x) <-> V_x^u`
  - `pi(x|y) <-> w`
  - `W(y|x) <-> V^f_{y,x}(z,w)`
  - `W(y) <-> V_y^v`
  - `pi(y|x) <-> z f_{x,y}-w`
  - `f(x,y) <-> z f_{x,y}`
- With `gamma=1/2` and degenerate `z=z^*`, BS collapses algebraically to the
  Shimer--Smith payoff form after defining `\tilde f(x,y)=z^* f_{x,y}`.
- With nondegenerate `z`, a type pair maps to a distribution of realized
  surplus/payoffs, and wages are observed only when `z` clears the threshold.
- The BS relation between ITT and ATT is:
  `ITT_x(y) = [1-G(\bar z_{x,y})] ATT_x(y)`.
- Hence, if `0 < Pr(accept|x,y) < 1` and accepted matches have positive surplus,
  then `0 < ITT_x(y) < ATT_x(y)`.
- For BS's Pareto shape parameter `theta`, formulas with `1/(theta-1)` need
  `theta>1`. This is not the same as Bilal's market tightness `theta_t`, which
  only needs to be positive.

### Section 2: Bilal Slides and the Low-Rank Modeling Section

Main point:

Bilal's slides are used as the generic DMP search scaffold, not as a direct
nesting target for the low-rank wage model.

The section now presents the DMP logic linearly:

1. matching function maps unemployment and vacancies to meetings
2. tightness `theta_t=v_t/u_t` determines job-finding and vacancy-filling rates
3. Bellman equations value unemployment, employment, vacancies, and filled jobs
4. joint surplus removes the wage because the wage is an internal transfer
5. Nash bargaining splits surplus
6. free entry pins down tightness

Correction already made:

- Bilal surplus equation uses the negative foregone-search term:
  `S_t = z_t-b-\beta f_t E_t[E_{t+1}-U_{t+1}]
         +\beta(1-s)E_t[S_{t+1}]`.

Low-rank modeling content:

- The project interface is:
  `Y_{ij}=X_{ij}'delta+alpha_i+psi_j+u_i' Lambda v_j+epsilon_{ij}`.
- The saturated `K x L` type-cell mean surface is embedded by:
  - double-demeaning the type-cell matrix
  - assigning type indicators as worker and firm factors
  - setting `Lambda` equal to the double-centered interaction matrix
- The rank bound is:
  `rank(M) <= min{K-1,L-1}`.

### BLM Nesting Section

Main point:

The memo should never simply say "we nest BLM" without qualification.

Correct statement:

- The low-rank interface nests BLM's static wage-equation/type-cell mean layer.
- It does not nest the full BLM empirical framework.

Reason:

- The low-rank model is a family of static conditional mean surfaces,
  `mathcal F_star`.
- Full BLM is a family of joint panel-data distributions,
  `mathcal P_BLM`, including earnings distributions, worker-type proportions,
  firm-class/mixture structure, and mobility/dynamic transition laws.
- The mean projection `Pi_mean(P_BLM)` can lie inside `F_star`, but
  `P_BLM` itself is not even the same type of object as `F_star`.

Important nuance from the user:

- Crippa/BLM rank confusion: "low rank" can mean algebraic rank of the
  interaction matrix or effective dimension/parameter count.
- The simple BLM interactive mean `a_t(k)+b_t(k) alpha_i` is algebraically
  rank one if `alpha_i` is scalar, but it can still carry one worker parameter
  per worker. Rank-one surface does not automatically mean dimension-reduced.
- Surface nesting is not factor identification. If multiple normalizations
  generate the same static mean surface, the low-rank interface nests the
  observational surface, not the structural factors without normalization.

### Section 3: Sorkin--Warwar Event Studies and Kline Edge Effects

Sorkin--Warwar event-study section:

- Rewritten to be self-contained.
- It defines the event-study regression:
  `Delta Y_i = beta_ES Delta psi_i + error_i`.
- Under additive AKM, `beta_ES=1`.
- With match effects:
  `Delta Y_i = Delta psi_i + Delta mu_i`, so
  `beta_ES = 1 + Cov(Delta mu_i, Delta psi_i)/Var(Delta psi_i)`.
- Therefore the event study can pass with match effects if match-quality changes
  are not aligned with firm-effect changes in the tested sample.
- Paired movers test separability more directly because two workers moving along
  the same `A -> B` transition difference out firm effects and test whether the
  worker wage gap is portable.
- SW's model has payoff `h_i+p_j+m_{ij}`.
- SW simulation: full-sample event-study coefficient `0.99`; EE coefficient
  `0.57`; EUE coefficient `0.99`; EE share `41%`.
- Low power comes partly from pooling: pooled slopes include between-group mean
  differences, not only within-group slopes.

Kline Section 3:

- AKM is a path-independence restriction on directed mover edge effects.
- Kline's unrestricted edge model is `R=E Delta+u`.
- AKM imposes `Delta=B'psi`.
- Cycle sums are the content: if edge effects are differences of firm levels,
  closed cycles sum to zero.
- Kline's edge effects can have causal interpretation under exclusion,
  parallel trends, and stationarity, without requiring additive separability.

Kline low-rank nesting section:

- Rewritten linearly.
- Local notation table added.
- Flow:
  1. Low-rank model is about wage levels.
  2. Kline unrestricted edge model is about selected mover wage changes.
  3. AKM is nested as `Lambda=0`.
  4. A low-rank wage surface implies structured edge effects:
     `Delta_{jk}=psi_k-psi_j+mu_{jk}' Lambda (v_k-v_j)`.
  5. Unrestricted Kline edge effects are not nested because arbitrary edge
     vectors do not have to satisfy the shared-factor restriction.
- Dimension-count argument:
  Holding edge compositions fixed, low-rank edge effects are governed by shared
  parameters `(psi_j, v_j, Lambda)`, with at most `J+J d_v+d_u d_v` raw degrees
  of freedom before normalizations. Kline's unrestricted edge vector has `|E|`
  degrees of freedom. With many edges and fixed low dimensions, generic edge
  tables are not in the image of the low-rank representation.

## Style Rules for Continuing

Use this style unless the user asks otherwise:

1. Make every section self-contained enough to explain to an unknowledgeable peer.
2. Put a notation table at the beginning of every major section or any subsection
   that introduces a separate notation system.
3. Use the same notation-table format:
   `longtable` with two `p{...}` columns.
4. Every variable introduced in text or equations should appear in a notation
   table nearby.
5. Every factual claim should be either:
   - cited to a paper,
   - cited to the old PDF/TeX notes,
   - or labeled as derived/derived comparison/derived synthesis.
6. Do not rely on "as above" or "as in the old notes" when the section should be
   teachable by itself.
7. Prefer linear derivations:
   - define object
   - write equation
   - rearrange
   - state implication
   - explain intuition
   - cite source
8. Separate objects that live in different spaces:
   - wage-level surfaces
   - accepted-match wages
   - meeting values/ITT
   - edge effects
   - full panel distributions
9. Be careful with overloaded symbols:
   - BS `theta`: Pareto shape, needs `>1` for mean formulas.
   - Bilal `theta_t`: market tightness, positive but not necessarily greater
     than one.
   - `M`: matching function in Bilal, double-centered matrix in modeling,
     residual-maker in Kline.
   - `m_{ij}`: match effect in SW or mean surface in Kline/low-rank context,
     depending on the local table.
10. Avoid fluff. The user prefers rigorous but digestible mathematical writing.

## Verification Routine Used

After edits, run:

```powershell
$c=Get-Content -Raw 7_17.tex; [PSCustomObject]@{
  BeginDocument=([regex]::Matches($c,'\\begin\{document\}')).Count
  EndDocument=([regex]::Matches($c,'\\end\{document\}')).Count
  BeginLongtable=([regex]::Matches($c,'\\begin\{longtable\}')).Count
  EndLongtable=([regex]::Matches($c,'\\end\{longtable\}')).Count
  BeginEnumerate=([regex]::Matches($c,'\\begin\{enumerate\}')).Count
  EndEnumerate=([regex]::Matches($c,'\\end\{enumerate\}')).Count
  BeginArray=([regex]::Matches($c,'\\begin\{array\}')).Count
  EndArray=([regex]::Matches($c,'\\end\{array\}')).Count
}
Select-String -Path 7_17.tex -Pattern '[“”‘’–—]'
```

Current last known source-level state:

- six `longtable`s balanced
- two `array`s balanced
- no smart punctuation found

## Suggested Starter Prompt for the Next Conversation

Paste this into a future Codex conversation:

```text
We are continuing the search/selection/low-rank wage memo project. Please first
read:

- C:\Users\A8327\OneDrive\Documents\OI\handoff_2026-07-22_search_lowrank_memo.md
- C:\Users\A8327\OneDrive\Documents\OI\7_17.tex

Use the same style as the current memo: self-contained sections, notation table
for every section/subsection with its own notation, every variable defined near
where it is used, every factual claim cited to a paper/old note or labeled as
derived, and linear mathematical derivations.

Do not start editing immediately. First internalize the handoff and the current
memo, then tell me what context you loaded and ask only essential clarification
questions.
```

## Important Caution

The file `combined_search_lowrank_memo.tex` was referenced earlier in the chat,
but the current live merged file in the workspace is `7_17.tex`. Future work
should use `7_17.tex` unless the user explicitly points to another file.
