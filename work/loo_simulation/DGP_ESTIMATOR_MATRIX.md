# DGP-by-estimator simulation design

This experiment deliberately separates the process that generates the data
from the procedure used to estimate it. The five rows below are five different
population wage schedules. Every estimator is then run on a panel sampled from
every row.

| Data-generating process | KSS | BLM | Borovičková–Shimer 2020 | Project low-rank plug-in |
|---|:---:|:---:|:---:|:---:|
| Additive AKM | yes | yes | yes | yes |
| Crippa/Tukey nonadditive | yes | yes | yes | yes |
| Discrete-type BLM | yes | yes | yes | yes |
| Continuous low-rank factors | yes | yes | yes | yes |
| GKLP comparative advantage | yes | yes | yes | yes |

“Project low-rank plug-in” means the current plug-in estimator without the
unfinished leave-one-out correction. The code does not label it as the final
project LOO estimator.

## Common observation design

Let \(m_{ij}\) be the complete systematic log-wage schedule and let
\(P_{ij}\) be the population probability of worker \(i\) matching to firm
\(j\). All designs use uniform worker and firm marginals. Sorting is introduced
by

\[
 P_{ij}=a_i b_j\exp\{s_c\alpha_i\psi_j+s_h h_{ij}/\operatorname{sd}(h)\},
\]

where the balancing constants \(a_i,b_j\) restore the uniform marginals.
For each worker, the initial firm is drawn from \(P_{ij}/P_{i+}\). In each
later period, the worker redraws from that same conditional distribution with
probability 0.40 and otherwise stays. Observed wages are

\[
 Y_{it}=m_{iJ_{it}}+\varepsilon_{it},\qquad
 \varepsilon_{it}\overset{iid}{\sim}N(0,0.5^2).
\]

The production configuration uses 300 workers, 18 firms, ten periods, and
100 Monte Carlo replications in every row. Using the same panel dimensions
keeps differences across rows attributable to the DGP rather than sample size.

## The five DGPs

All Gaussian draws below are independent across workers and firms before the
stated correlations are imposed. Centering and normalization are performed in
each realized finite population, so the declared interaction singular values
hold exactly rather than only in expectation.

### 1. Additive AKM

\[
 m_{ij}=\alpha_i+\psi_j,
 \qquad \alpha_i^0,\psi_j^0\overset{iid}{\sim}N(0,1).
\]

The realized \(\alpha_i^0\) and \(\psi_j^0\) are separately centered and scaled
to variance one. This is the correctly specified benchmark for additive
AKM/KSS.

### 2. Crippa/Tukey nonadditivity

\[
 m_{ij}=\alpha_i+\psi_j+\beta_0\alpha_i\psi_j,
 \qquad \beta_0=0.75,
\]

with independently drawn, finite-population-standardized Gaussian
\(\alpha_i\) and \(\psi_j\). The interaction is nonadditive and rank one. This
is the Tukey surface in Crippa, *Identification, Estimation, and Inference in
Two-Sided Interaction Models* (2025), Section 2.2.

### 3. Discrete-type BLM

Workers are randomly assigned to two equally sized latent types and firms to
three equally sized latent classes. Type and class main effects begin as
independent standard-normal draws and are normalized under their realized
type shares. A normalized rank-one type-by-class interaction is added:

\[
 m_{ij}=a_{L_i}+p_{K_j}
       +\lambda u_{L_i}v_{K_j},\qquad \lambda=1.
\]

Thus wages are exactly constant inside each of the six worker-type by
firm-class cells. With two worker types, an arbitrary centered interaction
across a \(2\times3\) table has rank at most one, so this does not restrict the
nonadditive dimension of that table. This row captures the discrete latent
worker- and firm-type structure used by Bonhomme, Lamadon, and Manresa.

### 4. Continuous low-rank factors

\[
 m_{ij}=\alpha_i+\psi_j+U_i'\Lambda V_j,
 \qquad \Lambda=\operatorname{diag}(1,0.5).
\]

Before finite-population normalization, \(\alpha_i,\psi_j\) and the two factor
innovations are standard normal. Each raw worker factor has correlation 0.35
with the raw worker main effect; each raw firm factor has correlation 0.25
with the raw firm main effect. Factor columns are then centered and
orthonormalized under uniform weights. This is the DGP closest to the project
model and has exact interaction rank two.

### 5. GKLP comparative advantage

After residualizing observed covariates, the simulated perfect-information
wage slice is

\[
 m_{ij}=Z_i+c_j+b_j\eta_i
        +\frac12 b_j^2\sigma_e^2,
 \qquad \sigma_e=0.5.
\]

The raw pair \((Z_i,\eta_i)\) is jointly Gaussian with correlation 0.35, and
the raw pair \((c_j,b_j)\) is jointly Gaussian with correlation 0.25. The
\(b_j\eta_i\) term is rank-one comparative advantage; the convex
\(b_j^2\sigma_e^2/2\) term is retained in the firm component. The equation is
the perfect-information log-wage expression in Gibbons, Katz, Lemieux, and
Parent, *Comparative Advantage, Learning, and Sectoral Wage Determination*,
NBER Working Paper 8889 (2002).

## Estimators and truth comparisons

The estimator settings live once, outside the scenario definitions. The
production file therefore cannot accidentally switch BLM off for a
non-BLM row or change the KSS implementation with the DGP.

- **KSS:** PyTwoWay's heteroskedastic leave-out estimate (`kss_he`) is the
  canonical KSS column. The raw AKM and homoskedastic-correction rows are also
  retained as diagnostics. Firm variance and worker-firm covariance are
  compared both with the pseudo-true additive AKM projection and with the
  project's population quantities. Those two truths coincide under AKM but
  need not coincide under nonadditivity.

- **BLM:** The canonical column is estimated-group PyTwoWay BLM with two
  worker types and three firm classes in every DGP. Only the true BLM row has
  genuine simulated type labels. For the other rows, workers and firms are
  ordered by their product-marginal mean wages in the complete schedule and
  divided into equal-count \(2\times3\) reference groups. These labels define
  a fixed grouped projection used only to evaluate the simulation; they are
  never supplied to the estimated BLM procedure. BLM cell means are compared
  with that grouped truth, and the functionals implied by the fitted cell table
  are compared both with the grouped projection and the full-population
  project truth. The latter comparison deliberately includes BLM
  discretization error.

- **Borovičková–Shimer 2020:** PyTwoWay's weighted type-moment estimator is
  compared with its native population type moments. Its covariance is also
  compared with the project's assignment covariance. The targets are not
  relabeled as equal when the definitions differ.

- **Project procedure:** Fixed-rank fits and the BIC-selected fit are reported.
  The four plug-in quantities \(Q_F,H_F,\rho_H,C_{assign}\) are compared with
  both the complete-population truth and the truth for the retained analysis
  sample. This separates sample trimming from estimation error. The primary
  single project column is `project_plugin_bic`.

Crippa and GKLP happen to generate rank-one interaction matrices, but they are
not duplicate experiments: one ties the interaction to the additive worker and
firm effects, while the other uses distinct general-ability and
comparative-advantage coordinates and includes the GKLP convex firm term.

## Running without Codex

Once the code is on a machine, the simulation is an ordinary Python job. It
does not call an OpenAI API and consumes no Codex tokens or credits.

Install once on Windows PowerShell:

```powershell
py -3.11 -m venv .venv311
& .\.venv311\Scripts\python.exe -m pip install -e ".[estimators,test]"
```

Run the one-replication integration check:

```powershell
& .\.venv311\Scripts\python.exe scripts\run_monte_carlo.py `
  --config configs\dgp_estimator_matrix_pilot.json `
  --output results\dgp_estimator_matrix_pilot
```

Run the 100-replication experiment locally with four independent worker
processes:

```powershell
& .\.venv311\Scripts\python.exe scripts\run_monte_carlo_shards.py `
  --config configs\dgp_estimator_matrix.json `
  --output-root results\dgp_estimator_matrix `
  --shard-count 20 `
  --workers 4 `
  --resume
```

The launcher saves each shard immediately, validates resumed shards, and
creates `results\dgp_estimator_matrix\merged` only after all 100 replications
are present exactly once. On a SLURM cluster, submit
`scripts/slurm_dgp_estimator_matrix.sh` from the repository root:

```bash
sbatch scripts/slurm_dgp_estimator_matrix.sh
```

After the array completes, run the merge command printed at the end of that
script.

## Reproducibility files

- `configs/dgp_estimator_matrix.json`: production design.
- `configs/dgp_estimator_matrix_pilot.json`: cheap end-to-end check.
- `scripts/slurm_dgp_estimator_matrix.sh`: 20-task SLURM array template.
- `attempts.csv`: every success, unstable fit, and failure.
- `records.csv`: estimate, target, and error for every scalar comparison.
- `summary.csv`: unconditional Monte Carlo bias and RMSE.
- `conditional_stable_summary.csv`: explicitly conditional robustness table.

The older `full_ladder.json` and its writeup are retained for reproducibility,
but they are not the full whiteboard matrix described here.
