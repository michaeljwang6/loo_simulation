# Comparing Schedule-Based Low-Rank Functionals with KSS, BLM, and Borovičková-Shimer

## Scope and source convention

This note reports a seven-design, 100-replication Monte Carlo comparison of
the project's schedule-based estimands with additive fixed-effects/KSS,
grouped BLM, and Borovičková-Shimer procedures. The simulated complete wage
schedule makes every population target observable, so estimation error and
estimand differences can be evaluated separately.

The project procedure implemented here is the **low-rank plug-in without LOO
correction**. The project's full leave-worker-out estimator is still under
development and is not simulated.

Statements marked **Source** reproduce a definition, method, design choice, or
numerical output from the cited paper or repository file. Statements marked
**Derived** follow algebraically from those definitions or summarize the
reported Monte Carlo output. No literature result is attributed to the
simulation, and no simulation result is attributed to the cited papers.

*Source: `SIMULATION_CONTRACT.md`; `configs/full_ladder.json`;
`PRODUCTION_RESULTS.md`; reporting commit `8922da6`; production configuration
fingerprint `e03e0c242706502fcd58f20c1cc52c4deeb234f48c981a4885e13157d6ac2a09`.*

## Notation

| Symbol | Meaning | Provenance |
|---|---|---|
| `i, j` | Worker and firm indices; `I, J` are their population counts. | Source: project simulation contract. |
| `Y_ij` | Observed wage or log wage for worker `i` at firm `j`. | Source: project model. |
| `X_ij' delta` | Observed covariate component. It is set to zero in the current simulation. | Source: project model and simulation contract. |
| `alpha_i, psi_j` | Additive worker and firm components. | Source: project model. |
| `u_i, v_j, Lambda` | Worker factor, firm factor, and interaction matrix; `rank(Lambda)=r`. | Source: project model. |
| `M=(m_ij)` | Complete systematic wage schedule generated in each replication. | Source: simulation contract. |
| `P_obs` | Population assignment law over worker-firm pairs. | Source: simulation contract. |
| `p_i, q_j` | Worker and firm marginals of `P_obs`. | Source: simulation contract. |
| `mu, a_i, b_j, h_ij` | Product-weighted grand mean, worker main effect, firm main effect, and interaction in `m_ij=mu+a_i+b_j+h_ij`. | Source: simulation contract. |
| `Q_F` | Mean across workers of the firm-side variance of their wage schedules. | Source: project estimand definition. |
| `H_F` | Mean pairwise firm contrast variance across workers. | Source: project estimand definition. |
| `rho_H` | Normalized interaction share, `H_F/(2 Q_F)`. | Derived from the project definitions. |
| `C_assign` | Half the difference between systematic-wage variance under `P_obs` and under `p x q`. | Source: project estimand definition. |
| `V_psi^AKM, C_psialpha^AKM` | Firm-effect variance and worker-firm covariance in the assignment-weighted additive projection. | Derived comparison target defined in the simulation contract. |
| `A_lk^BLM` | Mean wage for latent worker type `l` and firm class `k`. | Source: BLM grouped model; local stationary specialization. |
| `rho_BS` | Correlation between Borovičková-Shimer worker and firm wage types on observed matches. | Source: Borovičková and Shimer (2020); local population analogue. |
| `r_hat_BIC` | Rank selected by the exploratory common-support BIC comparison. | Source: local implementation; not part of the unfinished LOO theory. |
| `e_b` | Replication error, estimate minus its declared population target. | Derived Monte Carlo notation. |

## 1. Model, estimands, and the comparison problem

The project model is

`Y_ij = X_ij' delta + alpha_i + psi_j + u_i' Lambda v_j + epsilon_ij`.

Each replication generates the full systematic schedule `M`, an assignment
law `P_obs`, and a noisy finite panel. Under the product reference distribution
`p x q`, the schedule has the canonical decomposition

`m_ij = mu + a_i + b_j + h_ij`.

The four headline project truths are

`Q_F = sum_i p_i Var_q(m_iJ)`,

`H_F = sum_jk q_j q_k Var_p(m_Ij - m_Ik)`,

`rho_H = H_F / (2 Q_F)`,

`C_assign = 0.5 { Var_Pobs(m_IJ) - Var_p×q(m_IJ) }`.

They satisfy `Q_F = Var_q(b_J) + H_F/2`. Under additive separability,
`h_ij=0`, so `H_F=rho_H=0`, `Q_F=Var_q(b_J)`, and `C_assign` reduces to the
assignment covariance between worker and firm main effects.

*Source: `SIMULATION_CONTRACT.md`, Sections 1-2. Derived identity: substitute
the product-weighted decomposition into the four definitions and use the
zero-mean restrictions on `a_i`, `b_j`, and `h_ij`.*

The central comparison is not always estimator against estimator for a common
target. Under nonadditivity, AKM/KSS estimates moments of the
assignment-weighted additive projection of `M`, whereas the project objects
remain functionals of the complete schedule. Consequently,

`estimate_p - theta_project = (estimate_p - theta_p*) + (theta_p* - theta_project)`.

The first term is estimation error for procedure `p`; the second is an
estimand or model gap. The simulation reports both.

*Derived decomposition from the declared native target `theta_p*` and project
target `theta_project`; see `SIMULATION_CONTRACT.md`, Section 5.*

## 2. Procedures

### 2.1 Low-rank plug-in without LOO correction

For a fixed interaction rank `r`, the current project procedure minimizes

`sum_(i,j in E) n_ij [Ybar_ij - alpha_i - psi_j - u_i'v_j]^2`,

using match counts `n_ij` as weights. The implementation uses alternating
weighted least squares, spectral plus perturbed initializations, factor
centering, and SVD normalization. Positive-rank fits use a connected support
core with at least `r+2` distinct matches per retained worker and firm. The
exploratory BIC selector compares candidate ranks on a common support sample.
Returned fits can be classified as stable or unstable; unstable values remain
in the unconditional procedure-performance calculation.

No quadratic or leave-worker-out correction is applied. This distinction is
essential: the procedure is a low-rank completion plug-in benchmark, not the
project's proposed full LOO estimator.

*Source: `SIMULATION_CONTRACT.md`, Sections 4 and 7;
`src/loo_sim/low_rank.py`; `src/loo_sim/monte_carlo.py`.*

### 2.2 AKM and KSS

The additive benchmark estimates worker and firm fixed effects and reports the
plug-in firm variance and worker-firm covariance. KSS-HO applies the
homoskedastic correction; KSS-HE applies the heteroskedastic leave-out
correction. Kline, Saggio, and Sølvsten develop leave-out estimators for
quadratic forms in linear models with unrestricted heteroskedasticity.

Under nonadditivity, the simulation evaluates AKM/KSS against its declared
native target: the assignment-weighted additive projection of `M`. It does
not assume that a correction derived for the additive linear model recovers
the schedule-based project functionals after an omitted interaction is added.

*Source: Kline, Saggio, and Sølvsten (2020), Econometrica 88(5), 1859-1898,
DOI 10.3982/ECTA16410; local implementation: PyTwoWay 0.3.21 and
`src/loo_sim/pytwoway_estimators.py`; target definition:
`SIMULATION_CONTRACT.md`, Section 5.1.*

### 2.3 Grouped BLM

BLM models earnings distributions with two-sided worker and firm
heterogeneity and permits nonlinear interactions. Its tractable two-step
implementation classifies firms before estimating worker-type by firm-class
earnings distributions. The simulation uses the stationary grouped mean
target `A_lk^BLM` and compares oracle firm groups with wage-distribution
clusters. Label alignment is used only to evaluate Monte Carlo loss.

*Source: Bonhomme, Lamadon, and Manresa (2019), Econometrica 87(3), 699-739,
DOI 10.3982/ECTA15722; local specialization:
`SIMULATION_CONTRACT.md`, Section 5.2, and `src/loo_sim/blm.py`.*

### 2.4 Borovičková-Shimer

The Borovičková-Shimer procedure estimates the correlation between worker and
firm wage types while allowing many workers and firms to have only a small
number of observations. The simulation evaluates it against its native
population analogue,

`rho_BS = Corr_Pobs(lambda_I^BS, mu_J^BS)`,

where each worker type is the worker's mean systematic wage over observed
matches and each firm type is the corresponding firm mean. This correlation
is not relabeled as the project assignment covariance.

*Source: Borovičková and Shimer (2020), "High Wage Workers Work for High Wage
Firms," February 2020 manuscript; earlier version NBER Working Paper 24074,
DOI 10.3386/w24074; local target definition: `SIMULATION_CONTRACT.md`,
Section 3.*

## 3. Monte Carlo design

The ladder changes one economically relevant feature at a time before adding
the grouped and rank-misspecification designs.

| Design | Population | Assignment/sorting | Panel | Candidate ranks |
|---|---|---|---|---|
| Additive, independent | 80 workers, 10 firms, rank 0 | Independent | `T=7` | 0, 1 |
| Additive, common sorting | 80 workers, 10 firms, rank 0 | Common sorting 0.8 | `T=7` | 0, 1 |
| Rank 1, independent | 80 workers, 10 firms, singular value 1 | Independent | `T=7` | 0, 1 |
| Rank 1, common sorting | 80 workers, 10 firms, singular value 1 | Common sorting 0.8 | `T=7` | 0, 1 |
| Rank 1, interaction sorting | 80 workers, 10 firms, singular value 1 | Interaction sorting 0.4 | `T=10` | 0, 1 |
| Grouped BLM | 300 workers, 18 firms; 2 worker types x 3 firm classes | Common 0.4; interaction 0.2 | `T=5` | 0, 1 |
| Rank 2 | 80 workers, 10 firms; singular values 1 and 0.5 | Common 0.4; interaction 0.4 | `T=15` | 0, 1, 2 |

All designs use wage-noise standard deviation 0.5. The redraw probability is
0.75 except in the grouped design, where it is 0.35. There are 100
replications per design. The production run contains 32,400 scalar
estimate-target records and 5,200 estimator attempts: 4,773 successful, 427
unstable, and zero hard failures.

For replication errors `e_b`, the reported bias, Monte Carlo standard error,
and RMSE are

`bias = mean_b(e_b)`,

`MCSE(bias) = sd_b(e_b) / sqrt(B)`,

`RMSE = sqrt(mean_b(e_b^2))`.

The primary result is unconditional and includes finite values returned by
unstable attempts. Stable-only results are separately labeled robustness
calculations; they do not replace the unconditional procedure result.

*Source: `configs/full_ladder.json`; `PRODUCTION_RESULTS.md`;
`src/loo_sim/monte_carlo.py`. Formulas are derived from the stored
replication-level errors.*

## 4. Results

### 4.1 Additive designs: common targets

![Figure 1. Additive-design bias and 95 percent Monte Carlo intervals.](reports/production/figures/additive-bias.png)

Under independent assignment, the low-rank plug-in without LOO correction,
AKM plug-in, KSS-HO, and KSS-HE have nearly identical RMSE. The low-rank
plug-in bias is 0.013 for `Q_F` and -0.005 for `C_assign`; KSS-HE bias is
0.007 and -0.001, respectively.

Under common sorting, the targets still coincide, but the low-rank plug-in
has bias -0.035 (MCSE 0.009) for `Q_F` and -0.072 (0.006) for `C_assign`.
KSS-HE bias is 0.007 (0.009) and -0.002 (0.004). Thus the sorted-design
difference is finite-sample plug-in behavior, not an estimand gap. It is the
kind of bias the future project correction is intended to address, but this
experiment does not establish what the unfinished correction will do.

*Source: `reports/production/tables/additive_comparison.csv`. Derived
interpretation: equality of targets follows from additive separability in
Section 1.*

### 4.2 Correct rank is not a stability certificate

![Figure 2. BIC rank selection and unstable-attempt frequencies.](reports/production/figures/stability-rank-selection.png)

BIC selects the true rank in every additive and rank-one free-factor
replication and in 96 percent of rank-two replications. Nevertheless, 6-12
percent of BIC fits are unstable in the rank-one and rank-two free-factor
designs. In the grouped design, BIC selects rank one in 90 percent of
replications but is unstable in 88 percent.

**Derived interpretation.** Rank selection answers which candidate fits best
on the common observed support. Functional stability additionally requires
the completed schedule to produce reliable off-support project functionals.
The first property does not imply the second.

*Source: `reports/production/tables/stability_rank_selection.csv` and the
stored attempt diagnostics.*

### 4.3 Nonadditivity: stable fits versus unconditional risk

![Figure 3. Unconditional and stable-only RMSE for project functionals.](reports/production/figures/nonadditive-unconditional-vs-stable-rmse.png)

Conditional on a stable fit, free-factor RMSE ranges from 0.142 to 0.296 for
`Q_F`, 0.255 to 0.513 for `H_F`, 0.104 to 0.201 for `C_assign`, and 0.028 to
0.064 for `rho_H`. These values show that stable completion can recover the
four schedule-based objects with moderate error.

The unconditional conclusion is different. The 6-12 percent unstable
free-factor attempts generate very large functionals, producing
unconditional `Q_F` RMSE between 823 and 26,563. In the grouped design,
stable-only `Q_F` RMSE is already 3.586, only 12 fits are stable, and
unconditional `Q_F` RMSE is about 38,100.

**Derived interpretation.** The stable-only panel diagnoses what happens
after conditioning on a fit-dependent screen. It is informative about the
numerical failure mode but cannot be the headline performance measure. The
unconditional panel shows the operational risk of using the current plug-in
procedure without an additional stabilization or correction step.

*Source: `reports/production/tables/nonadditive_comparison.csv`;
`PRODUCTION_RESULTS.md`, nonadditive-results section.*

### 4.4 KSS can estimate its own target while missing `Q_F`

![Figure 4. Native AKM/KSS target gaps and KSS-HE native-target bias.](reports/production/figures/akm-project-estimand-gap.png)

Across the five nonadditive designs, the native AKM firm-variance target is
0.847 to 1.127 below project `Q_F`. By contrast, KSS-HE bias relative to that
native target ranges from -0.010 to 0.001, with MCSE between 0.009 and 0.017.

**Derived interpretation.** Small native-target estimation error is compatible
with a large error relative to the project estimand. This is not a failure of
KSS: it is an estimand difference created by projecting a nonadditive schedule
onto an additive model. KSS also does not produce project `H_F` or `rho_H`.

*Source: `reports/production/tables/nonadditive_comparison.csv`;
native-target definitions in `SIMULATION_CONTRACT.md`, Section 5.1.*

### 4.5 The grouped design favors grouped BLM

![Figure 5. Stable-only RMSE in the grouped BLM design.](reports/production/figures/grouped-blm-comparison.png)

Oracle-group BLM is stable in 100 of 100 replications; estimated-group BLM is
stable in 99. Their aligned cell-mean RMSE is about 0.051. For the four grouped
project functionals, oracle-group RMSE is 0.056, 0.085, 0.013, and 0.017;
estimated-group RMSE is 0.092, 0.110, 0.019, and 0.014. The low-rank plug-in
is stable in only 12 replications, with stable-only RMSE 3.586, 7.431, 4.814,
and 0.478.

**Derived interpretation.** The grouped BLM procedure uses the correct
dimension reduction for this DGP. A low matrix rank does not by itself make
individual-level low-rank completion reliable when the observed support is
weak and the economically relevant structure is discrete worker-type by
firm-class heterogeneity.

*Source: `reports/production/tables/grouped_blm_comparison.csv`. The final
sentence is a comparison derived from the DGP and the reported stability and
RMSE values.*

### 4.6 Borovičková-Shimer estimates a different sorting object

![Figure 6. Bias of the Borovičková-Shimer native worker-firm correlation estimator.](reports/production/figures/bs20-native-correlation-bias.png)

The implemented Borovičková-Shimer procedure returns a value in every
replication but is negatively biased for its own native correlation target.
Bias ranges from -0.049 in the grouped design to -0.404 in the rank-two
design; the additive-independent bias is -0.149. The procedure also uses a
smaller valid sample after removing return spells and insufficiently observed
workers or firms.

**Derived interpretation.** These results are finite-sample properties of the
implemented cleaning and estimation pipeline, not evidence that
`rho_BS=C_assign`. The two objects remain conceptually different: one is a
correlation between matched worker and firm wage types, while the other is a
variance difference for the complete schedule.

*Source: `reports/production/tables/bs20_native.csv`;
`SIMULATION_CONTRACT.md`, Section 3; PyTwoWay 0.3.21 wrapper in
`src/loo_sim/pytwoway_estimators.py`.*

## 5. Interpretation and limits

The experiment yields three conclusions that should remain separate.

1. **Native-estimand performance.** KSS has small average bias for the
   additive projection in these designs; BLM performs well for the grouped
   target; Borovičková-Shimer exhibits material negative finite-sample bias
   for its native correlation.
2. **Project-estimand performance.** The low-rank plug-in without LOO
   correction can recover `Q_F`, `H_F`, `C_assign`, and `rho_H` when completion
   is stable, but it has conditional plug-in bias and severe unconditional
   tail risk.
3. **Estimand gaps.** Under nonadditivity, good AKM/KSS estimation does not
   imply recovery of the schedule-based project objects. The firm-variance
   gap is economically and numerically large in every calibrated
   nonadditive design.

The simulation does not evaluate standard errors for the unfinished project
estimator, prove that its eventual LOO correction removes the observed bias,
or establish that the present stability diagnostic is optimal. The next
controlled experiment is to freeze this DGP ladder and replace the current
plug-in with the completed leave-worker-out estimator once its estimand,
correction, and feasible standard-error formula are finalized.

*Derived synthesis from Figures 1-6 and their source tables. Scope limitation:
`SIMULATION_CONTRACT.md`, estimator-label and reporting contracts.*

## References

- Bonhomme, Stéphane, Thibaut Lamadon, and Elena Manresa. 2019. "A
  Distributional Framework for Matched Employer Employee Data."
  *Econometrica* 87(3): 699-739.
  [https://doi.org/10.3982/ECTA15722](https://doi.org/10.3982/ECTA15722).
- Borovičková, Katarína, and Robert Shimer. 2020. "High Wage Workers Work for
  High Wage Firms." February 2020 manuscript, listed on Robert Shimer's
  working-paper page. Earlier version: NBER Working Paper 24074.
  [Working-paper page](https://sites.google.com/site/robertshimer/research/workingpapers);
  [NBER DOI](https://doi.org/10.3386/w24074).
- Kline, Patrick, Raffaele Saggio, and Mikkel Sølvsten. 2020. "Leave-Out
  Estimation of Variance Components." *Econometrica* 88(5): 1859-1898.
  [https://doi.org/10.3982/ECTA16410](https://doi.org/10.3982/ECTA16410).
- PyTwoWay 0.3.21. Python implementation used for the FE/KSS, BLM, and
  Borovičková-Shimer comparison procedures.
  [https://github.com/tlamadon/pytwoway](https://github.com/tlamadon/pytwoway).
- Project simulation sources: `SIMULATION_CONTRACT.md`,
  `configs/full_ladder.json`, `PRODUCTION_RESULTS.md`, and
  `reports/production/report_metadata.json`.
