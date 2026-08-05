# DGP-by-estimator Monte Carlo: methods and results

This report asks a deliberately symmetric question: what happens when each of four procedures is applied to data from each of five worker-firm wage models? The answer is not a single winner. KSS and BLM perform well when their own structures are correct, while nonadditivity, rank selection, sparse support, and differences in estimands explain the cross-model reversals.

Every numerical statement below is derived from the immutable merged output with configuration fingerprint `3e8ab3b57d35190989c47117f139b414b9f656db8a933d6c5d1df6d2aa5fbcc4`. Equations attributed to papers are cited; all remaining equations and calculations are definitions or direct derivations from the simulation code.

## 1. Question and design

The design is fixed before estimation. Each DGP has 100 replications, and every procedure is applied to every DGP.

| Common parameter | Value |
|---|---:|
| Workers / firms / periods | 300 / 18 / 10 |
| Observations before cleaning | 3,000 |
| Grand mean of systematic wages | 0 |
| Firm-redraw probability after period 1 | 0.40 |
| Wage disturbance | independent $N(0,0.5^2)$ |
| Worker and firm population marginals | uniform |
| Monte Carlo replications per DGP | 100 |
| Master random seed | 20260804 |

All Gaussian coordinates are redrawn in every replication. Main effects are standardized to finite-population mean zero and variance one. Factor columns are centered and orthonormalized under the uniform population weights, so the declared singular values hold exactly in each realized population.

| DGP | Latent-variable construction | Interaction parameters |
|---|---|---|
| AKM | Independent raw worker and firm $N(0,1)$ effects | Rank 0; $m_{ij}=\alpha_i+\psi_j$ |
| Crippa/Tukey | Independent raw $\alpha_i,\psi_j\sim N(0,1)$ | Rank 1; $h_{ij}=0.75\alpha_i\psi_j$; singular value 0.75 |
| BLM types | Two worker types with 150 workers each and three firm classes with 6 firms each; raw type effects and factors are $N(0,1)$ | Rank 1; singular value 1 |
| Low-rank factors | Raw worker factor correlation with $\alpha_i$ is 0.35; raw firm factor correlation with $\psi_j$ is 0.25; innovations are Gaussian | Rank 2; singular values $(1,0.5)$ |
| GKLP | Raw $(Z_i,\eta_i)$ correlation 0.35; raw $(c_j,b_j)$ correlation 0.25 | Rank 1; $\sigma_e=0.5$ and $m_{ij}=Z_i+c_j+b_j\eta_i+\frac{1}{2}b_j^2\sigma_e^2$ |

The Crippa row uses the Tukey surface in Crippa (2025, Section 2.2). The GKLP row uses the residualized perfect-information wage expression in Gibbons, Katz, Lemieux, and Parent (2002, 2005). The BLM row is discrete; the low-rank and GKLP rows use continuous Gaussian coordinates.

Observed matches are drawn from the balanced assignment law

$$
P_{ij}=a_i b_j\exp\{0.5\,g_i f_j+0.3\,h_{ij}/\operatorname{sd}(h)\}.
$$

Here $g_i$ and $f_j$ are the DGP's worker and firm main components. For AKM, $h_{ij}=0$ and the interaction-sorting term is omitted. The balancing constants $a_i$ and $b_j$ restore uniform worker and firm marginals. A worker's first firm is drawn from the worker-specific conditional assignment distribution. In later periods, the worker redraws from that same distribution with probability 0.40 and otherwise stays; a redraw may select the same firm.

## 2. Estimands and estimators

| Symbol | Definition | Interpretation |
|---|---|---|
| $Q_F$ | product-weighted firm-side schedule variance | Firm contribution in the complete schedule |
| $H_F$ | twice the product-weighted interaction variance | Magnitude of nonadditivity |
| $\rho_H$ | interaction share of $Q_F$ | Relative importance of nonadditivity |
| $C_{\mathrm{assign}}$ | covariance induced by observed assignment | Sorting covariance in realized matches |

The project procedure is the current low-rank plug-in with BIC rank selection, without the unfinished LOO correction. KSS estimates additive-projection variance components. BLM estimates a discrete worker-type by firm-class wage surface. BS20 estimates moments of worker and firm wage types. Because these objects are not identical under nonadditivity, Sections 4.3 and 4.4 distinguish accuracy for a procedure's native target from accuracy for the common project target.

| Procedure | Settings used in every DGP |
|---|---|
| KSS-HE | PyTwoWay heteroskedastic correction with fe_exact=False; approximate rather than exact trace and leverage calculations |
| BLM | Estimated firm groups; 2 worker types; 3 firm classes; CDF resolution 10; 4 initializations; retain best 2; at most 250 iterations; threshold $10^{-6}$ |
| BS20 | Weighted PyTwoWay estimator after strongly connected, spell-level, no-return cleaning |
| Project plug-in | 3 starts for positive rank; tolerance $10^{-6}$; at most 300 iterations; common support core requiring degree 3 for candidate set $\{0,1\}$ and degree 4 for $\{0,1,2\}$ |

The BIC candidates are $\{0,1\}$ for AKM and grouped BLM and $\{0,1,2\}$ for Crippa, continuous low-rank, and GKLP. These choices matter because a candidate set that excludes the true rank cannot recover it.

## 3. Evaluation rules

The report uses three execution statuses: **pass or completed**, **returned with warning**, and **failure**. They do not have the same strength across procedures:

| Procedure | Meaning of pass or completed | Meaning of warning |
|---|---|---|
| KSS-HE and BS20 | PyTwoWay returned an estimate; no additional numerical stability test is applied | Not used |
| BLM | Both mover and stayer likelihood paths avoid a one-iteration decline larger than $10^{-4}$ | BLM returned values, but at least one likelihood path violated that monotonicity check |
| Project plug-in | The selected fit converges within 300 iterations; for positive rank, at least 2 of 3 starts differ in objective by no more than 0.0001 times the larger of 1 and the best objective, and their $Q_F$, $H_F$, and $C_{\mathrm{assign}}$ spreads are no more than 0.001 times the larger of 1 and the absolute best value | The fit returned finite values but failed at least one listed condition |

A **failure** means no estimate was returned. The thresholds above are engineering warning rules, not theory-derived or calibrated statistical cutoffs. In particular, a diagnostic pass does not imply correct specification, correct rank, low bias, or proximity to truth. Headline RMSE therefore includes every returned value. RMSE conditional on a diagnostic pass is shown only as a sensitivity calculation.

For continuous DGPs, BLM receives no oracle type labels. Product-marginal wage means define fixed equal-count reference groups only for scoring its fitted cell table. The BLM functionals are also compared with the full project truth, so discretization error remains part of the comparison.

## 4. Results

### 4.1 Completion and numerical warnings

Each cell reports pass or completed / returned with warning / failed. For KSS and BS20, the first number means completion only. BLM is much more sensitive to cleaned-sample support. The project fit passes its convergence and multi-start checks in every additive and Crippa replication, but in only 6% of grouped-BLM replications. The Crippa result is a useful warning: all fits pass numerically even though BIC selects the wrong rank in every replication.

| DGP | KSS | BLM | BS20 | Project plug-in |
|---|---:|---:|---:|---:|
| AKM | 100 / 0 / 0 | 43 / 24 / 33 | 100 / 0 / 0 | 100 / 0 / 0 |
| Crippa/Tukey | 100 / 0 / 0 | 57 / 22 / 21 | 100 / 0 / 0 | 100 / 0 / 0 |
| BLM types | 100 / 0 / 0 | 46 / 35 / 19 | 100 / 0 / 0 | 6 / 94 / 0 |
| Low-rank factors | 100 / 0 / 0 | 46 / 24 / 30 | 100 / 0 / 0 | 78 / 22 / 0 |
| GKLP | 100 / 0 / 0 | 50 / 32 / 18 | 100 / 0 / 0 | 94 / 6 / 0 |

![Figure 1. Pass or completion, returned-with-warning, and failed attempts. KSS and BS20 use completion only.](reports/dgp_estimator_matrix/figures/status-matrix.png)

BLM has 121 failures out of 500 attempts. 118 are directly attributable to no stayer event or a missing stayer firm class after procedure-specific cleaning. The remaining failures are invalid likelihood starts. This is why BLM accuracy cannot be summarized without its return rate.

### 4.2 Correctly specified benchmarks

The expected benchmarks work. Under the additive AKM DGP, KSS-HE estimates $Q_F$ with bias +0.0032 and RMSE 0.048; its assignment-covariance bias is -0.0001. Under the grouped BLM DGP, the 81 returned BLM fits have biases -0.0050 for $Q_F$, -0.0004 for $H_F$, and +0.0023 for $C_{\mathrm{assign}}$. Thus the main problem for BLM in its preferred row is support and completion, not bias among returned fits.

### 4.3 Cross-DGP accuracy

Figure 2 compares RMSE against the common population-project truth among all returned estimates. `N/A` means that the implemented procedure does not report that object; it does not mean zero or failure.

| Procedure | $Q_F$ | $H_F$ | $\rho_H$ | $C_{\mathrm{assign}}$ |
|---|---:|---:|---:|---:|
| KSS-HE | Reported | N/A | N/A | Reported |
| BLM | Reported | Reported | Reported | Reported |
| BS20 | N/A | N/A | N/A | Reported |
| Project plug-in | Reported | Reported | Reported | Reported |

KSS's additive model implicitly sets interaction quantities to zero, but this implementation does not report them as KSS estimates. BS20 reports a native firm-type variance, but it is not labeled $Q_F$ because the two population objects differ. The heatmap therefore leaves those cells N/A instead of silently redefining the procedures.

The two `0*` cells require a different explanation. Under the AKM DGP, the project truth has $h_{ij}=0$, hence $H_F=\rho_H=0$. BIC also selects rank zero in all 100 AKM replications, so the fitted project model imposes the same zeros. Their raw RMSEs are 6.59e-31 and 3.31e-31; these are floating-point residue displayed as structural zeros. They test whether rank selection creates a false interaction, not ordinary precision in estimating a nonzero quantity. The log color scale is used because positive-rank warning cases generate errors above one million in some cells.

![Figure 2. RMSE against the common project truth among all returned estimates. N/A means not reported; 0* marks AKM structural zeros.](reports/dgp_estimator_matrix/figures/common-target-rmse.png)

KSS remains accurate for its additive target, but its error relative to project $Q_F$ expands when nonadditivity changes the target. BLM is highly accurate on the grouped DGP when it returns. The project procedure performs well for GKLP conditional on a diagnostic pass, but performs poorly for Crippa because BIC always removes the true rank-one interaction. Under the rank-two DGP, BIC always selects rank one and some returned fits have very large functional errors.

### 4.4 Estimand differences: native versus project targets

Section 4.3 measures each reported output against the project truth. That error combines two distinct components. For procedure $p$ with native target $\theta_p$ and project target $\theta_{\mathrm{proj}}$:

$$
\widehat{\theta}_p-\theta_{\mathrm{proj}}=(\widehat{\theta}_p-\theta_p)+(\theta_p-\theta_{\mathrm{proj}}).
$$

The first term is estimation error for the procedure's own object. The second is an estimand difference. This section reports the second term, averaged over all 100 simulated populations; it contains no estimator sampling error.

| DGP | KSS $Q_F$ gap | KSS covariance gap | BS20 covariance gap |
|---|---:|---:|---:|
| AKM | +0.0000 | +0.0000 | +0.416 |
| Crippa/Tukey | -0.394 | -0.0028 | +0.824 |
| BLM types | -0.752 | +0.014 | +0.622 |
| Low-rank factors | -1.127 | -0.017 | +0.459 |
| GKLP | -0.741 | -0.080 | +0.452 |

![Figure 3. KSS and BS20 native population targets minus the corresponding project targets.](reports/dgp_estimator_matrix/figures/native-project-target-gaps.png)

For example, under Crippa the mean KSS native firm-variance target is 1.169, while project $Q_F$ is 1.562. The target gap is -0.394. KSS's native-target bias is only -0.015, but its RMSE against project $Q_F$ is 0.524. Much of the common-target error is therefore disagreement about the object, not failure to estimate the KSS object.

BLM creates an additional estimand difference when a continuous wage schedule is reduced to two worker groups by three firm groups. The table below reports grouped-schedule functional minus full-schedule project functional. The grouped BLM DGP uses its true simulated labels; all other rows use deterministic equal-count groups based on product-marginal wage means.

| DGP | BLM $Q_F$ gap | BLM $H_F$ gap | BLM $\rho_H$ gap | BLM covariance gap |
|---|---:|---:|---:|---:|
| AKM | +0.0037 | +0.011 | +0.0057 | -0.155 |
| Crippa/Tukey | -0.226 | -0.591 | -0.162 | -0.241 |
| BLM types | -0.0000 | -0.0000 | +0.0000 | +0.0000 |
| Low-rank factors | -1.213 | -2.452 | -0.532 | -0.186 |
| GKLP | -0.867 | -1.959 | -0.475 | -0.271 |

The project plug-in has no analogous row because its native target is the project target by definition. These population gaps are why native-target performance and common-target performance must be reported separately in the grand comparison.

### 4.5 Rank selection

After removing additive worker and firm effects, rank is the number of independent interaction dimensions in

$$
m_{ij}=\alpha_i+\psi_j+\sum_{\ell=1}^{r}\lambda_{\ell}U_{i\ell}V_{j\ell}.
$$

Rank zero is AKM: workers have no firm-specific comparative advantage. Rank one allows one comparative-advantage dimension; rank two allows two. The estimator does not observe $r$, so it fits every candidate on the same retained sample and selects the smallest observation-level BIC:

$$
\operatorname{BIC}(r)=n\log(\operatorname{SSE}_r/n)+\operatorname{df}(r)\log n,
$$

where $\operatorname{df}(r)=N+J-1+r(N+J-2-r)$. The first term rewards fit; the second penalizes the additional worker and firm factor coordinates. This BIC is an exploratory simulation rule, not a derived part of the unfinished LOO theory.

| DGP | True rank | Selected ranks (0 / 1 / 2) | Diagnostic pass |
|---|---:|---:|---:|
| AKM | 0 | 100 / 0 / 0 | 100% |
| Crippa/Tukey | 1 | 100 / 0 / 0 | 100% |
| BLM types | 1 | 2 / 98 / 0 | 6% |
| Low-rank factors | 2 | 0 / 100 / 0 | 78% |
| GKLP | 1 | 0 / 100 / 0 | 94% |

![Figure 4. Each cell is the percentage of 100 replications selecting that rank. The orange outline marks the true rank; the right label is the separate diagnostic-pass rate.](reports/dgp_estimator_matrix/figures/rank-selection.png)

A dark cell inside the orange outline means correct selection. BIC is correct in 100% of AKM and GKLP replications and 98% of grouped-BLM replications. It fails systematically in the two most informative continuous tests: Crippa is assigned rank zero in all 100 replications, and the rank-two DGP is assigned rank one in all 100. In Crippa, the fit improvement from rank one never overcomes the BIC penalty. In the rank-two DGP, the selection pattern is consistent with retaining the stronger singular-value-1 dimension while discarding the weaker singular-value-0.5 dimension.

Rank selection and numerical warnings answer different questions. Crippa passes the project numerical checks in 100% of replications while choosing the wrong rank in 100%. On the grouped-BLM DGP, the project selector chooses the correct rank in 98% but passes its multi-start diagnostic in only 6%. Selecting too low a rank forces real nonadditivity to zero; selecting too high a rank can fit noise and destabilize schedule completion. A future LOO correction cannot repair either kind of rank mistake, so rank choice must be solved before interpreting LOO bias correction as the main remaining problem.

### 4.6 Numerical warnings and tail risk

Figure 5 compares RMSE across all returned project-BIC values with RMSE conditional on passing the convergence and multi-start checks in Section 3. Large gaps mean that warning cases dominate squared error. The pass-only line is a useful sensitivity calculation, but it conditions on an outcome of estimation and therefore cannot replace unconditional procedure performance. The thresholds are heuristic, so the figure should be read as evidence of tail-risk concentration rather than a formal good-fit/bad-fit classification.

![Figure 5. Project plug-in RMSE among all returned fits and conditional on a diagnostic pass.](reports/dgp_estimator_matrix/figures/project-returned-vs-stable-rmse.png)

## 5. Interpretation

First, the original all-AKM comparison was uninformative because it rewarded additive estimators by construction. The full matrix now exposes both model misspecification and estimand differences.

Second, correct specification is visible but not sufficient. KSS is accurate under AKM, and returned BLM fits are accurate under the grouped BLM DGP. BLM nevertheless needs enough mover and stayer support after cleaning.

Third, the current project plug-in's central weakness is rank selection and positive-rank numerical reproducibility, not merely small-sample bias. The Crippa and rank-two results show that BIC can erase or truncate economically meaningful nonadditivity. Rank choice and multi-start agreement must be addressed before a leave-out correction can solve bias.

Fourth, no single RMSE table can honestly compare all procedures on all quantities. N/A cells, structural zeros, and native-target gaps have different meanings. KSS and BS20 have native targets that differ from the schedule-based objects; BLM adds discretization error outside a genuinely grouped DGP; and only BLM and the project procedure yield all four project functionals. The report therefore keeps completion, numerical warnings, native-target accuracy, estimand differences, and common-target accuracy conceptually separate.

## References

Bonhomme, S., Lamadon, T., and Manresa, E. (2019). A Distributional Framework for Matched Employer-Employee Data. *Econometrica*, 87(3), 699-739. [doi:10.3982/ECTA15722](https://doi.org/10.3982/ECTA15722).

Borovickova, K., and Shimer, R. (2017; February 2020 manuscript version). High Wage Workers Work for High Wage Firms. NBER Working Paper 24074. [doi:10.3386/w24074](https://doi.org/10.3386/w24074).

Crippa, F. (2025). Identification, Estimation, and Inference in Two-Sided Interaction Models. Manuscript supplied to the project, Section 2.2.

Gibbons, R., Katz, L. F., Lemieux, T., and Parent, D. (2002). Comparative Advantage, Learning, and Sectoral Wage Determination. NBER Working Paper 8889. Published in *Journal of Labor Economics* 23(4), 681-723 (2005). [doi:10.3386/w8889](https://doi.org/10.3386/w8889).

Kline, P., Saggio, R., and Solvsten, M. (2020). Leave-Out Estimation of Variance Components. *Econometrica*, 88(5), 1859-1898. [doi:10.3982/ECTA16410](https://doi.org/10.3982/ECTA16410).
