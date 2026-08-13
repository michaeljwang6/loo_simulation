# DGP-by-estimator Monte Carlo: methods and results

> **Archived result set.** The numerical results below use the superseded
> full-history BLM conversion, which classified a worker as a stayer only if
> the worker remained at one firm throughout all ten periods. Static BLM
> instead defines mover status over one declared pair of periods. The code now
> uses periods 0 and 1, and a new 25,000-worker, 5,000-firm cluster run is
> specified with forty panel periods in
> `configs/dgp_estimator_matrix_cluster.json`. Until that run is
> complete, the BLM completion rates and cross-DGP BLM results below should not
> be treated as current evidence. The non-BLM archived results are unchanged.

This report asks a deliberately symmetric question: what happens when each of four procedures is applied to data from each of five worker-firm wage models? The answer is not a single winner. KSS and BLM perform well when their own structures are correct, while nonadditivity, rank selection, sparse support, and differences in estimands explain the cross-model reversals.

Every numerical statement below is derived from the support-audited merged output with configuration fingerprint `3e8ab3b57d35190989c47117f139b414b9f656db8a933d6c5d1df6d2aa5fbcc4`. The support audit reconstructs the original panels and BLM clustering from their saved seeds; it does not refit a successful model. Equations attributed to papers are cited. Every other equation is labeled as a definition or derived below.

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

| Symbol | Exact definition | Interpretation |
|---|---|---|
| $P_{ij}$ | observed population match probability | Assignment law used to sample firms |
| $p_i,q_j$ | marginals of $P_{ij}$ | Worker and firm population weights |
| $m_{ij}$ | $E[Y_{ijt}\mid i,j]$ | Complete systematic wage schedule |
| $h_{ij}$ | interaction in $m_{ij}=\mu+a_i+b_j+h_{ij}$ under weights $p_iq_j$ | Nonadditive match component |
| $Q_F$ | $E_{pq}[(m_{ij}-E_q[m_{ij}\mid i])^2]$ | Firm-side schedule variation |
| $H_F$ | $2E_{pq}[h_{ij}^2]$ | Twice the interaction variance |
| $\rho_H$ | $H_F/(2Q_F)$ when $Q_F>0$ | Interaction share of firm-side variation |
| $C_{\mathrm{assign}}$ | $\{\operatorname{Var}_P(m)-\operatorname{Var}_{pq}(m)\}/2$ | Variance contribution of observed assignment |

The project procedure is the current low-rank plug-in with BIC rank selection, without the unfinished LOO correction. KSS estimates additive-projection variance components. BLM estimates a discrete worker-type by firm-class wage surface. BS20 estimates moments of worker and firm wage types. Because these objects are not identical under nonadditivity, Sections 4.3 and 4.4 distinguish accuracy for a procedure's native target from accuracy for the common project target.

| Procedure | Settings used in every DGP |
|---|---|
| KSS-HE | PyTwoWay heteroskedastic correction with fe_exact=False; approximate rather than exact trace and leverage calculations |
| BLM | Estimated firm groups; 2 worker types; 3 firm classes; CDF resolution 10; 4 initializations; retain best 2; at most 250 iterations; threshold $10^{-6}$ |
| BS20 | Weighted PyTwoWay estimator after strongly connected, spell-level, no-return cleaning |
| Project plug-in | 3 starts for positive rank; tolerance $10^{-6}$; at most 300 iterations; common support core requiring degree 3 for candidate set $\{0,1\}$ and degree 4 for $\{0,1,2\}$ |

The BIC candidates are $\{0,1\}$ for AKM and grouped BLM and $\{0,1,2\}$ for Crippa, continuous low-rank, and GKLP. These choices matter because a candidate set that excludes the true rank cannot recover it.

## 3. Evaluation rules

The report uses four execution statuses: **pass or completed**, **returned with warning**, **unsupported sample**, and **estimator failure**. They do not have the same strength across procedures:

| Procedure | Pass or completed | Returned with warning | Unsupported sample |
|---|---|---|---|
| KSS-HE and BS20 | PyTwoWay returned an estimate | Not used | Not used |
| BLM | Every stayer class and mover class-pair is observed, and both likelihood paths avoid a one-step decline larger than $10^{-4}$ | Complete support, but a likelihood path violates that monotonicity check | At least one of the 3 stayer classes or 9 mover class-pairs is absent after cleaning and clustering |
| Project plug-in | The selected fit converges; for positive rank, at least 2 of 3 starts have objective and functional spreads within the exact tolerances stated below | A finite fit fails at least one of those diagnostics | Not used |

An **unsupported sample** is rejected before BLM fitting, whereas an **estimator failure** means an admissible sample reached the estimator but no value was returned. The $10^{-4}$ likelihood tolerance and the project multi-start thresholds are engineering warning rules, not statistical critical values. A pass therefore does not establish correct specification, rank, or low bias. RMSE includes every value returned from an admissible sample; the pass-only RMSE is a separately labeled sensitivity calculation.

BLM preparation is: drop workers who return to a previous firm; collapse spells; estimate three firm clusters from empirical wage CDFs using K-means; form mover and stayer event-study samples; check all required observed cells; run four mover starts; retain the best two; select by connectedness; and finally fit the stayer model. The support check is necessary because PyTwoWay 0.3.21 forms probability arrays of size $2\times 3^2$ for movers and $2\times3$ for stayers. A missing firm-class cell otherwise becomes a label-dependent reshape error or a zero probability row.

For continuous DGPs, BLM receives no oracle type labels. Product-marginal wage means define fixed equal-count reference groups only for scoring its fitted cell table. The BLM functionals are also compared with the full project truth, so discretization error remains part of the comparison.

## 4. Results

### 4.1 Completion and numerical warnings

Each cell reports pass or completed / returned with warning / unsupported / failed. For KSS and BS20, the first number means completion only. The project fit passes its convergence and multi-start checks in every additive and Crippa replication, but in only 6% of grouped-BLM replications. Crippa shows why this is only a numerical check: all project fits pass even though BIC selects the wrong rank in every replication.

| DGP | KSS | BLM | BS20 | Project plug-in |
|---|---:|---:|---:|---:|
| AKM | 100 / 0 / 0 / 0 | 43 / 0 / 57 / 0 | 100 / 0 / 0 / 0 | 100 / 0 / 0 / 0 |
| Crippa/Tukey | 100 / 0 / 0 / 0 | 57 / 0 / 43 / 0 | 100 / 0 / 0 / 0 | 100 / 0 / 0 / 0 |
| BLM types | 100 / 0 / 0 / 0 | 46 / 1 / 53 / 0 | 100 / 0 / 0 / 0 | 6 / 94 / 0 / 0 |
| Low-rank factors | 100 / 0 / 0 / 0 | 46 / 0 / 54 / 0 | 100 / 0 / 0 / 0 | 78 / 22 / 0 / 0 |
| GKLP | 100 / 0 / 0 / 0 | 50 / 0 / 50 / 0 | 100 / 0 / 0 / 0 | 94 / 6 / 0 / 0 |

![Figure 1. Pass or completion, returned-with-warning, unsupported samples, and estimator failures. KSS and BS20 use completion only.](reports/dgp_estimator_matrix/figures/status-matrix.png)

The corrected classification finds 257 unsupported BLM samples and 0 estimator failures among 500 attempts. The exact reasons are:

| DGP | BLM status | Exact reason | Count |
|---|---|---|---:|
| AKM | unsupported | At least one mover class-pair absent | 2 |
| AKM | unsupported | At least one stayer firm class absent | 53 |
| AKM | unsupported | No stayer events | 2 |
| BLM types | unsupported | At least one stayer firm class absent | 53 |
| Crippa/Tukey | unsupported | At least one stayer firm class absent | 42 |
| Crippa/Tukey | unsupported | No stayer events | 1 |
| GKLP | unsupported | At least one stayer firm class absent | 49 |
| GKLP | unsupported | Both stayer classes and mover class-pairs absent | 1 |
| Low-rank factors | unsupported | At least one mover class-pair absent | 4 |
| Low-rank factors | unsupported | At least one stayer firm class absent | 49 |
| Low-rank factors | unsupported | Both stayer classes and mover class-pairs absent | 1 |

The high unsupported rate follows from the observation design. As an approximation, the probability of remaining at the same firm in one later period is $0.6+0.4/18=0.622$. Remaining at the same firm through all nine later periods therefore has probability $0.622^9=0.014$, or about four workers out of 300. After returners are dropped, those few stayers must cover all three firm classes. This calculation is derived from the configured redraw rule; the exact audit uses the realized cleaned samples.

### 4.2 Correctly specified benchmarks

The expected benchmarks work. Under the additive AKM DGP, KSS-HE estimates $Q_F$ with bias +0.0032 and RMSE 0.048; its assignment-covariance bias is -0.0001. Under the grouped BLM DGP, the 47 admissible returned BLM fits have biases -0.0054 for $Q_F$, -0.0031 for $H_F$, and +0.0025 for $C_{\mathrm{assign}}$. These conditional results describe the supported subsample, not all 100 replications; feasibility is therefore part of BLM's performance rather than a footnote.

### 4.3 Cross-DGP accuracy

Figure 2 compares each reported or model-implied output with the common project truth, using admissible returned estimates only. `N/A` means the procedure has no defensible mapping to that project object; it does not mean zero or failure.

| Procedure | $Q_F$ | $H_F$ | $\rho_H$ | $C_{\mathrm{assign}}$ |
|---|---:|---:|---:|---:|
| KSS-HE | Estimated | 0, imposed | 0, imposed when $Q_F>0$ | Estimated additive covariance |
| BLM | Reported | Reported | Reported | Reported |
| BS20 | N/A | N/A | N/A | N/A; native covariance shown in Section 4.4 |
| Project plug-in | Reported | Reported | Reported | Reported |

KSS now appears under $H_F$ and $\rho_H$ as a structural benchmark. Its fitted schedule is additive, so $h_{ij}=0$, $H_F=0$, and $\rho_H=0$ whenever fitted $Q_F>0$. These are restrictions imposed by the model, not interaction estimates learned from the data. BS20 is no longer placed under $C_{\mathrm{assign}}$: it estimates $\operatorname{Cov}_P(\lambda_i,\mu_j)$, a different object defined in Section 4.4.

The `0*` cells require a different explanation. Under the AKM DGP, the project truth has $h_{ij}=0$, hence $H_F=\rho_H=0$. KSS imposes the same zeros, and BIC selects rank zero in all 100 project replications. The project raw RMSEs are 6.59e-31 and 3.31e-31; these are floating-point residue displayed as structural zeros. They test whether rank selection creates a false interaction, not ordinary precision in estimating a nonzero quantity. The log color scale is used because positive-rank warning cases generate errors above one million in some cells.

![Figure 2. RMSE against the common project truth among admissible returned estimates. N/A means no defensible mapping; 0* marks AKM structural zeros.](reports/dgp_estimator_matrix/figures/common-target-rmse.png)

KSS remains accurate for its additive target, but its error relative to project $Q_F$ expands when nonadditivity changes the target. BLM is highly accurate on the grouped DGP when it returns. The project procedure performs well for GKLP conditional on a diagnostic pass, but performs poorly for Crippa because BIC always removes the true rank-one interaction. Under the rank-two DGP, BIC always selects rank one and some returned fits have very large functional errors.

### 4.4 Estimand differences: native versus project targets

Section 4.3 measures each reported output against the project truth. That error combines two distinct components. For procedure $p$ with native target $\theta_p$ and project target $\theta_{\mathrm{proj}}$:

$$
\widehat{\theta}_p-\theta_{\mathrm{proj}}=(\widehat{\theta}_p-\theta_p)+(\theta_p-\theta_{\mathrm{proj}}).
$$

The first term is estimation error for the procedure's own object. The second is an estimand difference. This section reports the second term, averaged over all 100 simulated populations; it contains no estimator sampling error.

For KSS firm variance, the sign under independent assignment follows directly from the product-weighted ANOVA. Since $m_{ij}=\mu+a_i+b_j+h_{ij}$ has zero weighted margins,

$$
Q_F=\operatorname{Var}_q(b_j)+E_{pq}[h_{ij}^2]=\operatorname{Var}_q(b_j)+H_F/2.
$$

Under independent assignment, the population additive projection recovers $b_j$. Therefore KSS native firm variance minus project $Q_F$ equals $-H_F/2\leq0$. Under sorted assignment, the additive projection can absorb part of $h_{ij}$, so the negative sign is no longer a theorem; the plotted magnitudes then depend on the DGP.

| DGP | KSS $Q_F$ gap | KSS covariance gap | BS20 covariance gap |
|---|---:|---:|---:|
| AKM | +0.0000 | +0.0000 | +0.416 |
| Crippa/Tukey | -0.394 | -0.0028 | +0.824 |
| BLM types | -0.752 | +0.014 | +0.622 |
| Low-rank factors | -1.127 | -0.017 | +0.459 |
| GKLP | -0.741 | -0.080 | +0.452 |

![Figure 3. Population estimand differences, not estimator bias: native procedure target minus project target.](reports/dgp_estimator_matrix/figures/native-project-target-gaps.png)

For example, under Crippa the mean KSS native firm-variance target is 1.169, while project $Q_F$ is 1.562. The target gap is -0.394. KSS's native-target bias is only -0.015, but its RMSE against project $Q_F$ is 0.524. Much of the common-target error is therefore disagreement about the object, not failure to estimate the KSS object.

For BS20, define the native wage types $\lambda_i=E_P[m_{ij}\mid i]$ and $\mu_j=E_P[m_{ij}\mid j]$. BS20 targets $C_{\mathrm{BS}}=\operatorname{Cov}_P(\lambda_i,\mu_j)$; the project instead targets $C_{\mathrm{assign}}=\{\operatorname{Var}_P(m)-\operatorname{Var}_{pq}(m)\}/2$. The simulation reads PyTwoWay's `cov(lambda, mu)` directly and compares its population target with $C_{\mathrm{assign}}$ only in this estimand-gap section. No algebraic reconstruction of a BS20 value is used.

The positive BS20 gap has a derivation in the additive case. If $m_{ij}=a_i+b_j$, let $Tb(i)=E[b_J\mid I=i]$ and $T^*a(j)=E[a_I\mid J=j]$. Expanding the two conditional wage types yields

$$
C_{\mathrm{BS}}-C_{\mathrm{assign}}=\lVert Tb\rVert^2+\lVert T^*a\rVert^2+\langle Tb,TT^*a\rangle\geq0.
$$

The inequality follows from Cauchy--Schwarz and the contraction property of conditional expectation. For a general nonadditive schedule there is no universal sign, so the positive gaps outside AKM are features of the simulated sorting laws rather than a theorem.

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
