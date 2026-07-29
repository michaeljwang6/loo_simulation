# Simulation contract

## 1. Population interface

Every replication starts from:

- a complete systematic wage schedule \(M=(m_{ij})\);
- an observed assignment law \(P^{obs}=(P^{obs}_{ij})\);
- observed marginals \(p_i=\sum_jP^{obs}_{ij}\) and
  \(q_j=\sum_iP^{obs}_{ij}\);
- an observation process that produces a finite worker--firm panel and wage
  noise.

Unless a design explicitly says otherwise, the project product reference is
\(P_I\otimes P_J\), using the observed assignment marginals.

## 2. Project ground truths

The canonical product-weighted decomposition is

\[
m_{ij}=\mu+a_i+b_j+h_{ij},
\]

where

\[
\mu=\sum_{ij}p_iq_jm_{ij},\quad
a_i=\sum_jq_jm_{ij}-\mu,\quad
b_j=\sum_ip_im_{ij}-\mu.
\]

The headline truths are

\[
Q_F=\sum_i p_i\operatorname{Var}_{q}(m_{iJ}),
\]

\[
H_F=\sum_{jk}q_jq_k
 \operatorname{Var}_{p}(m_{Ij}-m_{Ik}),
\]

\[
\rho_H=\frac{H_F}{2Q_F},
\]

and

\[
C^w_{\mathrm{assign}}
=\frac12\left\{
\operatorname{Var}_{P^{obs}}(m_{IJ})
-\operatorname{Var}_{p\otimes q}(m_{IJ})
\right\}.
\]

The truth engine must verify

\[
Q_F=\operatorname{Var}_{q}(b_J)+\frac12H_F.
\]

Under additive separability, \(h_{ij}=0\), so

\[
H_F=0,\qquad
Q_F=\operatorname{Var}_{q}(b_J),\qquad
C^w_{\mathrm{assign}}=\operatorname{Cov}_{P^{obs}}(a_I,b_J).
\]

## 3. Borovičková--Shimer truth

The BS20 observed-match wage types are

\[
\lambda_i^{BS}
=\sum_j\frac{P^{obs}_{ij}}{p_i}m_{ij},
\qquad
\mu_j^{BS}
=\sum_i\frac{P^{obs}_{ij}}{q_j}m_{ij}.
\]

The native BS20 truth is

\[
\rho_{BS}
=\operatorname{Corr}_{P^{obs}}
  (\lambda_I^{BS},\mu_J^{BS}).
\]

This is not relabeled as \(C^w_{\mathrm{assign}}(M)\). The BS covariance is
the assignment contribution of the additive replacement schedule
\(s_{ij}=\lambda_i^{BS}+\mu_j^{BS}-\bar m_{obs}\).

## 4. Estimator labels

The initial comparison uses:

- **Project truth:** functionals of the known complete \(M\);
- **Project oracle low-rank:** functionals using true rank or true factors;
- **Project low-rank plug-in:** functionals of an estimated low-rank
  completion, without the unfinished LOO correction;
- **FE/KSS:** PyTwoWay fixed-effects estimates and heteroskedastic
  leave-out correction;
- **BLM:** PyTwoWay grouped estimator, with oracle-group and estimated-group
  variants where feasible;
- **BS20:** PyTwoWay Borovičková--Shimer estimator.

No output may call the project plug-in estimate the "full LOO estimator."

### Initial low-rank plug-in implementation

For a fixed interaction rank \(r\), the initial project estimator solves

\[
\min_{\alpha,\psi,U,V}
\sum_{(i,j)\in E}
n_{ij}\left(
\bar Y_{ij}-\alpha_i-\psi_j-u_i'v_j
\right)^2.
\]

It uses alternating weighted least squares, a spectral initialization plus
perturbed starts, and match counts \(n_{ij}\) as weights. Factor centering and
SVD normalization fix a convenient gauge without changing the completed wage
schedule. The current simulation sets \(X_{ij}'\delta=0\); more generally the
input outcome to this implementation must first be residualized for observed
covariates.

Positive-rank fits default to the largest connected support core with at least
\(r+2\) distinct matches per worker and firm. This is one observation more
than the conditional-regression minimum and is a finite-sample stability
restriction. Every output reports the retained support, rectangle count,
convergence, near-optimal-start functional dispersion, and completed-schedule
functionals.

The optional BIC rank selector uses a common support sample across candidate
ranks and is an exploratory simulation device, not part of the unfinished LOO
theory. No quadratic bias correction is applied.

## 5. Two evaluation panels

### Native-estimand performance

Each procedure is compared with its own population target.

### Harmonized economic comparison

When a procedure produces a fitted wage schedule, the project functionals
may also be applied to that fitted schedule. Report

\[
\widehat\theta^{(p)}-\theta_{\mathrm{project}}
=
\left(\widehat\theta^{(p)}-\theta_p^\star\right)
+
\left(\theta_p^\star-\theta_{\mathrm{project}}\right),
\]

separating estimation error from the estimand/model gap.

BS20 is excluded from schedule-completion comparisons unless an explicit
bridge restriction is imposed.

## 5.1 AKM/KSS population target under nonadditivity

Without additional controls, the native AKM/KSS population effects are the
assignment-weighted additive projection

\[
(\mu^{AKM},\alpha^{AKM},\psi^{AKM})
=\arg\min_{\mu,\alpha,\psi}
\sum_{ij}P^{obs}_{ij}
\left(m_{ij}-\mu-\alpha_i-\psi_j\right)^2,
\]

with \(p\)-weighted worker effects and \(q\)-weighted firm effects normalized
to zero. The native PyTwoWay comparison moments are

\[
V_\psi^{AKM}=\operatorname{Var}_{q}(\psi_J^{AKM}),
\qquad
C_{\psi\alpha}^{AKM}
=\operatorname{Cov}_{P^{obs}}
(\psi_J^{AKM},\alpha_I^{AKM}).
\]

Under additive separability,

\[
V_\psi^{AKM}=Q_F,\qquad
C_{\psi\alpha}^{AKM}=C^w_{\mathrm{assign}}.
\]

Under independent assignment with nonadditivity, the AKM projection agrees
with the canonical product-weighted main effects and

\[
V_\psi^{AKM}=Q_F-\frac12H_F.
\]

For general nonadditive sorting, the AKM projection can absorb conditional
means of the interaction. Its native moments must therefore be reported
separately from the project objects.

The projection defines the comparison target under misspecification; it does
not assert that the KSS correction retains its additive-model unbiasedness
when the omitted interaction behaves like a persistent match component.

## 6. Initial DGP ladder

1. Additive schedule, independent assignment.
2. Additive schedule, sorting on common components.
3. Rank-one free-factor schedule, independent assignment.
4. Rank-one free-factor schedule, sorting on common components.
5. Rank-one free-factor schedule, sorting on interaction gains.
6. BLM-style grouped schedule.
7. Rank-two truth fitted at ranks zero, one, and two.

Observation-process violations and graph stress tests are added only after
the deterministic pilot passes.
