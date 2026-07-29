# Production Monte Carlo results

## Scope and reproducibility

This report summarizes the calibrated seven-rung experiment in
`configs/full_ladder.json` with 100 global replications. The complete local
output is in the ignored directory `results/full_ladder_production/merged`.
Its resolved configuration fingerprint is
`e03e0c242706502fcd58f20c1cc52c4deeb234f48c981a4885e13157d6ac2a09`.

The experiment does **not** implement the project's unfinished full LOO
estimator. Throughout, the project procedure is the **low-rank plug-in without
LOO correction**.

The 20 independently saved shards passed strict checks for complete coverage
of replications 0--99, non-overlap, configuration equality, and record
uniqueness. The merged output contains:

- 32,400 scalar estimate--target records;
- 5,200 estimator-attempt diagnostics;
- 4,773 successful attempts;
- 427 attempts that returned values but were classified as unstable;
- zero hard failures.

The 427 unstable rows are procedure-attempt diagnostics. Fixed-rank and
BIC-selected procedures are reported separately even when BIC selects the
same fitted rank, so they should not be interpreted as 427 independent
numerical fits.

`summary.csv` is the primary unconditional procedure-performance table and
includes values returned by unstable fits. `conditional_stable_summary.csv`
is a separately labeled robustness table that uses only successful,
functionally stable returned values while retaining the full attempt and
instability counts. Bias Monte Carlo standard errors are computed from the
replication-level estimate-minus-target errors.

## Main findings

1. **Additive separability:** FE/KSS and the project procedure have the same
   population targets for firm variance and worker--firm assignment
   covariance. They perform similarly under independent assignment. Under
   common-component sorting, the low-rank plug-in without LOO correction has
   visible downward bias, especially for assignment covariance.
2. **Nonadditive separability:** FE/KSS is approximately unbiased for its
   native assignment-weighted additive projection. That native firm-variance
   target is 0.85--1.13 below project \(Q_F\) in the calibrated nonadditive
   designs. Accurate AKM/KSS estimation therefore does not imply recovery of
   project \(Q_F\).
3. **Project plug-in:** Conditional on a stable fit, the rank-one and rank-two
   plug-ins recover all four project functionals with moderate error in the
   free-factor designs. However, 6--12% of BIC attempts are unstable in those
   nonadditive designs. The resulting unconditional RMSE is dominated by
   explosive returned values. This stability risk is part of procedure
   performance and cannot be removed from the headline result.
4. **Grouped DGP:** BLM is clearly better suited to its native grouped design.
   The low-rank BIC procedure is unstable in 88% of attempts and performs
   poorly even in the 12 stable attempts. Oracle and estimated-group BLM
   recover both cell means and grouped project functionals well.
5. **BS20:** BS20 returns results in every replication but exhibits systematic
   negative finite-sample bias relative to its own native correlation target,
   especially under sorting and rank-two nonadditivity. It is not relabeled
   as estimating the project assignment covariance.

## Stability and rank selection

| Scenario | BIC selected rank | Correct-rank rate | BIC unstable rate |
|---|---:|---:|---:|
| Additive, independent | 0 in 100/100 | 100% | 0% |
| Additive, common sorting | 0 in 100/100 | 100% | 0% |
| Rank 1, independent | 1 in 100/100 | 100% | 11% |
| Rank 1, common sorting | 1 in 100/100 | 100% | 12% |
| Rank 1, interaction sorting | 1 in 100/100 | 100% | 6% |
| Grouped BLM | 0 in 10/100; 1 in 90/100 | 90% | 88% |
| Rank 2 | 1 in 4/100; 2 in 96/100 | 96% | 9% |

Correct rank selection is not a stability certificate. The fixed rank-one
plug-in was unstable in 80% of additive-independent and 79% of
additive-common-sorting attempts, confirming that BIC appropriately avoided
the deliberately overspecified interaction. In the grouped design, however,
the fixed rank-one fit was unstable in 98% of attempts and BIC could not make
the positive-rank problem reliable.

## Additive designs

Bias is followed by its Monte Carlo standard error in parentheses, then RMSE.
The project BIC procedure selected rank zero and was stable in every draw.
KSS-HE is shown as the representative heteroskedastic leave-out correction;
AKM plug-in and KSS-HO have nearly identical RMSE in these designs.

| Scenario and procedure | Firm variance / \(Q_F\) | Assignment covariance |
|---|---:|---:|
| Independent, project plug-in | +0.013 (0.008); 0.082 | -0.005 (0.006); 0.059 |
| Independent, KSS-HE | +0.007 (0.008); 0.082 | -0.001 (0.006); 0.059 |
| Common sorting, project plug-in | -0.035 (0.009); 0.099 | -0.072 (0.006); 0.094 |
| Common sorting, KSS-HE | +0.007 (0.009); 0.087 | -0.002 (0.004); 0.043 |

The rank-zero project fit imposes \(H_F=\rho_H=0\), which is correct in these
two designs. The sorted-design bias is therefore a finite-sample plug-in
issue, not an estimand gap. It is precisely the kind of bias the future
project LOO correction is intended to address, but this experiment does not
claim that the unfinished correction would eliminate it.

## Nonadditive free-factor designs

The following table compares unconditional BIC RMSE with the separately
reported stable-only robustness RMSE. `Stable n` is out of 100 attempts.

| Scenario | Unstable | Unconditional \(Q_F/H_F\) RMSE | Stable \(Q_F\) | Stable \(H_F\) | Stable \(C_{\rm assign}\) | Stable \(\rho_H\) | Stable n |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rank 1, independent | 11% | 823 / 1,626 | 0.296 | 0.513 | 0.183 | 0.051 | 89 |
| Rank 1, common sorting | 12% | 26,560 / 51,170 | 0.217 | 0.432 | 0.201 | 0.064 | 88 |
| Rank 1, interaction sorting | 6% | 1,038 / 2,057 | 0.142 | 0.255 | 0.104 | 0.034 | 94 |
| Rank 2 | 9% | 849 / 1,679 | 0.154 | 0.263 | 0.122 | 0.028 | 91 |
| Grouped BLM | 88% | 38,100 / 75,450 | 3.586 | 7.431 | 4.814 | 0.478 | 12 |

The extreme unconditional values are not transcription errors. A small number
of functionally unstable completions generate very large project functionals.
The estimator diagnoses those fits as unstable, but a full procedure must
still account for their frequency. Stable-only results describe performance
after conditioning on a diagnostic that uses the returned fit.

Stable-only bias is not zero in the free-factor designs. For example:

| Scenario | \(Q_F\) bias (MCSE) | \(H_F\) bias (MCSE) | \(C_{\rm assign}\) bias (MCSE) | \(\rho_H\) bias (MCSE) |
|---|---:|---:|---:|---:|
| Rank 1, independent | +0.180 (0.025) | +0.326 (0.042) | -0.093 (0.017) | +0.032 (0.004) |
| Rank 1, common sorting | +0.092 (0.021) | +0.286 (0.035) | -0.150 (0.014) | +0.045 (0.005) |
| Rank 1, interaction sorting | +0.075 (0.013) | +0.147 (0.022) | -0.056 (0.009) | +0.017 (0.003) |
| Rank 2 | +0.089 (0.013) | +0.172 (0.021) | -0.070 (0.011) | +0.015 (0.003) |

These are conditional-on-stability biases and should not be interpreted as
unconditional bias estimates.

## AKM/KSS estimand gap under nonadditivity

KSS-HE bias relative to the native AKM projection is small compared with its
Monte Carlo standard error. The important difference is the population
target itself.

| Scenario | Native AKM firm-variance target minus project \(Q_F\) | KSS-HE native bias (MCSE) |
|---|---:|---:|
| Grouped BLM | -0.900 | -0.001 (0.015) |
| Rank 1, independent | -1.000 | -0.010 (0.015) |
| Rank 1, common sorting | -0.986 | -0.009 (0.017) |
| Rank 1, interaction sorting | -0.847 | -0.004 (0.013) |
| Rank 2 | -1.127 | +0.001 (0.009) |

Thus KSS-HE estimates its declared native firm-variance target accurately
while being far below project \(Q_F\) by construction. The native-minus-project
covariance gaps happen to be small in the free-factor designs (between -0.013
and +0.002) and are +0.044 in the grouped design. That is a feature of these
DGPs, not a general equivalence under nonadditivity. FE/KSS also does not
deliver project \(H_F\) or \(\rho_H\).

Across all seven scenarios, KSS-HE native firm-variance RMSE ranges from 0.082
to 0.166 and native covariance RMSE from 0.042 to 0.067. The homoskedastic and
heteroskedastic corrections remove the small average AKM plug-in bias in
several designs, but their RMSE differences are modest at these sample sizes.

## Grouped BLM comparison

| Procedure | Stable attempts | Mean aligned cell RMSE | \(Q_F\) RMSE | \(H_F\) RMSE | \(C_{\rm assign}\) RMSE | \(\rho_H\) RMSE |
|---|---:|---:|---:|---:|---:|---:|
| BLM, oracle firm groups | 100/100 | 0.051 | 0.056 | 0.085 | 0.013 | 0.017 |
| BLM, estimated firm groups | 99/100 | 0.051 | 0.092 | 0.110 | 0.019 | 0.014 |
| Project BIC plug-in, stable only | 12/100 | n/a | 3.586 | 7.431 | 4.814 | 0.478 |

For both BLM variants, grouped-project bias is indistinguishable from zero at
Monte Carlo precision. Estimated firm grouping modestly increases \(Q_F\),
\(H_F\), and assignment-covariance RMSE but retains strong recovery.

## BS20 native performance

BS20 succeeded in all 700 scenario-replication attempts. Its native
worker--firm correlation bias is:

| Scenario | Bias (MCSE) | RMSE |
|---|---:|---:|
| Additive, independent | -0.149 (0.009) | 0.174 |
| Additive, common sorting | -0.212 (0.008) | 0.225 |
| Rank 1, independent | -0.174 (0.013) | 0.216 |
| Rank 1, common sorting | -0.161 (0.009) | 0.185 |
| Rank 1, interaction sorting | -0.198 (0.010) | 0.221 |
| Grouped BLM | -0.049 (0.006) | 0.078 |
| Rank 2 | -0.404 (0.012) | 0.421 |

The wrapper's BS20-valid sample is smaller because return spells and
insufficient worker/firm matches are removed. For example, it retains on
average 415 of 560 observations in the additive-independent design, 379 of
560 under additive common sorting, and 434 of 1,200 in the rank-two design.
FE/KSS retains all observations in those same simulated panels. Sample
selection is part of the implemented BS20 procedure and must accompany the
performance comparison.

## Interpretation and next use

The experiment supports three separate reporting panels:

1. **Native-estimand performance:** KSS is reliable for the AKM projection,
   BLM is reliable for the grouped target, and BS20 has notable downward
   finite-sample bias for its native match-type correlation.
2. **Project-estimand performance:** The low-rank plug-in can recover
   \(Q_F,H_F,C_{\rm assign},\rho_H\) when its completion is stable, but it has
   non-negligible conditional plug-in bias and severe unconditional tail risk.
3. **Estimand gaps:** Under nonadditivity, the AKM/KSS firm-variance target is
   materially different from project \(Q_F\), independently of estimation
   quality.

The natural next research step is not to relabel the current plug-in as LOO.
It is to add the project's completed leave-worker-out correction when its
formula is finalized, then rerun the same frozen DGP ladder and compare
whether it reduces plug-in bias without worsening stability or sample
retention.
