# Five-replication pre-production audit

## Purpose

This is a computational gate, not a substantive Monte Carlo conclusion. It
uses five global replications of `configs/full_ladder.json` to check that the
calibrated seven-rung ladder, estimator wrappers, shard checkpoints, and
strict merge all work together before the 100-replication run.

The validated post-fix output is in the ignored local directory
`results/preproduction_5_fe_cleaning_fix/merged`. Its resolved configuration
fingerprint is
`6ba55a3d9905475a53cb3fddfa2eefb181ea4860a5e3804b7b2268f91b31805e`.

## Execution gate

- Five disjoint shards covered global replication indices 0 through 4.
- Every shard produced 324 scalar records and 52 estimator attempts.
- The complete merge contains 1,620 records and 260 attempts.
- No estimator attempt failed: 237 were successful and 23 returned values
  explicitly classified as unstable.
- All five rank-two FE/KSS samples retained 1,200 observations, 80 workers,
  and 10 firms.

The original pre-production run exposed an FE/KSS cleaning error in four of
five rank-two replications. The wrapper dropped every worker who ever returned
to a previous firm before constructing the leave-out-spell connected set. In
longer panels that rule erased the mover network. PyTwoWay's FE Monte Carlo
keeps return spells when constructing this sample. Aligning the wrapper with
that behavior restored the complete sample and finite FE, KSS-HO, and KSS-HE
estimates in all five replications. The exact previously failing random seed
is now a regression test.

## Diagnostics that must remain visible

The exploratory BIC selector chose the declared rank in all five replications
of every scenario: rank zero in the two additive designs, rank one in the
four rank-one/grouped designs, and rank two in the rank-two design.

Correct rank selection did not imply stable functionals:

- the deliberately overfit rank-one estimator was unstable in 5/5 additive
  common-sorting and 4/5 additive-independent replications;
- the rank-one plug-in and its identical BIC-selected fit were unstable in
  1/5 rank-one-independent replications;
- the rank-one plug-in and BIC-selected fit were unstable in 5/5 grouped-BLM
  replications;
- the rank-two plug-in and its identical BIC-selected fit were unstable in
  1/5 rank-two replications.

These returned values remain in the unconditional performance summary. They
produce very large unconditional errors in the weak-support cases and must
not be silently deleted. A later conditional-on-stability table is a separate
robustness analysis, not a replacement for procedure-level performance.

## Five-replication signals

These numbers only guide the production audit:

- FE/KSS completed on every design. Relative to the native AKM target, firm
  variance RMSE ranged from about 0.05 to 0.17 and worker--firm covariance
  RMSE from about 0.04 to 0.10 across the seven scenarios. With five
  replications there is no reliable ranking among FE, KSS-HO, and KSS-HE.
- In the grouped DGP, both BLM variants had mean aligned cell-mean RMSE 0.053.
  Their grouped project-functional RMSEs were 0.052 for `q_f`, 0.102 for
  `h_f`, 0.010 for `c_assign`, and 0.020 for `rho_h`.
- The project plug-in was well behaved in the calibrated rank-one
  common-sorting and interaction-sorting designs. Independent rank-one and
  grouped designs exposed weak-support instability, which is a procedure
  outcome to measure in the full experiment.
- BS20 was comparatively close to its own native target in the independent
  and grouped pilots and showed larger negative errors under sorting and
  nonadditivity. Five draws are insufficient for a substantive bias claim.

## Readiness decision

The execution framework passes the pre-production gate. The full run should
preserve:

1. unconditional summaries including unstable returned values;
2. failure and instability rates beside every performance table;
3. native-estimand and population-project comparisons as separate panels;
4. a separately labeled conditional-on-stability robustness table.

No output from this experiment is the project's unfinished full LOO
estimator. The project procedure remains labeled **low-rank plug-in without
LOO correction**.
