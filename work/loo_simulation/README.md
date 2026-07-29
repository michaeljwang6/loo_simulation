# LOO numerical experiments

This directory contains the simulation framework for comparing the LOO
project's schedule-based objects with established two-way estimators.

The project model is

\[
Y_{ij}=X_{ij}'\delta+\alpha_i+\psi_j+u_i'\Lambda v_j+\varepsilon_{ij}.
\]

The full LOO bias correction is still under development. Consequently, the
first numerical experiments label the project estimator as a **low-rank
plug-in estimator**, not as the final LOO estimator.

The implementation is deliberately staged:

1. verify population truths and algebraic identities;
2. generate observed worker--firm panels from a known full schedule;
3. add project low-rank plug-in estimates;
4. wrap PyTwoWay's FE/KSS, BLM, and Borovičková--Shimer estimators;
5. run Monte Carlo experiments and decompose estimation error from estimand
   differences.

See `SIMULATION_CONTRACT.md` for the exact estimands and comparison rules.

The first estimator smoke test is:

```powershell
& .\.venv311\Scripts\python.exe scripts\estimator_pilot.py
```

It reports exact population truths alongside PyTwoWay's AKM plug-in,
homoskedastic correction, heteroskedastic KSS correction, and weighted BS20
moments. It also runs the project's match-count-weighted alternating
least-squares estimator at the oracle rank and with exploratory BIC rank
selection. Each estimator reports the size of its own cleaned analysis sample.
Under nonadditivity the pilot reports the population assignment-weighted AKM
projection, estimator-specific targets, and the resulting estimand gaps. The
project output remains a low-rank plug-in benchmark; the unfinished project
LOO correction is not implemented or labeled as complete.

The grouped-model BLM pilot is:

```powershell
& .\.venv311\Scripts\python.exe scripts\blm_pilot.py
```

It compares PyTwoWay BLM with oracle firm classes and wage-distribution
clusters against the known stationary worker-type by firm-class mean table.
Latent worker labels, and estimated firm labels where applicable, are aligned
only for simulation evaluation.

## Monte Carlo runner

The configuration-driven runner executes the DGP ladder replication by
replication and isolates estimator failures rather than aborting the study.
Start with the one-replication representative pilot:

```powershell
& .\.venv311\Scripts\python.exe scripts\run_monte_carlo.py `
  --config configs\quick_pilot.json `
  --output results\quick_pilot
```

After inspecting those diagnostics, the complete seven-rung configuration is:

```powershell
& .\.venv311\Scripts\python.exe scripts\run_monte_carlo.py `
  --config configs\full_ladder.json `
  --output results\full_ladder
```

The full configuration declares 100 replications, but `--replications` and
`--seed` can override those two fields without editing the JSON. Results are
written as:

- `records.csv`: one estimate--target--error row per scalar metric;
- `attempts.csv`: convergence, failure, stability, and retained-sample
  diagnostics for every estimator attempt;
- `attempt_summary.csv`: estimator-level success, instability, and failure
  rates, including estimators that never return a value;
- `summary.csv`: bias, sampling standard deviation, RMSE, and mean retained
  sample size by scenario, estimator, metric, and target type;
- `config.json` and `metadata.json`: the resolved reproducibility contract.

Low-rank plug-in rows distinguish the complete population target from the
truth on the retained analysis sample. FE/KSS rows distinguish their native
assignment-weighted additive-projection target from the project target.
BS20 rows distinguish its native type-moment target, and grouped BLM rows
report both native cell means and grouped project functionals. Numerical
values from unstable fits remain auditable and are counted as unstable rather
than silently deleted.

Before changing the baseline panel length or positive-rank support
restriction, run the low-rank-only support calibration:

```powershell
& .\.venv311\Scripts\python.exe scripts\run_monte_carlo.py `
  --config configs\support_calibration.json `
  --output results\support_calibration
```

It compares rank-one and rank-two oracle-rank fits at 7, 10, 15, and 20
periods. The output is intentionally separate from the main comparison:
support restrictions can stabilize matrix completion while changing the
retained worker population, so both the full-population and retained-sample
targets must be inspected.

The paired interaction-sorting calibration uses common random numbers across
sorting strengths:

```powershell
& .\.venv311\Scripts\python.exe scripts\run_monte_carlo.py `
  --config configs\interaction_sorting_calibration.json `
  --output results\interaction_sorting_calibration
```

The production ladder uses `interaction_sorting=0.4` and 10 periods for the
rank-one interaction-sorting baseline. The `0.8` design remains in this
calibration as an intentional weak-support stress test. The rank-two rung uses
15 periods.

## Resumable production execution

Run the 100-replication ladder as independently saved shards rather than one
long serial process:

```powershell
& .\.venv311\Scripts\python.exe scripts\run_monte_carlo_shards.py `
  --config configs\full_ladder.json `
  --output-root results\full_ladder_production `
  --shard-count 20 `
  --workers 2 `
  --resume
```

This assigns five global replication indices to each shard and runs at most
two shards concurrently. Every completed shard is saved immediately. Rerunning
the command with `--resume` validates configuration fingerprints and
replication indices before reusing completed work. Once all shards finish, the
launcher strictly validates full coverage, non-overlap, and record uniqueness,
then writes the combined tables to `merged`.

Shards can also be run individually with `run_monte_carlo.py --shard-index`
and `--shard-count`. To merge independently launched shards:

```powershell
$shards = Get-ChildItem results\full_ladder_production `
  -Directory -Filter "shard_*"
& .\.venv311\Scripts\python.exe scripts\merge_monte_carlo.py `
  --inputs $shards.FullName `
  --output results\full_ladder_production\merged
```
