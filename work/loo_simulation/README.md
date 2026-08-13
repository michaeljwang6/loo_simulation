# LOO numerical experiments

This directory contains the simulation framework for comparing the LOO
project's schedule-based objects with established two-way estimators.

The project model is

\[
Y_{ij}=X_{ij}'\delta+\alpha_i+\psi_j+u_i'\Lambda v_j+\varepsilon_{ij}.
\]

The full LOO bias correction is still under development. Consequently, the
first numerical experiments label the project estimator as the **low-rank
plug-in without LOO correction**, not as the final LOO estimator.

The implementation is deliberately staged:

1. verify population truths and algebraic identities;
2. generate observed worker--firm panels from a known full schedule;
3. add project low-rank plug-in estimates;
4. wrap PyTwoWay's FE/KSS, BLM, and Borovickova--Shimer estimators;
5. run Monte Carlo experiments and decompose estimation error from estimand
   differences.

See `SIMULATION_CONTRACT.md` for the exact estimands and comparison rules.

## Full DGP-by-estimator matrix

The new experiment implements the complete whiteboard cross-product: additive
AKM, Crippa/Tukey, discrete-type BLM, continuous low-rank-factor, and GKLP DGPs
are each estimated by KSS, BLM, Borovickova--Shimer 2020, and the current
project low-rank plug-in. The DGP specification and estimator settings are
separate in the configuration, so no estimator is restricted to its preferred
DGP.

See `DGP_ESTIMATOR_MATRIX.md` for the exact distributions, formulas, target
comparisons, and commands for running locally or on a SLURM cluster without
using Codex credits. Start with:

```powershell
& .\.venv311\Scripts\python.exe scripts\run_monte_carlo.py `
  --config configs\dgp_estimator_matrix_pilot.json `
  --output results\dgp_estimator_matrix_pilot
```

The corrected runner estimates BLM firm classes from the full panel, then
constructs the static BLM sample from declared periods 0 and 1. A stayer is a
worker at the same firm in those two periods; later moves do not change that
classification. The runner checks that this two-period sample observes all
three stayer classes and all nine mover class-pairs. Panels that fail this
condition are labeled `unsupported`; estimator failures are a separate status.

## Next cluster production run

The next production design is declared separately from the archived small
run in `configs/dgp_estimator_matrix_cluster.json`. It uses 25,000 workers,
5,000 firms, forty periods, and 100 replications for every DGP. The panel sets
`redraw_probability=0.2`: in each transition a worker mechanically retains
the current firm with probability 0.8 and otherwise redraws from the
worker-specific assignment distribution. Because a redraw can select the
same firm, the realized probability of an unchanged firm is slightly above
0.8. The lower end of the requested worker range is deliberate: it supplies
about 5,000 period-0-to-period-1 redraws while using half as many dense
worker-by-firm cells as a 50,000-worker design.

The panel length follows two cluster preflights. Ten periods produced no common
four-degree support core for rank two. Twenty periods restored that core and
eliminated support failures, but every positive-rank fit was numerically
unstable: fifteen of sixteen distinct fits reached the 300-iteration limit,
and the remaining fit had only one near-optimal start. With retention 0.8, the
expected number of distinct firms per worker (ignoring rare redraws to the same
firm) is only $1+19(0.2)=4.8$ at twenty periods, barely above the worker
intercept and two factor coordinates in a rank-two fit. Forty periods raises
this expectation to $1+39(0.2)=8.8$. The next preflight therefore changes
only panel length: it keeps the estimator's support rule, three starts,
tolerance, and 300-iteration cap fixed. BLM classification still uses only
periods 0 and 1; all forty periods are used to estimate its firm classes.

Submit the run as 50 resumable shards and automatically merge after all array
tasks succeed:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[estimators]"
bash -n scripts/slurm_dgp_estimator_matrix_cluster.sh \
  scripts/slurm_merge_dgp_estimator_matrix_cluster.sh \
  scripts/submit_dgp_estimator_matrix_cluster.sh
sbatch --array=0 scripts/slurm_dgp_estimator_matrix_cluster.sh
```

Use that single shard as the cluster-specific memory and wall-time preflight.
After it completes successfully, submit the full array and dependent merge:

```bash
bash scripts/submit_dgp_estimator_matrix_cluster.sh
```

The array specification is `0-49%4`: all 50 shards are created, with at most
four running simultaneously. With 100 replications, each shard owns two
global replication indices and runs all five DGPs for those indices. The
completed pilot shard is reused because every task passes `--resume`. The
template requests 128 GB for each array task because the present truth engine
and low-rank estimator still materialize dense worker-by-firm arrays. Cluster
partitions and account directives are site-specific and should be added to
the two Slurm scripts before submission.

For the production run completed before this support check was added, create a
support-audited result without refitting any estimator, then regenerate the
tables and figures:

```powershell
& .\.venv311\Scripts\python.exe scripts\reclassify_blm_support.py `
  --input results\dgp_estimator_matrix\merged `
  --output results\dgp_estimator_matrix_support_audited\merged
& .\.venv311\Scripts\python.exe scripts\report_dgp_estimator_matrix.py `
  --input results\dgp_estimator_matrix_support_audited\merged
```

The standalone LaTeX manuscript is `DGP_ESTIMATOR_RESULTS.tex`. Compile it
with Tectonic or another current LaTeX engine:

```powershell
tectonic DGP_ESTIMATOR_RESULTS.tex --outdir output\pdf
Copy-Item output\pdf\DGP_ESTIMATOR_RESULTS.pdf `
  output\pdf\dgp_estimator_matrix_report.pdf -Force
```

The official Tectonic installation instructions are at
<https://tectonic-typesetting.github.io/install.html>.

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
- `attempts.csv`: convergence, unsupported-sample, failure, stability, and retained-sample
  diagnostics for every estimator attempt;
- `attempt_summary.csv`: estimator-level success, instability, unsupported, and failure
  rates, including estimators that never return a value;
- `summary.csv`: bias, error standard deviation, Monte Carlo standard error
  of the bias, RMSE, and mean retained sample size by scenario, estimator,
  metric, and target type, including returned values classified as unstable;
- `conditional_stable_summary.csv`: the separately labeled robustness table
  computed only from successful, functionally stable returned values while
  retaining the full attempt counts;
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

The five-replication computational gate and its interpretation are recorded
in `PREPRODUCTION_AUDIT.md`. In particular, rank selection and functional
stability are separate diagnostics: a selected positive rank can still yield
unstable plug-in functionals under weak support.

The completed 100-replication findings, including native-estimand,
population-project, unconditional, and conditional-on-stability comparisons,
are documented in `PRODUCTION_RESULTS.md`.

Generate the paper-ready PDF/PNG figures and CSV/LaTeX tables with:

```powershell
& .\.venv311\Scripts\python.exe -m pip install -e ".[report]"
$env:MPLCONFIGDIR = (Resolve-Path .).Path + "\.mplconfig"
& .\.venv311\Scripts\python.exe scripts\report_results.py
```

The generated bundle is written to `reports\production`.

The integrated methods-and-results note is `SIMULATION_WRITEUP.md`. Render its
verified PDF with:

```powershell
& .\.venv311\Scripts\python.exe -m pip install -e ".[writeup]"
& .\.venv311\Scripts\python.exe scripts\build_simulation_writeup_pdf.py
```

The PDF is written to `output\pdf\loo_simulation_writeup.pdf`.

Shards can also be run individually with `run_monte_carlo.py --shard-index`
and `--shard-count`. To merge independently launched shards:

```powershell
$shards = Get-ChildItem results\full_ladder_production `
  -Directory -Filter "shard_*"
& .\.venv311\Scripts\python.exe scripts\merge_monte_carlo.py `
  --inputs $shards.FullName `
  --output results\full_ladder_production\merged
```
