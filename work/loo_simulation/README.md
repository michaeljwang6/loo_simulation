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
moments. Each estimator also reports the size of its own cleaned analysis
sample. The project output remains a population truth/plug-in benchmark; the
unfinished project LOO correction is not implemented or labeled as complete.
