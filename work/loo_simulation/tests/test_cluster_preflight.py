from dataclasses import replace
from pathlib import Path
import subprocess
import sys

from loo_sim import (
    EstimatorConfig,
    MonteCarloConfig,
    ScenarioConfig,
    run_monte_carlo,
    save_monte_carlo_results,
)


def _saved_preflight(tmp_path: Path, *, unstable_bic: bool) -> Path:
    config = MonteCarloConfig(
        scenarios=(
            ScenarioConfig(
                name="additive_gate_test",
                population_kind="free_factor",
                population_kwargs={
                    "n_workers": 18,
                    "n_firms": 5,
                    "rank": 0,
                    "singular_values": (),
                },
                panel_kwargs={
                    "n_periods": 8,
                    "redraw_probability": 0.8,
                    "error_sd": 0.1,
                },
                true_rank=0,
                plugin_ranks=(0,),
            ),
        ),
        replications=1,
        seed=8675309,
        estimators=EstimatorConfig(
            run_low_rank=True,
            run_bic=True,
            run_fe_kss=False,
            run_bs20=False,
            run_blm=False,
            low_rank_n_starts=1,
        ),
    )
    result = run_monte_carlo(config)
    expanded_scenario = replace(
        config.scenarios[0],
        plugin_ranks=(0, 1),
    )
    expanded_config = replace(
        config,
        scenarios=(expanded_scenario,),
    )
    rank_zero = next(
        attempt
        for attempt in result.attempts
        if attempt.estimator == "project_plugin_r0"
    )
    allowed_overrank = replace(
        rank_zero,
        estimator="project_plugin_r1",
        status="unstable",
        message="deliberately over-ranked test warning",
    )
    attempts = list(result.attempts) + [allowed_overrank]
    if unstable_bic:
        attempts = [
            replace(
                attempt,
                status="unstable",
                message="unstable selected fit",
            )
            if attempt.estimator == "project_plugin_bic"
            else attempt
            for attempt in attempts
        ]
    audited = replace(
        result,
        config=expanded_config,
        attempts=tuple(attempts),
    )
    return save_monte_carlo_results(audited, tmp_path / "preflight")


def _run_gate(result_directory: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/audit_cluster_preflight.py",
            str(result_directory),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_preflight_gate_allows_only_overrank_warning(tmp_path: Path) -> None:
    completed = _run_gate(
        _saved_preflight(tmp_path, unstable_bic=False)
    )

    assert completed.returncode == 0
    assert "Permitted deliberately over-ranked warnings: 1" in completed.stdout
    assert "PREFLIGHT PASSED" in completed.stdout


def test_preflight_gate_rejects_unstable_bic(tmp_path: Path) -> None:
    completed = _run_gate(
        _saved_preflight(tmp_path, unstable_bic=True)
    )

    assert completed.returncode == 1
    assert "unexpected instability" in completed.stdout
    assert "PREFLIGHT FAILED" in completed.stdout
