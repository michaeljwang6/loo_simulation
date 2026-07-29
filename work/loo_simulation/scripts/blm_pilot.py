"""Deterministic oracle-group and estimated-group BLM pilot."""

from __future__ import annotations

from dataclasses import asdict
import json

from loo_sim import (
    compute_blm_grouped_target,
    generate_grouped_population,
    sample_panel,
)
from loo_sim.pytwoway_estimators import (
    align_blm_cell_means,
    estimate_blm,
)


def _compact_result(result, alignment):
    return {
        "variant": result.variant,
        "sample": asdict(result.sample),
        "mover_log_likelihood": result.mover_log_likelihood,
        "stayer_log_likelihood": result.stayer_log_likelihood,
        "connectedness": result.connectedness,
        "mover_min_likelihood_change": (
            result.mover_min_likelihood_change
        ),
        "stayer_min_likelihood_change": (
            result.stayer_min_likelihood_change
        ),
        "mover_likelihood_monotone": (
            result.mover_likelihood_monotone
        ),
        "stayer_likelihood_monotone": (
            result.stayer_likelihood_monotone
        ),
        "stationary_cell_means": (
            result.stationary_cell_means.tolist()
        ),
        "alignment": {
            "worker_permutation": alignment.worker_permutation,
            "firm_permutation": alignment.firm_permutation,
            "rmse": alignment.rmse,
            "max_absolute_error": alignment.max_absolute_error,
        },
    }


def main() -> None:
    population = generate_grouped_population(
        n_workers=300,
        n_firms=18,
        n_worker_types=2,
        n_firm_types=3,
        rank=1,
        singular_values=(1.0,),
        seed=1101,
    )
    panel = sample_panel(
        population,
        n_periods=5,
        redraw_probability=0.35,
        error_sd=0.5,
        seed=1102,
    )
    target = compute_blm_grouped_target(
        population.schedule,
        population.assignment,
        population.worker_groups,
        population.firm_groups,
    )

    oracle = estimate_blm(
        panel,
        n_worker_types=2,
        n_firm_types=3,
        firm_groups=population.firm_groups,
        n_init=4,
        n_best=2,
        seed=1103,
    )
    estimated = estimate_blm(
        panel,
        n_worker_types=2,
        n_firm_types=3,
        n_init=4,
        n_best=2,
        seed=1104,
    )
    oracle_alignment = align_blm_cell_means(
        oracle.stationary_cell_means,
        target.cell_means,
        allow_firm_permutation=False,
    )
    estimated_alignment = align_blm_cell_means(
        estimated.stationary_cell_means,
        target.cell_means,
        allow_firm_permutation=True,
    )

    output = {
        "population_target": {
            "cell_means": target.cell_means.tolist(),
            "within_cell_variance": target.within_cell_variance,
            "project_q_f": target.project_functionals.q_f,
            "project_h_f": target.project_functionals.h_f,
            "project_c_assign": (
                target.project_functionals.c_assign
            ),
        },
        "raw_panel": {
            "observations": panel.n_observations,
            "workers": panel.n_workers,
            "firms": panel.n_firms_observed,
            "mover_share": panel.mover_share,
        },
        "oracle_firm_groups": _compact_result(
            oracle,
            oracle_alignment,
        ),
        "estimated_firm_groups": _compact_result(
            estimated,
            estimated_alignment,
        ),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
