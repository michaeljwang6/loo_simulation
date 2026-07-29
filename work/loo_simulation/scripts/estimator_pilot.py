"""Deterministic one-replication pilot for the first PyTwoWay comparisons."""

from __future__ import annotations

from dataclasses import asdict
import json

import numpy as np

from loo_sim import (
    compute_population_truth,
    compute_procedure_targets,
    generate_population,
    sample_panel,
    select_low_rank_bic,
)
from loo_sim.pytwoway_estimators import estimate_bs20, estimate_fe_kss


def run_scenario(name: str, *, rank: int, singular_values: tuple[float, ...]):
    population = generate_population(
        n_workers=80,
        n_firms=10,
        rank=rank,
        singular_values=singular_values,
        common_sorting=0.6,
        interaction_sorting=0.6 if rank else 0.0,
        seed=701,
    )
    targets = compute_procedure_targets(
        population.schedule,
        population.assignment,
    )
    truth = targets.project
    panel = sample_panel(
        population,
        n_periods=7,
        redraw_probability=0.75,
        error_sd=0.5,
        seed=702,
    )
    fe_kss = estimate_fe_kss(panel, exact=True)
    bs20 = estimate_bs20(panel)
    rank_selection = select_low_rank_bic(
        panel,
        candidate_ranks=(0, 1),
        n_starts=3,
        seed=704,
    )
    oracle_rank_plugin = rank_selection.estimates[
        rank_selection.candidate_ranks.index(rank)
    ]
    analysis_truth = compute_population_truth(
        population.schedule[
            np.ix_(
                oracle_rank_plugin.worker_ids,
                oracle_rank_plugin.firm_ids,
            )
        ],
        oracle_rank_plugin.assignment,
    )

    return {
        "scenario": name,
        "population_truth": {
            "project_q_f": truth.q_f,
            "project_h_f": truth.h_f,
            "project_rho_h": truth.rho_h,
            "project_c_assign": truth.c_assign,
            "akm_firm_variance": targets.akm.firm_variance,
            "akm_covariance": targets.akm.covariance,
            "bs_covariance": truth.bs_covariance,
            "bs_correlation": truth.bs_correlation,
        },
        "estimand_gaps": {
            "akm_firm_variance_minus_project_q_f": (
                targets.akm_firm_variance_gap
            ),
            "akm_covariance_minus_project_c_assign": (
                targets.akm_covariance_gap
            ),
            "bs_covariance_minus_project_c_assign": (
                targets.bs_covariance_gap
            ),
        },
        "raw_panel": {
            "observations": panel.n_observations,
            "workers": panel.n_workers,
            "firms": panel.n_firms_observed,
            "mover_share": panel.mover_share,
        },
        "project_low_rank_plugin": {
            "label": oracle_rank_plugin.label,
            "oracle_rank": rank,
            "bic_selected_rank": rank_selection.selected_rank,
            "bic_by_rank": dict(
                zip(
                    rank_selection.candidate_ranks,
                    rank_selection.bic_values,
                )
            ),
            "sample": asdict(oracle_rank_plugin.sample),
            "converged": oracle_rank_plugin.converged,
            "functionally_stable": (
                oracle_rank_plugin.functionally_stable
            ),
            "near_optimal_starts": (
                oracle_rank_plugin.near_optimal_starts
            ),
            "start_objectives": oracle_rank_plugin.start_objectives,
            "singular_values": (
                oracle_rank_plugin.singular_values.tolist()
            ),
            "estimates": {
                "q_f": oracle_rank_plugin.functionals.q_f,
                "h_f": oracle_rank_plugin.functionals.h_f,
                "rho_h": oracle_rank_plugin.functionals.rho_h,
                "c_assign": oracle_rank_plugin.functionals.c_assign,
            },
            "analysis_sample_truth": {
                "q_f": analysis_truth.q_f,
                "h_f": analysis_truth.h_f,
                "rho_h": analysis_truth.rho_h,
                "c_assign": analysis_truth.c_assign,
            },
            "schedule_estimation_error": {
                "q_f": (
                    oracle_rank_plugin.functionals.q_f
                    - analysis_truth.q_f
                ),
                "h_f": (
                    oracle_rank_plugin.functionals.h_f
                    - analysis_truth.h_f
                ),
                "c_assign": (
                    oracle_rank_plugin.functionals.c_assign
                    - analysis_truth.c_assign
                ),
            },
        },
        "fe_kss": asdict(fe_kss),
        "bs20": asdict(bs20),
    }


def main() -> None:
    results = [
        run_scenario("additive", rank=0, singular_values=()),
        run_scenario("rank_one", rank=1, singular_values=(0.75,)),
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
