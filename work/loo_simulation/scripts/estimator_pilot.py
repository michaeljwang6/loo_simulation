"""Deterministic one-replication pilot for the first PyTwoWay comparisons."""

from __future__ import annotations

from dataclasses import asdict
import json

from loo_sim import compute_population_truth, generate_population, sample_panel
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
    truth = compute_population_truth(population.schedule, population.assignment)
    panel = sample_panel(
        population,
        n_periods=7,
        redraw_probability=0.75,
        error_sd=0.5,
        seed=702,
    )
    fe_kss = estimate_fe_kss(panel, exact=True)
    bs20 = estimate_bs20(panel)

    return {
        "scenario": name,
        "population_truth": {
            "project_q_f": truth.q_f,
            "project_h_f": truth.h_f,
            "project_rho_h": truth.rho_h,
            "project_c_assign": truth.c_assign,
            "bs_covariance": truth.bs_covariance,
            "bs_correlation": truth.bs_correlation,
        },
        "raw_panel": {
            "observations": panel.n_observations,
            "workers": panel.n_workers,
            "firms": panel.n_firms_observed,
            "mover_share": panel.mover_share,
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
