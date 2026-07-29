"""Run a deterministic population-level pilot without finite-sample noise."""

from __future__ import annotations

from loo_sim import compute_population_truth, generate_population


def main() -> None:
    designs = [
        {
            "name": "additive-independent",
            "rank": 0,
            "common_sorting": 0.0,
            "interaction_sorting": 0.0,
        },
        {
            "name": "additive-common-sorting",
            "rank": 0,
            "common_sorting": 1.0,
            "interaction_sorting": 0.0,
        },
        {
            "name": "rank1-independent",
            "rank": 1,
            "singular_values": (0.75,),
            "common_sorting": 0.0,
            "interaction_sorting": 0.0,
        },
        {
            "name": "rank1-interaction-sorting",
            "rank": 1,
            "singular_values": (0.75,),
            "common_sorting": 0.0,
            "interaction_sorting": 1.0,
        },
    ]

    header = (
        "design",
        "Q_F",
        "H_F",
        "rho_H",
        "C_assign",
        "C_ab",
        "C_hh",
        "rho_BS",
    )
    print(" | ".join(f"{item:>26}" for item in header))
    print("-" * 235)

    for index, design in enumerate(designs):
        name = str(design.pop("name"))
        dgp = generate_population(
            n_workers=80,
            n_firms=30,
            seed=1000 + index,
            **design,
        )
        truth = compute_population_truth(dgp.schedule, dgp.assignment)
        values = (
            name,
            truth.q_f,
            truth.h_f,
            truth.rho_h,
            truth.c_assign,
            truth.c_ab,
            truth.c_hh,
            truth.bs_correlation,
        )
        print(
            f"{values[0]:>26} | "
            + " | ".join(f"{value:>26.6f}" for value in values[1:])
        )


if __name__ == "__main__":
    main()
