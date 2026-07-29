import numpy as np

from loo_sim.dgp import generate_population
from loo_sim.truth import compute_population_truth


def test_assignment_balancing_preserves_declared_marginals() -> None:
    dgp = generate_population(
        n_workers=30,
        n_firms=12,
        rank=1,
        common_sorting=1.0,
        interaction_sorting=0.5,
        seed=10,
    )

    assert np.allclose(dgp.assignment.sum(axis=1), dgp.worker_weights)
    assert np.allclose(dgp.assignment.sum(axis=0), dgp.firm_weights)
    assert np.isclose(dgp.assignment.sum(), 1.0)


def test_rank_zero_is_additive() -> None:
    dgp = generate_population(
        n_workers=20,
        n_firms=8,
        rank=0,
        common_sorting=0.8,
        seed=20,
    )
    truth = compute_population_truth(dgp.schedule, dgp.assignment)

    assert np.allclose(truth.interaction, 0.0)
    assert np.isclose(truth.h_f, 0.0)
    assert np.isclose(truth.c_assign, truth.c_ab)


def test_rank_one_free_factor_is_nonadditive() -> None:
    dgp = generate_population(
        n_workers=25,
        n_firms=10,
        rank=1,
        singular_values=(0.75,),
        interaction_sorting=1.0,
        seed=30,
    )
    truth = compute_population_truth(dgp.schedule, dgp.assignment)

    assert truth.h_f > 0
    assert truth.a_h > 0
    assert not np.isclose(truth.c_assign - truth.c_ab, 0.0)
    assert truth.rho_h > 0
