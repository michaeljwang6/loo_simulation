import numpy as np
import pytest

from loo_sim.dgp import generate_grouped_population, generate_population
from loo_sim.targets import compute_blm_grouped_target
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


def test_requested_singular_values_equal_weighted_interaction_spectrum() -> None:
    requested = np.array([1.5, 0.4])
    dgp = generate_population(
        n_workers=40,
        n_firms=20,
        rank=2,
        singular_values=tuple(requested),
        seed=40,
    )

    worker_gram = dgp.worker_factors.T @ (
        dgp.worker_weights[:, None] * dgp.worker_factors
    )
    firm_gram = dgp.firm_factors.T @ (
        dgp.firm_weights[:, None] * dgp.firm_factors
    )
    weighted_interaction = (
        np.sqrt(dgp.worker_weights)[:, None]
        * dgp.interaction
        * np.sqrt(dgp.firm_weights)[None, :]
    )
    actual = np.linalg.svd(weighted_interaction, compute_uv=False)[:2]

    assert np.allclose(worker_gram, np.eye(2))
    assert np.allclose(firm_gram, np.eye(2))
    assert np.allclose(actual, requested)


def test_rank_and_singular_value_validation() -> None:
    with pytest.raises(ValueError, match="rank cannot exceed"):
        generate_population(n_workers=3, n_firms=4, rank=3)

    with pytest.raises(ValueError, match="non-increasing"):
        generate_population(
            n_workers=10,
            n_firms=8,
            rank=2,
            singular_values=(0.5, 1.0),
        )


def test_grouped_dgp_has_exact_type_class_schedule_and_targets() -> None:
    dgp = generate_grouped_population(
        n_workers=60,
        n_firms=15,
        n_worker_types=3,
        n_firm_types=3,
        rank=2,
        singular_values=(1.0, 0.4),
        seed=50,
    )
    grouped = compute_blm_grouped_target(
        dgp.schedule,
        dgp.assignment,
        dgp.worker_groups,
        dgp.firm_groups,
    )
    individual = compute_population_truth(dgp.schedule, dgp.assignment)

    assert np.allclose(
        dgp.schedule,
        dgp.cell_means[
            dgp.worker_groups[:, None],
            dgp.firm_groups[None, :],
        ],
    )
    assert np.allclose(grouped.cell_means, dgp.cell_means)
    assert np.isclose(grouped.within_cell_variance, 0.0)
    assert np.isclose(
        grouped.project_functionals.q_f,
        individual.q_f,
    )
    assert np.isclose(
        grouped.project_functionals.h_f,
        individual.h_f,
    )
    assert np.isclose(
        grouped.project_functionals.c_assign,
        individual.c_assign,
    )
