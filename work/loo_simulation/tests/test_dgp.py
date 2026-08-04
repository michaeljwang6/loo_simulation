import numpy as np
import pytest

from loo_sim.dgp import (
    generate_akm_population,
    generate_crippa_population,
    generate_gklp_population,
    generate_grouped_population,
    generate_population,
)
from loo_sim.targets import (
    compute_blm_evaluation_groups,
    compute_blm_grouped_target,
)
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


def test_named_akm_dgp_is_additive() -> None:
    dgp = generate_akm_population(
        n_workers=20,
        n_firms=8,
        common_sorting=0.8,
        seed=21,
    )
    truth = compute_population_truth(dgp.schedule, dgp.assignment)

    assert dgp.singular_values.size == 0
    assert np.allclose(dgp.interaction, 0.0)
    assert np.allclose(truth.interaction, 0.0)
    assert np.isclose(truth.h_f, 0.0)


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


def test_crippa_dgp_has_exact_tukey_interaction() -> None:
    beta_0 = -0.65
    dgp = generate_crippa_population(
        n_workers=40,
        n_firms=15,
        beta_0=beta_0,
        common_sorting=0.4,
        interaction_sorting=0.3,
        seed=41,
    )
    expected_interaction = (
        beta_0
        * dgp.worker_main[:, None]
        * dgp.firm_main[None, :]
    )
    weighted_interaction = (
        np.sqrt(dgp.worker_weights)[:, None]
        * dgp.interaction
        * np.sqrt(dgp.firm_weights)[None, :]
    )
    spectrum = np.linalg.svd(weighted_interaction, compute_uv=False)

    assert np.allclose(dgp.interaction, expected_interaction)
    assert np.allclose(
        dgp.schedule,
        dgp.worker_main[:, None]
        + dgp.firm_main[None, :]
        + expected_interaction,
    )
    assert np.isclose(spectrum[0], abs(beta_0))
    assert np.allclose(spectrum[1:], 0.0, atol=1e-12)


def test_gklp_dgp_has_exact_rank_one_comparative_advantage() -> None:
    dgp = generate_gklp_population(
        n_workers=40,
        n_firms=15,
        ability_correlation=0.4,
        firm_type_correlation=0.3,
        productivity_shock_sd=0.6,
        seed=42,
    )
    weighted_interaction = (
        np.sqrt(dgp.worker_weights)[:, None]
        * dgp.interaction
        * np.sqrt(dgp.firm_weights)[None, :]
    )
    spectrum = np.linalg.svd(weighted_interaction, compute_uv=False)

    assert np.allclose(
        dgp.schedule,
        dgp.worker_main[:, None]
        + dgp.firm_main[None, :]
        + dgp.interaction,
    )
    assert np.isclose(spectrum[0], 1.0)
    assert np.allclose(spectrum[1:], 0.0, atol=1e-12)
    assert compute_population_truth(dgp.schedule, dgp.assignment).h_f > 0


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


def test_blm_evaluation_groups_are_defined_for_continuous_dgp() -> None:
    dgp = generate_population(
        n_workers=24,
        n_firms=12,
        rank=2,
        singular_values=(1.0, 0.4),
        seed=60,
    )
    groups = compute_blm_evaluation_groups(
        dgp.schedule,
        dgp.assignment,
        n_worker_types=2,
        n_firm_types=3,
    )
    target = compute_blm_grouped_target(
        dgp.schedule,
        dgp.assignment,
        groups.worker_groups,
        groups.firm_groups,
    )

    assert np.array_equal(
        np.bincount(groups.worker_groups),
        np.array([12, 12]),
    )
    assert np.array_equal(
        np.bincount(groups.firm_groups),
        np.array([4, 4, 4]),
    )
    assert target.cell_means.shape == (2, 3)
    assert target.within_cell_variance > 0


def test_blm_evaluation_groups_order_additive_main_effects() -> None:
    dgp = generate_akm_population(
        n_workers=12,
        n_firms=9,
        seed=61,
    )
    groups = compute_blm_evaluation_groups(
        dgp.schedule,
        dgp.assignment,
        n_worker_types=2,
        n_firm_types=3,
    )

    assert np.all(
        dgp.worker_main[groups.worker_groups == 0].max()
        <= dgp.worker_main[groups.worker_groups == 1].min()
    )
    for lower_group in range(2):
        assert np.all(
            dgp.firm_main[groups.firm_groups == lower_group].max()
            <= dgp.firm_main[groups.firm_groups == lower_group + 1].min()
        )
