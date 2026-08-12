import numpy as np
import pytest
import loo_sim.targets as target_module

from loo_sim import (
    compute_akm_population_target,
    compute_procedure_targets,
    generate_population,
)


def test_additive_akm_target_equals_project_additive_components() -> None:
    worker = np.array([-1.0, 0.0, 1.0])
    firm = np.array([-0.5, 0.5])
    schedule = 2.0 + worker[:, None] + firm[None, :]
    assignment = np.array(
        [
            [0.25, 0.05],
            [0.15, 0.15],
            [0.05, 0.35],
        ]
    )

    targets = compute_procedure_targets(schedule, assignment)

    assert np.allclose(targets.akm.residual, 0.0)
    assert np.allclose(
        targets.akm.firm_effect,
        targets.project.firm_main,
    )
    assert np.allclose(
        targets.akm.worker_effect,
        targets.project.worker_main,
    )
    assert np.isclose(
        targets.akm.firm_variance,
        targets.project.q_f,
    )
    assert np.isclose(
        targets.akm.covariance,
        targets.project.c_assign,
    )
    assert np.isclose(targets.akm_firm_variance_gap, 0.0)
    assert np.isclose(targets.akm_covariance_gap, 0.0)


def test_independent_assignment_akm_gap_is_half_nonadditivity() -> None:
    population = generate_population(
        n_workers=30,
        n_firms=12,
        rank=1,
        singular_values=(0.75,),
        common_sorting=0.0,
        interaction_sorting=0.0,
        seed=801,
    )
    targets = compute_procedure_targets(
        population.schedule,
        population.assignment,
    )

    assert np.allclose(
        targets.akm.worker_effect,
        targets.project.worker_main,
    )
    assert np.allclose(
        targets.akm.firm_effect,
        targets.project.firm_main,
    )
    assert np.allclose(
        targets.akm.residual,
        targets.project.interaction,
    )
    assert np.isclose(
        targets.akm.firm_variance,
        targets.project.q_f - 0.5 * targets.project.h_f,
    )
    assert np.isclose(
        targets.akm_firm_variance_gap,
        -0.5 * targets.project.h_f,
    )
    assert np.isclose(targets.akm.covariance, 0.0, atol=1e-12)
    assert np.isclose(targets.project.c_assign, 0.0, atol=1e-12)


def test_sorted_nonadditive_target_satisfies_weighted_normal_equations() -> None:
    population = generate_population(
        n_workers=25,
        n_firms=9,
        rank=1,
        singular_values=(0.8,),
        common_sorting=0.7,
        interaction_sorting=0.6,
        seed=802,
    )
    target = compute_akm_population_target(
        population.schedule,
        population.assignment,
    )

    assert np.allclose(
        np.sum(population.assignment * target.residual, axis=1),
        0.0,
    )
    assert np.allclose(
        np.sum(population.assignment * target.residual, axis=0),
        0.0,
    )
    assert target.residual_variance > 0
    assert np.isfinite(target.firm_variance)
    assert np.isfinite(target.covariance)


def test_akm_projection_matches_direct_weighted_least_squares() -> None:
    rng = np.random.default_rng(803)
    schedule = rng.normal(size=(6, 4))
    assignment = rng.exponential(size=schedule.shape)
    assignment /= assignment.sum()
    target = compute_akm_population_target(schedule, assignment)

    worker = np.repeat(np.arange(6), 4)
    firm = np.tile(np.arange(4), 6)
    design = np.zeros((24, 1 + 5 + 3))
    design[:, 0] = 1.0
    design[worker > 0, worker[worker > 0]] = 1.0
    firm_offset = 6
    design[firm > 0, firm_offset + firm[firm > 0] - 1] = 1.0
    sqrt_weight = np.sqrt(assignment.ravel())
    coefficients = np.linalg.lstsq(
        design * sqrt_weight[:, None],
        schedule.ravel() * sqrt_weight,
        rcond=None,
    )[0]
    direct_fit = (design @ coefficients).reshape(schedule.shape)

    assert np.allclose(target.fitted_schedule, direct_fit)


def test_disconnected_assignment_has_no_unique_akm_moment_target() -> None:
    schedule = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    assignment = np.array(
        [
            [0.5, 0.0],
            [0.0, 0.5],
        ]
    )

    with pytest.raises(ValueError, match="connected assignment support"):
        compute_akm_population_target(schedule, assignment)


def test_matrix_free_akm_projection_matches_explicit_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    population = generate_population(
        n_workers=35,
        n_firms=14,
        rank=1,
        singular_values=(0.8,),
        common_sorting=0.7,
        interaction_sorting=0.6,
        seed=804,
    )
    explicit = compute_akm_population_target(
        population.schedule,
        population.assignment,
    )
    monkeypatch.setattr(target_module, "_EXPLICIT_AKM_MAX_CELLS", 0)
    matrix_free = compute_akm_population_target(
        population.schedule,
        population.assignment,
    )

    assert np.allclose(
        matrix_free.worker_effect,
        explicit.worker_effect,
        atol=1e-9,
    )
    assert np.allclose(
        matrix_free.firm_effect,
        explicit.firm_effect,
        atol=1e-9,
    )
    assert np.allclose(
        matrix_free.residual,
        explicit.residual,
        atol=1e-9,
    )
