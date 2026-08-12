import numpy as np

from loo_sim.truth import compute_population_truth


def test_additive_schedule_reduces_to_akm_objects() -> None:
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
    truth = compute_population_truth(schedule, assignment)

    expected_firm_variance = np.sum(
        truth.firm_weights
        * (
            truth.firm_main
            - np.sum(truth.firm_weights * truth.firm_main)
        )
        ** 2
    )
    assert np.allclose(truth.interaction, 0.0)
    assert np.isclose(truth.h_f, 0.0)
    assert np.isclose(truth.q_f, expected_firm_variance)
    assert np.isclose(truth.c_assign, truth.c_ab)
    assert np.isclose(truth.c_ah, 0.0)
    assert np.isclose(truth.c_bh, 0.0)
    assert np.isclose(truth.c_hh, 0.0)


def test_large_additive_schedule_uses_scale_aware_centering_check() -> None:
    """Do not reject a valid decomposition because of roundoff scale."""

    rng = np.random.default_rng(12345)
    shape = (50, 30)
    raw_assignment = rng.random(shape)
    assignment = raw_assignment / raw_assignment.sum()
    schedule = 1e6 * (
        rng.normal(size=(shape[0], 1))
        + rng.normal(size=(1, shape[1]))
    )

    truth = compute_population_truth(schedule, assignment)

    schedule_scale = np.max(np.abs(schedule))
    assert np.max(np.abs(truth.interaction)) <= 1e-12 * schedule_scale
    assert truth.h_f <= 1e-20 * schedule_scale**2


def test_rank_one_interaction_satisfies_dispersion_identity() -> None:
    worker_main = np.array([-0.8, -0.1, 0.4, 0.5])
    firm_main = np.array([-0.6, 0.2, 0.4])
    worker_factor = np.array([-1.0, -0.5, 0.25, 1.25])
    firm_factor = np.array([-0.75, 0.0, 0.75])

    schedule = (
        1.5
        + worker_main[:, None]
        + firm_main[None, :]
        + 0.7 * worker_factor[:, None] * firm_factor[None, :]
    )
    assignment = np.full(schedule.shape, 1.0 / schedule.size)
    truth = compute_population_truth(schedule, assignment)

    assert truth.h_f > 0
    assert 0 < truth.rho_h < 1
    assert np.isclose(
        truth.q_f,
        np.var(truth.firm_main) + 0.5 * truth.h_f,
    )


def test_bs_types_use_observed_conditional_assignment() -> None:
    schedule = np.array(
        [
            [1.0, 2.0],
            [3.0, 5.0],
        ]
    )
    assignment = np.array(
        [
            [0.40, 0.10],
            [0.10, 0.40],
        ]
    )
    truth = compute_population_truth(schedule, assignment)

    assert np.allclose(truth.bs_worker_type, [1.2, 4.6])
    assert np.allclose(truth.bs_firm_type, [1.4, 4.4])
    assert truth.bs_correlation > 0
