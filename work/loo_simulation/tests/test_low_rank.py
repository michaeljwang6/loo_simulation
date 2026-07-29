import numpy as np
import pytest

from loo_sim import (
    PanelData,
    compute_population_truth,
    fit_low_rank_plugin,
    generate_population,
    select_low_rank_bic,
)


def _complete_panel(
    schedule: np.ndarray,
    *,
    repetitions: int = 1,
    error_sd: float = 0.0,
    seed: int = 1001,
) -> PanelData:
    n_workers, n_firms = schedule.shape
    worker_id = np.repeat(
        np.arange(n_workers),
        n_firms * repetitions,
    )
    firm_id = np.tile(
        np.repeat(np.arange(n_firms), repetitions),
        n_workers,
    )
    period = np.tile(
        np.arange(n_firms * repetitions),
        n_workers,
    )
    systematic = schedule[worker_id, firm_id]
    error = np.random.default_rng(seed).normal(
        scale=error_sd,
        size=systematic.size,
    )
    return PanelData(
        worker_id=worker_id.astype(np.int64),
        firm_id=firm_id.astype(np.int64),
        period=period.astype(np.int64),
        outcome=systematic + error,
        systematic_wage=systematic,
        error=error,
    )


def test_rank_zero_recovers_complete_additive_schedule() -> None:
    worker = np.array([-1.0, -0.2, 0.4, 0.8])
    firm = np.array([-0.6, 0.1, 0.5])
    schedule = 1.5 + worker[:, None] + firm[None, :]
    panel = _complete_panel(schedule)

    estimate = fit_low_rank_plugin(panel, rank=0)

    assert estimate.label == "low-rank plug-in without LOO correction"
    assert estimate.converged
    assert np.allclose(estimate.fitted_schedule, schedule)
    assert np.isclose(estimate.functionals.h_f, 0.0)
    assert np.isclose(estimate.edge_objective, 0.0)


def test_oracle_rank_recovers_complete_rank_one_schedule() -> None:
    population = generate_population(
        n_workers=12,
        n_firms=7,
        rank=1,
        singular_values=(0.8,),
        seed=1002,
    )
    panel = _complete_panel(population.schedule)

    estimate = fit_low_rank_plugin(
        panel,
        rank=1,
        n_starts=3,
        tolerance=1e-12,
        seed=1003,
    )
    truth = compute_population_truth(
        population.schedule,
        np.full(population.schedule.shape, 1.0 / population.schedule.size),
    )

    assert estimate.converged
    assert estimate.functionally_stable
    assert np.allclose(estimate.fitted_schedule, population.schedule, atol=1e-8)
    assert np.allclose(estimate.singular_values, [0.8], atol=1e-8)
    assert np.isclose(estimate.functionals.q_f, truth.q_f, atol=1e-8)
    assert np.isclose(estimate.functionals.h_f, truth.h_f, atol=1e-8)
    assert np.isclose(
        estimate.functionals.c_assign,
        truth.c_assign,
        atol=1e-8,
    )


def test_multiple_starts_report_invariant_functional_stability() -> None:
    population = generate_population(
        n_workers=10,
        n_firms=6,
        rank=1,
        singular_values=(0.6,),
        seed=1004,
    )
    estimate = fit_low_rank_plugin(
        _complete_panel(population.schedule),
        rank=1,
        n_starts=4,
        tolerance=1e-11,
        seed=1005,
    )

    assert len(estimate.start_objectives) == 4
    assert np.ptp(estimate.start_q_f) < 1e-8
    assert np.ptp(estimate.start_h_f) < 1e-8
    assert np.ptp(estimate.start_c_assign) < 1e-8
    assert estimate.functionally_stable
    assert estimate.near_optimal_starts == 4
    assert estimate.sample.rectangles > 0


def test_exploratory_bic_selects_rank_one_in_noiseless_complete_panel() -> None:
    population = generate_population(
        n_workers=12,
        n_firms=7,
        rank=1,
        singular_values=(0.9,),
        seed=1006,
    )
    selection = select_low_rank_bic(
        _complete_panel(population.schedule, repetitions=2),
        candidate_ranks=(0, 1, 2),
        n_starts=3,
        tolerance=1e-11,
        seed=1007,
    )

    assert selection.selected_rank == 1
    assert selection.selected.label == (
        "low-rank plug-in without LOO correction"
    )
    assert len(selection.bic_values) == 3


def test_rank_one_support_core_drops_underidentified_worker() -> None:
    population = generate_population(
        n_workers=7,
        n_firms=5,
        rank=1,
        singular_values=(0.7,),
        seed=1008,
    )
    panel = _complete_panel(population.schedule)
    keep = (panel.worker_id != 0) | (panel.firm_id < 2)
    sparse_panel = PanelData(
        worker_id=panel.worker_id[keep],
        firm_id=panel.firm_id[keep],
        period=panel.period[keep],
        outcome=panel.outcome[keep],
        systematic_wage=panel.systematic_wage[keep],
        error=panel.error[keep],
    )

    estimate = fit_low_rank_plugin(
        sparse_panel,
        rank=1,
        n_starts=3,
        tolerance=1e-11,
        seed=1009,
    )

    assert estimate.sample.dropped_workers == 1
    assert estimate.sample.workers == 6
    assert 0 not in estimate.worker_ids
    assert np.allclose(
        estimate.fitted_schedule,
        population.schedule[1:, :],
        atol=1e-8,
    )

    with pytest.raises(ValueError, match="at least rank \\+ 1"):
        fit_low_rank_plugin(
            sparse_panel,
            rank=1,
            minimum_degree=1,
        )
