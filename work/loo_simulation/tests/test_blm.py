import numpy as np
import pytest

from loo_sim import generate_grouped_population, sample_panel
from loo_sim.panel import PanelData
from loo_sim.pytwoway_estimators import (
    align_blm_cell_means,
    compute_blm_support_diagnostics,
    estimate_blm,
    prepare_blm_data,
)


pytest.importorskip("pytwoway")


def _two_period_classification_panel() -> tuple[PanelData, np.ndarray]:
    """Create complete static-BLM support plus moves after period one."""

    firm_groups = np.repeat(np.arange(3, dtype=np.int64), 2)
    histories: list[list[int]] = [
        [0, 0, 2, 2],
        [2, 2, 4, 4],
        [4, 4, 0, 0],
    ]
    first_firm = (0, 2, 4)
    alternate_firm = (1, 3, 5)
    for origin_group in range(3):
        for destination_group in range(3):
            origin = first_firm[origin_group]
            destination = (
                alternate_firm[destination_group]
                if origin_group == destination_group
                else first_firm[destination_group]
            )
            histories.append([origin, destination, destination, destination])

    firm_id = np.asarray(histories, dtype=np.int64).reshape(-1)
    n_workers, n_periods = len(histories), 4
    worker_id = np.repeat(np.arange(n_workers, dtype=np.int64), n_periods)
    period = np.tile(np.arange(n_periods, dtype=np.int64), n_workers)
    outcome = (
        10.0 * worker_id
        + period
        + firm_groups[firm_id].astype(float)
    )
    return (
        PanelData(
            worker_id=worker_id,
            firm_id=firm_id,
            period=period,
            outcome=outcome,
            systematic_wage=outcome.copy(),
            error=np.zeros_like(outcome),
        ),
        firm_groups,
    )


def test_blm_support_diagnostics_identify_exact_missing_cells() -> None:
    support = compute_blm_support_diagnostics(
        mover_origins=np.array([0, 0, 1, 2]),
        mover_destinations=np.array([0, 1, 2, 2]),
        stayer_groups=np.array([0, 2]),
        n_firm_types=3,
    )

    assert support.stayer_groups == (0, 2)
    assert support.missing_stayer_groups == (1,)
    assert support.mover_pairs == ((0, 0), (0, 1), (1, 2), (2, 2))
    assert support.missing_mover_pairs == (
        (0, 2),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    )
    assert not support.complete


def test_blm_classifies_stayers_only_over_declared_period_pair() -> None:
    panel, firm_groups = _two_period_classification_panel()

    prepared = prepare_blm_data(
        panel,
        n_firm_types=3,
        firm_groups=firm_groups,
        periods=(0, 1),
        seed=1100,
    )

    assert prepared.sample.first_period == 0
    assert prepared.sample.second_period == 1
    assert prepared.sample.observations == 2 * panel.n_workers
    assert prepared.sample.event_rows == panel.n_workers
    assert prepared.sample.stayer_rows == 3
    assert prepared.sample.mover_rows == 9
    assert prepared.support.complete
    assert set(prepared.sdata.loc[:, "i"]) == {0, 1, 2}
    first_stayer = prepared.sdata.loc[
        prepared.sdata.loc[:, "i"] == 0, :
    ].iloc[0]
    assert first_stayer["y1"] == panel.outcome[0]
    assert first_stayer["y2"] == panel.outcome[1]


@pytest.mark.parametrize("periods", [(1, 1), (2, 1), (0, 9)])
def test_blm_rejects_invalid_or_unavailable_period_pairs(periods) -> None:
    panel, firm_groups = _two_period_classification_panel()

    with pytest.raises(ValueError):
        prepare_blm_data(
            panel,
            n_firm_types=3,
            firm_groups=firm_groups,
            periods=periods,
        )


@pytest.fixture(scope="module")
def grouped_panel():
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
    return population, panel


def test_blm_oracle_preparation_retains_movers_stayers_and_groups(
    grouped_panel,
) -> None:
    population, panel = grouped_panel
    prepared = prepare_blm_data(
        panel,
        n_firm_types=3,
        firm_groups=population.firm_groups,
        seed=1103,
    )

    assert prepared.variant == "oracle_firm_groups"
    assert prepared.sample.mover_rows > 0
    assert prepared.sample.stayer_rows > 0
    assert prepared.sample.firm_groups == 3
    assert {"g1", "g2", "y1", "y2"}.issubset(
        prepared.jdata.columns
    )


@pytest.mark.parametrize("oracle_groups", [True, False])
def test_blm_recovers_grouped_cell_means(
    grouped_panel,
    oracle_groups: bool,
) -> None:
    population, panel = grouped_panel
    estimate = estimate_blm(
        panel,
        n_worker_types=2,
        n_firm_types=3,
        firm_groups=population.firm_groups if oracle_groups else None,
        n_init=2,
        n_best=1,
        n_iterations=100,
        threshold=1e-4,
        seed=1104 + int(oracle_groups),
    )
    alignment = align_blm_cell_means(
        estimate.stationary_cell_means,
        population.cell_means,
        allow_firm_permutation=not oracle_groups,
    )

    assert estimate.sample.mover_rows > 0
    assert estimate.sample.stayer_rows > 0
    assert estimate.mover_likelihood_monotone
    assert estimate.stayer_likelihood_monotone
    assert np.isfinite(estimate.connectedness)
    # Estimated grouping uses only the declared two-period cross-section,
    # whereas oracle grouping is given the true firm classes. The looser
    # finite-sample bound for estimated groups reflects that extra first-step
    # sampling noise; the 25,000-worker production design is much larger.
    assert alignment.rmse < (0.25 if not oracle_groups else 0.15)
