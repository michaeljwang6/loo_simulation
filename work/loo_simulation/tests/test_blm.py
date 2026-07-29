import numpy as np
import pytest

from loo_sim import generate_grouped_population, sample_panel
from loo_sim.pytwoway_estimators import (
    align_blm_cell_means,
    estimate_blm,
    prepare_blm_data,
)


pytest.importorskip("pytwoway")


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
    assert alignment.rmse < 0.15
