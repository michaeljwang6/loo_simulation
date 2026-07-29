import numpy as np

from loo_sim import generate_population, sample_panel


def test_panel_shape_and_noiseless_outcomes() -> None:
    population = generate_population(
        n_workers=15,
        n_firms=6,
        rank=1,
        seed=100,
    )
    panel = sample_panel(
        population,
        n_periods=5,
        redraw_probability=0.4,
        error_sd=0.0,
        seed=200,
    )

    assert panel.n_observations == 75
    assert panel.n_workers == 15
    assert panel.n_firms_observed <= 6
    assert np.allclose(panel.outcome, panel.systematic_wage)
    assert np.allclose(panel.error, 0.0)


def test_zero_redraw_probability_produces_no_movers() -> None:
    population = generate_population(
        n_workers=20,
        n_firms=8,
        rank=0,
        seed=300,
    )
    panel = sample_panel(
        population,
        n_periods=4,
        redraw_probability=0.0,
        seed=400,
    )

    firm_history = panel.firm_id.reshape(20, 4)
    assert np.all(firm_history == firm_history[:, [0]])
    assert np.isclose(panel.mover_share, 0.0)


def test_panel_sampling_is_reproducible() -> None:
    population = generate_population(
        n_workers=12,
        n_firms=5,
        rank=1,
        seed=500,
    )
    left = sample_panel(population, seed=600)
    right = sample_panel(population, seed=600)

    assert np.array_equal(left.firm_id, right.firm_id)
    assert np.allclose(left.outcome, right.outcome)
