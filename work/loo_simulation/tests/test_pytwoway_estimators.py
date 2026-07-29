import numpy as np
import pytest

from loo_sim import generate_population, sample_panel
from loo_sim.pytwoway_estimators import (
    estimate_bs20,
    estimate_fe_kss,
    panel_to_bipartite,
    prepare_bs20_sample,
)


pytest.importorskip("pytwoway")


def _comparison_panel():
    population = generate_population(
        n_workers=120,
        n_firms=12,
        rank=0,
        common_sorting=0.6,
        seed=701,
    )
    return sample_panel(
        population,
        n_periods=8,
        redraw_probability=0.75,
        error_sd=0.5,
        seed=702,
    )


def test_panel_adapter_preserves_long_columns() -> None:
    panel = _comparison_panel()
    adata = panel_to_bipartite(panel)

    assert set(["i", "j", "t", "y"]).issubset(adata.columns)
    assert len(adata) == panel.n_observations
    assert np.array_equal(adata.loc[:, "i"].to_numpy(), panel.worker_id)
    assert np.allclose(adata.loc[:, "y"].to_numpy(), panel.outcome)


def test_bs20_pipeline_removes_returns_and_retains_valid_matches() -> None:
    panel = _comparison_panel()
    cleaned = prepare_bs20_sample(panel)

    assert cleaned.no_returns
    assert cleaned.groupby("i").size().min() >= 2
    assert cleaned.groupby("j").size().min() >= 2
    assert "w" in cleaned.columns


def test_bs20_wrapper_returns_finite_native_moments() -> None:
    panel = _comparison_panel()
    result = estimate_bs20(panel)
    cleaned = prepare_bs20_sample(panel)
    weights = cleaned.loc[:, "w"].to_numpy()
    wages = cleaned.loc[:, "y"].to_numpy()
    mean_y = np.average(wages, weights=weights)
    expected_var_y = np.average((wages - mean_y) ** 2, weights=weights)

    assert result.sample.observations <= 120 * 8
    assert result.sample.workers <= 120
    assert result.sample.firms <= 12
    assert np.isfinite(result.mean_y)
    assert np.isfinite(result.var_worker_type)
    assert np.isfinite(result.var_firm_type)
    assert np.isfinite(result.covariance)
    assert np.isfinite(result.correlation)
    assert np.isclose(result.mean_y, mean_y)
    assert np.isclose(result.var_y, expected_var_y)


def test_fe_kss_wrapper_returns_exact_bias_corrections() -> None:
    population = generate_population(
        n_workers=80,
        n_firms=10,
        rank=0,
        common_sorting=0.6,
        seed=701,
    )
    panel = sample_panel(
        population,
        n_periods=7,
        redraw_probability=0.75,
        error_sd=0.5,
        seed=702,
    )
    result = estimate_fe_kss(panel, exact=True)

    assert result.sample.observations <= 80 * 7
    assert result.sample.workers <= 80
    assert result.sample.firms <= 10
    assert np.isfinite(result.var_psi_fe)
    assert np.isfinite(result.var_psi_ho)
    assert np.isfinite(result.var_psi_he)
    assert np.isfinite(result.cov_psi_alpha_he)
    assert 0 <= result.min_leverage <= result.max_leverage < 1
