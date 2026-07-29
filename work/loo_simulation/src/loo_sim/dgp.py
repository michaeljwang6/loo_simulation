"""Population data-generating processes for full worker--firm schedules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PopulationDGP:
    """A complete simulated economy before finite-panel sampling."""

    schedule: FloatArray
    assignment: FloatArray
    worker_weights: FloatArray
    firm_weights: FloatArray
    worker_main: FloatArray
    firm_main: FloatArray
    worker_factors: FloatArray
    firm_factors: FloatArray
    singular_values: FloatArray
    interaction: FloatArray


def _standardize(values: FloatArray, weights: FloatArray) -> FloatArray:
    mean = np.sum(weights * values)
    centered = values - mean
    scale = np.sqrt(np.sum(weights * centered**2))
    return centered / scale if scale > 0 else centered


def _weighted_center_columns(
    values: FloatArray,
    weights: FloatArray,
) -> FloatArray:
    return values - np.sum(weights[:, None] * values, axis=0)


def _correlated_pair(
    n: int,
    rank: int,
    correlation: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray]:
    if not -1 <= correlation <= 1:
        raise ValueError("correlation must lie in [-1, 1].")
    main = rng.normal(size=n)
    independent = rng.normal(size=(n, rank))
    factors = (
        correlation * main[:, None]
        + np.sqrt(max(0.0, 1.0 - correlation**2)) * independent
    )
    return main, factors


def _balance_kernel(
    kernel: FloatArray,
    row_target: FloatArray,
    column_target: FloatArray,
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 20_000,
) -> FloatArray:
    """Balance a positive kernel to fixed marginals by iterative scaling."""

    matrix = np.asarray(kernel, dtype=float).copy()
    if np.any(matrix <= 0) or not np.all(np.isfinite(matrix)):
        raise ValueError("assignment kernel must be finite and strictly positive.")

    for _ in range(max_iterations):
        matrix *= (row_target / matrix.sum(axis=1))[:, None]
        matrix *= (column_target / matrix.sum(axis=0))[None, :]
        row_error = np.max(np.abs(matrix.sum(axis=1) - row_target))
        column_error = np.max(np.abs(matrix.sum(axis=0) - column_target))
        if max(row_error, column_error) < tolerance:
            return matrix

    raise RuntimeError("assignment balancing did not converge.")


def generate_population(
    *,
    n_workers: int = 100,
    n_firms: int = 40,
    rank: int = 1,
    singular_values: tuple[float, ...] | None = None,
    worker_factor_correlation: float = 0.0,
    firm_factor_correlation: float = 0.0,
    common_sorting: float = 0.0,
    interaction_sorting: float = 0.0,
    grand_mean: float = 0.0,
    seed: int = 12345,
) -> PopulationDGP:
    """Generate a complete free-factor wage schedule and assignment law.

    Assignment starts from uniform worker and firm marginals. Exponential
    tilting introduces common-component and interaction-gain sorting, after
    which iterative proportional fitting restores the declared marginals.
    This makes assignment comparisons hold the worker and firm populations
    fixed by construction.
    """

    if n_workers < 2 or n_firms < 2:
        raise ValueError("At least two workers and two firms are required.")
    if rank < 0:
        raise ValueError("rank cannot be negative.")

    if singular_values is None:
        singular = np.ones(rank, dtype=float)
    else:
        singular = np.asarray(singular_values, dtype=float)
        if singular.shape != (rank,):
            raise ValueError(
                f"singular_values must have length {rank}; got {singular.shape}."
            )
        if np.any(singular < 0):
            raise ValueError("singular_values cannot be negative.")

    rng = np.random.default_rng(seed)
    p = np.full(n_workers, 1.0 / n_workers)
    q = np.full(n_firms, 1.0 / n_firms)

    alpha, worker_factors = _correlated_pair(
        n_workers, rank, worker_factor_correlation, rng
    )
    psi, firm_factors = _correlated_pair(
        n_firms, rank, firm_factor_correlation, rng
    )
    alpha = _standardize(alpha, p)
    psi = _standardize(psi, q)
    worker_factors = _weighted_center_columns(worker_factors, p)
    firm_factors = _weighted_center_columns(firm_factors, q)

    if rank == 0:
        interaction = np.zeros((n_workers, n_firms), dtype=float)
    else:
        interaction = (worker_factors * singular) @ firm_factors.T

    schedule = (
        grand_mean
        + alpha[:, None]
        + psi[None, :]
        + interaction
    )

    common_score = alpha[:, None] * psi[None, :]
    if np.std(interaction) > 0:
        interaction_score = interaction / np.std(interaction)
    else:
        interaction_score = interaction
    log_kernel = common_sorting * common_score + interaction_sorting * interaction_score
    kernel = np.exp(np.clip(log_kernel, -30.0, 30.0))
    assignment = _balance_kernel(kernel, p, q)

    return PopulationDGP(
        schedule=schedule,
        assignment=assignment,
        worker_weights=p,
        firm_weights=q,
        worker_main=alpha,
        firm_main=psi,
        worker_factors=worker_factors,
        firm_factors=firm_factors,
        singular_values=singular,
        interaction=interaction,
    )
