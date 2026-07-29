"""Population data-generating processes for full worker--firm schedules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


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


@dataclass(frozen=True)
class GroupedPopulationDGP(PopulationDGP):
    """A population whose wage schedule is constant within type-class cells."""

    worker_groups: IntArray
    firm_groups: IntArray
    cell_means: FloatArray


def _standardize(values: FloatArray, weights: FloatArray) -> FloatArray:
    mean = np.sum(weights * values)
    centered = values - mean
    scale = np.sqrt(np.sum(weights * centered**2))
    return centered / scale if scale > 0 else centered


def _weighted_orthonormalize_columns(
    values: FloatArray,
    weights: FloatArray,
) -> FloatArray:
    """Center and orthonormalize columns under a discrete population law."""

    if values.shape[1] == 0:
        return values.copy()

    centered = values - np.sum(weights[:, None] * values, axis=0)
    weighted = np.sqrt(weights)[:, None] * centered
    orthonormal, triangular = np.linalg.qr(weighted, mode="reduced")

    # Fix the otherwise arbitrary QR column signs so seeded simulations remain
    # stable across linear-algebra implementations.
    signs = np.sign(np.diag(triangular))
    signs[signs == 0] = 1.0
    orthonormal *= signs

    return orthonormal / np.sqrt(weights)[:, None]


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
    if rank > min(n_workers - 1, n_firms - 1):
        raise ValueError(
            "rank cannot exceed min(n_workers - 1, n_firms - 1) after "
            "weighted centering."
        )

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
        if np.any(np.diff(singular) > 0):
            raise ValueError("singular_values must be in non-increasing order.")

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
    worker_factors = _weighted_orthonormalize_columns(worker_factors, p)
    firm_factors = _weighted_orthonormalize_columns(firm_factors, q)

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


def generate_grouped_population(
    *,
    n_workers: int = 600,
    n_firms: int = 30,
    n_worker_types: int = 2,
    n_firm_types: int = 3,
    rank: int = 1,
    singular_values: tuple[float, ...] | None = None,
    common_sorting: float = 0.4,
    interaction_sorting: float = 0.2,
    grand_mean: float = 0.0,
    seed: int = 24680,
) -> GroupedPopulationDGP:
    """Generate a BLM-style grouped wage schedule and assignment law."""

    if n_worker_types < 1 or n_worker_types > n_workers:
        raise ValueError(
            "n_worker_types must lie between one and n_workers."
        )
    if n_firm_types < 1 or n_firm_types > n_firms:
        raise ValueError("n_firm_types must lie between one and n_firms.")
    if rank < 0 or rank > min(n_worker_types - 1, n_firm_types - 1):
        raise ValueError(
            "rank must lie between zero and "
            "min(n_worker_types - 1, n_firm_types - 1)."
        )
    if singular_values is None:
        singular = np.ones(rank, dtype=float)
    else:
        singular = np.asarray(singular_values, dtype=float)
        if singular.shape != (rank,):
            raise ValueError(
                f"singular_values must have length {rank}; got "
                f"{singular.shape}."
            )
        if np.any(singular < 0):
            raise ValueError("singular_values cannot be negative.")
        if np.any(np.diff(singular) > 0):
            raise ValueError(
                "singular_values must be in non-increasing order."
            )

    rng = np.random.default_rng(seed)
    worker_groups = (
        np.arange(n_workers, dtype=np.int64) % n_worker_types
    )
    firm_groups = np.arange(n_firms, dtype=np.int64) % n_firm_types
    rng.shuffle(worker_groups)
    rng.shuffle(firm_groups)

    worker_type_weights = np.bincount(
        worker_groups, minlength=n_worker_types
    ).astype(float)
    worker_type_weights /= worker_type_weights.sum()
    firm_type_weights = np.bincount(
        firm_groups, minlength=n_firm_types
    ).astype(float)
    firm_type_weights /= firm_type_weights.sum()

    worker_type_main = _standardize(
        rng.normal(size=n_worker_types),
        worker_type_weights,
    )
    firm_type_main = _standardize(
        rng.normal(size=n_firm_types),
        firm_type_weights,
    )
    raw_worker_factors = rng.normal(size=(n_worker_types, rank))
    raw_firm_factors = rng.normal(size=(n_firm_types, rank))
    worker_type_factors = _weighted_orthonormalize_columns(
        raw_worker_factors,
        worker_type_weights,
    )
    firm_type_factors = _weighted_orthonormalize_columns(
        raw_firm_factors,
        firm_type_weights,
    )
    if rank == 0:
        type_interaction = np.zeros(
            (n_worker_types, n_firm_types),
            dtype=float,
        )
    else:
        type_interaction = (
            worker_type_factors * singular
        ) @ firm_type_factors.T

    cell_means = (
        grand_mean
        + worker_type_main[:, np.newaxis]
        + firm_type_main[np.newaxis, :]
        + type_interaction
    )
    worker_main = worker_type_main[worker_groups]
    firm_main = firm_type_main[firm_groups]
    worker_factors = worker_type_factors[worker_groups]
    firm_factors = firm_type_factors[firm_groups]
    interaction = type_interaction[
        worker_groups[:, np.newaxis],
        firm_groups[np.newaxis, :],
    ]
    schedule = cell_means[
        worker_groups[:, np.newaxis],
        firm_groups[np.newaxis, :],
    ]

    p = np.full(n_workers, 1.0 / n_workers)
    q = np.full(n_firms, 1.0 / n_firms)
    common_score = worker_main[:, np.newaxis] * firm_main[np.newaxis, :]
    if np.std(interaction) > 0:
        interaction_score = interaction / np.std(interaction)
    else:
        interaction_score = interaction
    log_kernel = (
        common_sorting * common_score
        + interaction_sorting * interaction_score
    )
    kernel = np.exp(np.clip(log_kernel, -30.0, 30.0))
    assignment = _balance_kernel(kernel, p, q)

    return GroupedPopulationDGP(
        schedule=schedule,
        assignment=assignment,
        worker_weights=p,
        firm_weights=q,
        worker_main=worker_main,
        firm_main=firm_main,
        worker_factors=worker_factors,
        firm_factors=firm_factors,
        singular_values=singular,
        interaction=interaction,
        worker_groups=worker_groups,
        firm_groups=firm_groups,
        cell_means=cell_means,
    )
