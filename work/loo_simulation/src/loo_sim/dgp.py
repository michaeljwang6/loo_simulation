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


def _assignment_from_components(
    worker_main: FloatArray,
    firm_main: FloatArray,
    interaction: FloatArray,
    worker_weights: FloatArray,
    firm_weights: FloatArray,
    *,
    common_sorting: float,
    interaction_sorting: float,
) -> FloatArray:
    """Create a sorted assignment law while preserving fixed marginals."""

    common_score = worker_main[:, None] * firm_main[None, :]
    interaction_sd = float(np.std(interaction))
    interaction_score = (
        interaction / interaction_sd
        if interaction_sd > 0
        else interaction
    )
    log_kernel = (
        common_sorting * common_score
        + interaction_sorting * interaction_score
    )
    kernel = np.exp(np.clip(log_kernel, -30.0, 30.0))
    return _balance_kernel(kernel, worker_weights, firm_weights)


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
    """Generate the project's continuous low-rank-factor DGP.

    Before finite-population normalization, worker and firm main effects and
    factor innovations are Gaussian. The requested factor correlations are
    imposed on those Gaussian draws. Main effects are then standardized and
    factor columns are centered and orthonormalized under uniform population
    weights, so ``singular_values`` are the exact weighted interaction
    singular values in every replication.

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

    assignment = _assignment_from_components(
        alpha,
        psi,
        interaction,
        p,
        q,
        common_sorting=common_sorting,
        interaction_sorting=interaction_sorting,
    )

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


def generate_akm_population(
    *,
    n_workers: int = 100,
    n_firms: int = 40,
    common_sorting: float = 0.0,
    grand_mean: float = 0.0,
    seed: int = 12345,
) -> PopulationDGP:
    """Generate the additive AKM benchmark as a distinct DGP."""

    return generate_population(
        n_workers=n_workers,
        n_firms=n_firms,
        rank=0,
        singular_values=(),
        common_sorting=common_sorting,
        interaction_sorting=0.0,
        grand_mean=grand_mean,
        seed=seed,
    )


def generate_crippa_population(
    *,
    n_workers: int = 100,
    n_firms: int = 40,
    beta_0: float = 0.75,
    common_sorting: float = 0.0,
    interaction_sorting: float = 0.0,
    grand_mean: float = 0.0,
    seed: int = 13579,
) -> PopulationDGP:
    r"""Generate Crippa's nonadditive Tukey mean surface.

    The population wage schedule is

    .. math::

        m_{ij}=\mu+\alpha_i+\psi_j+\beta_0\alpha_i\psi_j.

    Raw worker and firm types are independent standard normal draws. They are
    standardized to weighted mean zero and variance one in the realized
    finite population. A nonzero ``beta_0`` therefore gives an exact rank-one
    interaction with weighted singular value ``abs(beta_0)``.
    """

    if n_workers < 2 or n_firms < 2:
        raise ValueError("At least two workers and two firms are required.")
    if not np.isfinite(beta_0) or beta_0 == 0:
        raise ValueError("beta_0 must be finite and nonzero.")

    rng = np.random.default_rng(seed)
    p = np.full(n_workers, 1.0 / n_workers)
    q = np.full(n_firms, 1.0 / n_firms)
    alpha = _standardize(rng.normal(size=n_workers), p)
    psi = _standardize(rng.normal(size=n_firms), q)
    interaction = beta_0 * alpha[:, None] * psi[None, :]
    schedule = (
        grand_mean
        + alpha[:, None]
        + psi[None, :]
        + interaction
    )
    assignment = _assignment_from_components(
        alpha,
        psi,
        interaction,
        p,
        q,
        common_sorting=common_sorting,
        interaction_sorting=interaction_sorting,
    )
    firm_factor = np.sign(beta_0) * psi[:, None]
    return PopulationDGP(
        schedule=schedule,
        assignment=assignment,
        worker_weights=p,
        firm_weights=q,
        worker_main=alpha,
        firm_main=psi,
        worker_factors=alpha[:, None],
        firm_factors=firm_factor,
        singular_values=np.asarray([abs(beta_0)], dtype=float),
        interaction=interaction,
    )


def generate_gklp_population(
    *,
    n_workers: int = 100,
    n_firms: int = 40,
    ability_correlation: float = 0.35,
    firm_type_correlation: float = 0.25,
    productivity_shock_sd: float = 0.5,
    common_sorting: float = 0.0,
    interaction_sorting: float = 0.0,
    grand_mean: float = 0.0,
    seed: int = 97531,
) -> PopulationDGP:
    r"""Generate a static perfect-information GKLP log-wage slice.

    After residualizing observed covariates, the simulated schedule is the
    finite-population version of

    .. math::

        m_{ij}=Z_i+c_j+b_j\eta_i
        +\tfrac12 b_j^2\sigma_\varepsilon^2.

    ``(Z_i, eta_i)`` and ``(c_j, b_j)`` start as correlated Gaussian pairs.
    Main effects are standardized and factor coordinates are centered and
    normalized. The convex risk term remains part of the firm main effect;
    the comparative-advantage interaction ``eta_i b_j`` has exact weighted
    rank one.
    """

    if n_workers < 2 or n_firms < 2:
        raise ValueError("At least two workers and two firms are required.")
    if productivity_shock_sd < 0:
        raise ValueError("productivity_shock_sd cannot be negative.")

    rng = np.random.default_rng(seed)
    p = np.full(n_workers, 1.0 / n_workers)
    q = np.full(n_firms, 1.0 / n_firms)
    general_ability, sector_ability = _correlated_pair(
        n_workers,
        1,
        ability_correlation,
        rng,
    )
    firm_constant, firm_loading = _correlated_pair(
        n_firms,
        1,
        firm_type_correlation,
        rng,
    )
    general_ability = _standardize(general_ability, p)
    sector_ability = _weighted_orthonormalize_columns(
        sector_ability,
        p,
    )
    firm_constant = _standardize(firm_constant, q)
    firm_loading = _weighted_orthonormalize_columns(
        firm_loading,
        q,
    )
    b = firm_loading[:, 0]
    raw_firm_main = (
        firm_constant
        + 0.5 * b**2 * productivity_shock_sd**2
    )
    firm_main = raw_firm_main - np.sum(q * raw_firm_main)
    interaction = sector_ability @ firm_loading.T
    schedule = (
        grand_mean
        + general_ability[:, None]
        + firm_main[None, :]
        + interaction
    )
    assignment = _assignment_from_components(
        general_ability,
        firm_main,
        interaction,
        p,
        q,
        common_sorting=common_sorting,
        interaction_sorting=interaction_sorting,
    )
    return PopulationDGP(
        schedule=schedule,
        assignment=assignment,
        worker_weights=p,
        firm_weights=q,
        worker_main=general_ability,
        firm_main=firm_main,
        worker_factors=sector_ability,
        firm_factors=firm_loading,
        singular_values=np.ones(1, dtype=float),
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
    assignment = _assignment_from_components(
        worker_main,
        firm_main,
        interaction,
        p,
        q,
        common_sorting=common_sorting,
        interaction_sorting=interaction_sorting,
    )

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
