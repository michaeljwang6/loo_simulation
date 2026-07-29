"""Population functionals for a known worker--firm wage schedule."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PopulationTruth:
    """Exact schedule-based and BS20 population objects."""

    schedule: FloatArray
    assignment: FloatArray
    worker_weights: FloatArray
    firm_weights: FloatArray
    grand_mean_product: float
    observed_mean: float
    worker_main: FloatArray
    firm_main: FloatArray
    interaction: FloatArray
    q_f: float
    h_f: float
    rho_h: float
    c_assign: float
    c_ab: float
    c_ah: float
    c_bh: float
    c_hh: float
    a_h: float
    bs_worker_type: FloatArray
    bs_firm_type: FloatArray
    bs_covariance: float
    bs_worker_variance: float
    bs_firm_variance: float
    bs_correlation: float


def _as_float_array(value: ArrayLike, *, ndim: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions; got {array.ndim}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _normalize_probability(value: ArrayLike, *, ndim: int, name: str) -> FloatArray:
    array = _as_float_array(value, ndim=ndim, name=name)
    if np.any(array < 0):
        raise ValueError(f"{name} contains negative probabilities.")
    total = float(array.sum())
    if total <= 0:
        raise ValueError(f"{name} must have positive total mass.")
    return array / total


def _weighted_variance(values: FloatArray, weights: FloatArray) -> float:
    mean = float(np.sum(weights * values))
    return float(np.sum(weights * (values - mean) ** 2))


def _joint_covariance(
    left: FloatArray,
    right: FloatArray,
    assignment: FloatArray,
) -> float:
    left_mean = float(np.sum(assignment * left))
    right_mean = float(np.sum(assignment * right))
    return float(
        np.sum(assignment * (left - left_mean) * (right - right_mean))
    )


def compute_population_truth(
    schedule: ArrayLike,
    assignment: ArrayLike,
    *,
    atol: float = 1e-10,
) -> PopulationTruth:
    """Compute exact population objects from a full schedule and assignment.

    Parameters
    ----------
    schedule:
        Complete ``(n_workers, n_firms)`` systematic wage schedule.
    assignment:
        Joint observed-match probability matrix of the same shape. It is
        normalized internally to sum to one.
    atol:
        Numerical tolerance used to verify the canonical identities.
    """

    m = _as_float_array(schedule, ndim=2, name="schedule")
    assignment_prob = _normalize_probability(
        assignment, ndim=2, name="assignment"
    )
    if m.shape != assignment_prob.shape:
        raise ValueError(
            "schedule and assignment must have the same shape; "
            f"got {m.shape} and {assignment_prob.shape}."
        )

    p = assignment_prob.sum(axis=1)
    q = assignment_prob.sum(axis=0)
    if np.any(p <= 0) or np.any(q <= 0):
        raise ValueError("Every worker and firm must have positive assignment mass.")

    product = np.outer(p, q)
    grand_mean = float(np.sum(product * m))
    observed_mean = float(np.sum(assignment_prob * m))

    worker_main = m @ q - grand_mean
    firm_main = p @ m - grand_mean
    interaction = (
        m
        - grand_mean
        - worker_main[:, np.newaxis]
        - firm_main[np.newaxis, :]
    )

    row_centered = m - (m @ q)[:, np.newaxis]
    q_f = float(np.sum(product * row_centered**2))

    interaction_second_moment = float(np.sum(product * interaction**2))
    h_f = 2.0 * interaction_second_moment
    rho_h = h_f / (2.0 * q_f) if q_f > atol else float("nan")

    var_observed = float(
        np.sum(assignment_prob * (m - observed_mean) ** 2)
    )
    var_product = float(np.sum(product * (m - grand_mean) ** 2))
    c_assign = 0.5 * (var_observed - var_product)

    a_grid = np.broadcast_to(worker_main[:, np.newaxis], m.shape)
    b_grid = np.broadcast_to(firm_main[np.newaxis, :], m.shape)
    c_ab = _joint_covariance(a_grid, b_grid, assignment_prob)
    c_ah = _joint_covariance(a_grid, interaction, assignment_prob)
    c_bh = _joint_covariance(b_grid, interaction, assignment_prob)
    var_h_observed = _weighted_variance(interaction, assignment_prob)
    var_h_product = _weighted_variance(interaction, product)
    c_hh = 0.5 * (var_h_observed - var_h_product)
    a_h = float(np.sum(assignment_prob * interaction))

    bs_worker_type = np.sum(assignment_prob * m, axis=1) / p
    bs_firm_type = np.sum(assignment_prob * m, axis=0) / q
    bs_worker_grid = np.broadcast_to(bs_worker_type[:, np.newaxis], m.shape)
    bs_firm_grid = np.broadcast_to(bs_firm_type[np.newaxis, :], m.shape)
    bs_covariance = _joint_covariance(
        bs_worker_grid, bs_firm_grid, assignment_prob
    )
    bs_worker_variance = _weighted_variance(bs_worker_type, p)
    bs_firm_variance = _weighted_variance(bs_firm_type, q)
    bs_denom = np.sqrt(bs_worker_variance * bs_firm_variance)
    bs_correlation = (
        bs_covariance / bs_denom if bs_denom > atol else float("nan")
    )

    firm_main_variance = _weighted_variance(firm_main, q)
    if not np.isclose(q_f, firm_main_variance + 0.5 * h_f, atol=atol):
        raise ArithmeticError("Q_F decomposition failed.")
    if not np.allclose(interaction @ q, 0.0, atol=atol):
        raise ArithmeticError("Interaction row-centering condition failed.")
    if not np.allclose(p @ interaction, 0.0, atol=atol):
        raise ArithmeticError("Interaction column-centering condition failed.")
    if not np.isclose(
        c_assign, c_ab + c_ah + c_bh + c_hh, atol=atol
    ):
        raise ArithmeticError("Assignment-channel decomposition failed.")

    return PopulationTruth(
        schedule=m,
        assignment=assignment_prob,
        worker_weights=p,
        firm_weights=q,
        grand_mean_product=grand_mean,
        observed_mean=observed_mean,
        worker_main=worker_main,
        firm_main=firm_main,
        interaction=interaction,
        q_f=q_f,
        h_f=h_f,
        rho_h=rho_h,
        c_assign=c_assign,
        c_ab=c_ab,
        c_ah=c_ah,
        c_bh=c_bh,
        c_hh=c_hh,
        a_h=a_h,
        bs_worker_type=bs_worker_type,
        bs_firm_type=bs_firm_type,
        bs_covariance=bs_covariance,
        bs_worker_variance=bs_worker_variance,
        bs_firm_variance=bs_firm_variance,
        bs_correlation=bs_correlation,
    )
