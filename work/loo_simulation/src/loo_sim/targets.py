"""Population targets for the comparison procedures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .truth import PopulationTruth, compute_population_truth


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class AKMPopulationTarget:
    """Assignment-weighted additive projection of a complete wage schedule.

    This is the population object targeted by the default PyTwoWay FE/KSS
    moments when no additional controls are included. Effects are normalized
    to have zero mean under the observed worker and firm marginals.
    """

    observed_mean: float
    worker_effect: FloatArray
    firm_effect: FloatArray
    fitted_schedule: FloatArray
    residual: FloatArray
    worker_variance: float
    firm_variance: float
    covariance: float
    fitted_variance: float
    residual_variance: float


@dataclass(frozen=True)
class ProcedureTargets:
    """Native procedure targets and their gaps from the project estimands."""

    project: PopulationTruth
    akm: AKMPopulationTarget

    @property
    def akm_firm_variance_gap(self) -> float:
        """AKM/KSS firm-variance target minus project ``Q_F``."""

        return self.akm.firm_variance - self.project.q_f

    @property
    def akm_covariance_gap(self) -> float:
        """AKM/KSS covariance target minus project assignment covariance."""

        return self.akm.covariance - self.project.c_assign

    @property
    def bs_covariance_gap(self) -> float:
        """BS20 native covariance target minus project assignment covariance."""

        return self.project.bs_covariance - self.project.c_assign


def _weighted_variance(values: FloatArray, weights: FloatArray) -> float:
    mean = float(np.sum(weights * values))
    return float(np.sum(weights * (values - mean) ** 2))


def compute_akm_population_target(
    schedule: ArrayLike,
    assignment: ArrayLike,
    *,
    atol: float = 1e-10,
) -> AKMPopulationTarget:
    r"""Compute the pseudo-true additive FE projection under observed assignment.

    The projection solves

    .. math::

        \min_{\mu,\alpha,\psi}
        \sum_{ij} P^{obs}_{ij}
        (m_{ij}-\mu-\alpha_i-\psi_j)^2,

    with ``p``-weighted worker effects and ``q``-weighted firm effects
    normalized to zero. The assignment support must be connected so the
    worker and firm effects are identified up to the imposed normalization.
    """

    truth = compute_population_truth(schedule, assignment, atol=atol)
    m = truth.schedule
    probability = truth.assignment
    p = truth.worker_weights
    q = truth.firm_weights
    n_firms = m.shape[1]

    centered = m - truth.observed_mean
    row_mean = np.sum(probability * centered, axis=1) / p
    column_total = np.sum(probability * centered, axis=0)

    # Eliminate the worker effects from the weighted normal equations. The
    # resulting firm-side Laplacian has a one-dimensional constant null space
    # exactly when the worker--firm support graph is connected.
    laplacian = np.diag(q) - probability.T @ (
        probability / p[:, np.newaxis]
    )
    laplacian = 0.5 * (laplacian + laplacian.T)
    right_hand_side = column_total - probability.T @ row_mean

    rank = int(np.linalg.matrix_rank(laplacian, tol=atol))
    if rank != n_firms - 1:
        raise ValueError(
            "AKM population effects require connected assignment support; "
            f"firm-side normal matrix has rank {rank}, expected {n_firms - 1}."
        )

    kkt = np.block(
        [
            [laplacian, q[:, np.newaxis]],
            [q[np.newaxis, :], np.zeros((1, 1))],
        ]
    )
    solution = np.linalg.solve(
        kkt,
        np.concatenate([right_hand_side, np.zeros(1)]),
    )
    firm_effect = solution[:n_firms]
    worker_effect = row_mean - (
        probability @ firm_effect
    ) / p

    fitted_schedule = (
        truth.observed_mean
        + worker_effect[:, np.newaxis]
        + firm_effect[np.newaxis, :]
    )
    residual = m - fitted_schedule

    worker_grid = np.broadcast_to(worker_effect[:, np.newaxis], m.shape)
    firm_grid = np.broadcast_to(firm_effect[np.newaxis, :], m.shape)
    fitted_centered = worker_grid + firm_grid

    worker_variance = _weighted_variance(worker_effect, p)
    firm_variance = _weighted_variance(firm_effect, q)
    covariance = float(np.sum(probability * worker_grid * firm_grid))
    fitted_variance = float(np.sum(probability * fitted_centered**2))
    residual_variance = float(np.sum(probability * residual**2))

    row_residual = np.sum(probability * residual, axis=1)
    column_residual = np.sum(probability * residual, axis=0)
    observed_variance = float(np.sum(probability * centered**2))
    if not np.isclose(np.sum(p * worker_effect), 0.0, atol=atol):
        raise ArithmeticError("AKM worker-effect normalization failed.")
    if not np.isclose(np.sum(q * firm_effect), 0.0, atol=atol):
        raise ArithmeticError("AKM firm-effect normalization failed.")
    if not np.allclose(row_residual, 0.0, atol=atol):
        raise ArithmeticError("AKM worker normal equations failed.")
    if not np.allclose(column_residual, 0.0, atol=atol):
        raise ArithmeticError("AKM firm normal equations failed.")
    if not np.isclose(
        fitted_variance,
        worker_variance + firm_variance + 2.0 * covariance,
        atol=atol,
    ):
        raise ArithmeticError("AKM fitted-variance decomposition failed.")
    if not np.isclose(
        observed_variance,
        fitted_variance + residual_variance,
        atol=atol,
    ):
        raise ArithmeticError("AKM projection-variance decomposition failed.")

    return AKMPopulationTarget(
        observed_mean=truth.observed_mean,
        worker_effect=worker_effect,
        firm_effect=firm_effect,
        fitted_schedule=fitted_schedule,
        residual=residual,
        worker_variance=worker_variance,
        firm_variance=firm_variance,
        covariance=covariance,
        fitted_variance=fitted_variance,
        residual_variance=residual_variance,
    )


def compute_procedure_targets(
    schedule: ArrayLike,
    assignment: ArrayLike,
    *,
    atol: float = 1e-10,
) -> ProcedureTargets:
    """Compute project, AKM/KSS, and BS20 native population targets."""

    project = compute_population_truth(schedule, assignment, atol=atol)
    akm = compute_akm_population_target(schedule, assignment, atol=atol)
    return ProcedureTargets(project=project, akm=akm)
