"""Population targets for the comparison procedures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .truth import PopulationTruth, compute_population_truth


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_EXPLICIT_AKM_MAX_CELLS = 2_000_000


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


@dataclass(frozen=True)
class BLMGroupedPopulationTarget:
    """Native type-by-firm-class mean surface and grouped project objects."""

    cell_means: FloatArray
    group_assignment: FloatArray
    worker_type_weights: FloatArray
    firm_type_weights: FloatArray
    within_cell_variance: float
    project_functionals: PopulationTruth


@dataclass(frozen=True)
class BLMEvaluationGroups:
    """Oracle discretization used to evaluate BLM on any population DGP."""

    worker_groups: IntArray
    firm_groups: IntArray
    worker_score: FloatArray
    firm_score: FloatArray
    method: str


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
    return _compute_akm_from_truth(truth, atol=atol)


def _compute_akm_from_truth(
    truth: PopulationTruth,
    *,
    atol: float,
) -> AKMPopulationTarget:
    """Compute the additive projection while reusing a validated truth."""

    m = truth.schedule
    probability = truth.assignment
    p = truth.worker_weights
    q = truth.firm_weights
    n_firms = m.shape[1]

    centered = m - truth.observed_mean
    row_mean = np.sum(probability * centered, axis=1) / p
    column_total = np.sum(probability * centered, axis=0)

    right_hand_side = column_total - probability.T @ row_mean

    if probability.size <= _EXPLICIT_AKM_MAX_CELLS:
        # The explicit KKT solve is useful for small tests and gives a direct
        # connected-support rank check.
        laplacian = np.diag(q) - probability.T @ (
            probability / p[:, np.newaxis]
        )
        laplacian = 0.5 * (laplacian + laplacian.T)
        rank = int(np.linalg.matrix_rank(laplacian, tol=atol))
        if rank != n_firms - 1:
            raise ValueError(
                "AKM population effects require connected assignment "
                f"support; firm-side normal matrix has rank {rank}, "
                f"expected {n_firms - 1}."
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
    else:
        # Forming P' diag(1/p) P costs O(n_workers * n_firms**2), which is
        # prohibitive for the 25,000-by-5,000 cluster design. CG only needs
        # products by that matrix. The q q' term removes the constant null
        # direction and imposes q-weighted normalization.
        from scipy.sparse.linalg import LinearOperator, cg

        def firm_normal_product(value: FloatArray) -> FloatArray:
            return (
                q * value
                - probability.T @ ((probability @ value) / p)
                + q * float(q @ value)
            )

        operator = LinearOperator(
            (n_firms, n_firms),
            matvec=firm_normal_product,
            dtype=float,
        )
        firm_effect, info = cg(
            operator,
            right_hand_side,
            rtol=max(1e-12, atol * 0.01),
            atol=0.0,
            maxiter=max(1_000, 5 * n_firms),
        )
        if info != 0:
            raise ValueError(
                "Matrix-free AKM population projection did not converge; "
                f"scipy.sparse.linalg.cg returned info={info}. This can "
                "indicate disconnected or nearly disconnected support."
            )
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
    akm = _compute_akm_from_truth(project, atol=atol)
    return ProcedureTargets(project=project, akm=akm)


def compute_blm_grouped_target(
    schedule: ArrayLike,
    assignment: ArrayLike,
    worker_groups: ArrayLike,
    firm_groups: ArrayLike,
    *,
    atol: float = 1e-10,
    project: PopulationTruth | None = None,
) -> BLMGroupedPopulationTarget:
    """Aggregate a population into BLM worker-type by firm-class cells."""

    if project is None:
        project = compute_population_truth(schedule, assignment, atol=atol)
    else:
        if project.schedule.shape != np.shape(schedule):
            raise ValueError("project and schedule must have the same shape.")
    worker_group = np.asarray(worker_groups, dtype=np.int64)
    firm_group = np.asarray(firm_groups, dtype=np.int64)
    if worker_group.shape != (project.schedule.shape[0],):
        raise ValueError("worker_groups must have one entry per worker.")
    if firm_group.shape != (project.schedule.shape[1],):
        raise ValueError("firm_groups must have one entry per firm.")
    if np.any(worker_group < 0) or np.any(firm_group < 0):
        raise ValueError("group labels cannot be negative.")
    worker_labels = np.unique(worker_group)
    firm_labels = np.unique(firm_group)
    if not np.array_equal(
        worker_labels, np.arange(worker_labels.size)
    ):
        raise ValueError("worker group labels must be contiguous from zero.")
    if not np.array_equal(firm_labels, np.arange(firm_labels.size)):
        raise ValueError("firm group labels must be contiguous from zero.")

    worker_indicator = np.eye(worker_labels.size)[worker_group]
    firm_indicator = np.eye(firm_labels.size)[firm_group]
    group_assignment = (
        worker_indicator.T @ project.assignment @ firm_indicator
    )
    weighted_schedule = project.assignment * project.schedule
    group_wage_total = (
        worker_indicator.T @ weighted_schedule @ firm_indicator
    )
    del weighted_schedule
    if np.any(group_assignment <= 0):
        raise ValueError("Every BLM type-class cell must have positive mass.")
    cell_means = group_wage_total / group_assignment

    total_second_moment = float(
        np.einsum(
            "ij,ij,ij->",
            project.assignment,
            project.schedule,
            project.schedule,
            optimize=True,
        )
    )
    within_cell_variance = float(
        total_second_moment
        - np.sum(group_assignment * cell_means**2)
    )
    grouped_functionals = compute_population_truth(
        cell_means,
        group_assignment,
        atol=atol,
    )
    return BLMGroupedPopulationTarget(
        cell_means=cell_means,
        group_assignment=group_assignment,
        worker_type_weights=group_assignment.sum(axis=1),
        firm_type_weights=group_assignment.sum(axis=0),
        within_cell_variance=within_cell_variance,
        project_functionals=grouped_functionals,
    )


def _ordered_equal_count_groups(
    scores: FloatArray,
    n_groups: int,
    *,
    name: str,
) -> IntArray:
    if n_groups < 1 or n_groups > scores.size:
        raise ValueError(
            f"{name} group count must lie between one and {scores.size}."
        )
    order = np.argsort(scores, kind="mergesort")
    labels = np.empty(scores.size, dtype=np.int64)
    for position, index in enumerate(order):
        labels[index] = min(position * n_groups // scores.size, n_groups - 1)
    if np.unique(labels).size != n_groups:
        raise ArithmeticError(f"{name} grouping produced an empty group.")
    return labels


def compute_blm_evaluation_groups(
    schedule: ArrayLike,
    assignment: ArrayLike,
    *,
    n_worker_types: int,
    n_firm_types: int,
    project: PopulationTruth | None = None,
) -> BLMEvaluationGroups:
    """Construct a common oracle BLM discretization for any wage schedule.

    This is an evaluation device, not an assumption that a continuous DGP is
    literally grouped. Product-marginal mean wages from the complete schedule
    order workers and firms. Equal-count bins of those scores define the cell
    target against which BLM is aligned. This rule is deterministic and
    remains well-defined under an additive schedule, whose leading singular
    values can be tied. A true grouped DGP can instead pass its simulated
    labels directly to ``compute_blm_grouped_target``.
    """

    truth = (
        compute_population_truth(schedule, assignment)
        if project is None
        else project
    )
    if truth.schedule.shape != np.shape(schedule):
        raise ValueError("project and schedule must have the same shape.")
    worker_score = truth.worker_main.copy()
    firm_score = truth.firm_main.copy()
    worker_groups = _ordered_equal_count_groups(
        worker_score,
        n_worker_types,
        name="worker",
    )
    firm_groups = _ordered_equal_count_groups(
        firm_score,
        n_firm_types,
        name="firm",
    )
    return BLMEvaluationGroups(
        worker_groups=worker_groups,
        firm_groups=firm_groups,
        worker_score=worker_score,
        firm_score=firm_score,
        method="product_marginal_mean_equal_count_bins",
    )
