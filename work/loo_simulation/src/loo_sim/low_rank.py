"""Low-rank weighted-least-squares plug-in estimator.

This module intentionally does not implement the unfinished leave-out bias
correction. It estimates a completed systematic wage schedule and applies the
project functionals to that fitted schedule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .panel import PanelData
from .truth import PopulationTruth, compute_population_truth


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

PLUGIN_LABEL = "low-rank plug-in without LOO correction"


@dataclass(frozen=True)
class LowRankAnalysisSample:
    """Support retained for low-rank estimation."""

    observations: int
    edges: int
    workers: int
    firms: int
    dropped_workers: int
    dropped_firms: int
    min_worker_degree: int
    min_firm_degree: int
    rectangles: int


@dataclass(frozen=True)
class LowRankPluginResult:
    """One fixed-rank weighted-least-squares fit."""

    label: str
    rank: int
    converged: bool
    iterations: int
    sample: LowRankAnalysisSample
    worker_ids: IntArray
    firm_ids: IntArray
    assignment: FloatArray
    match_means: FloatArray
    match_counts: FloatArray
    grand_mean: float
    worker_main: FloatArray
    firm_main: FloatArray
    worker_factors: FloatArray
    firm_factors: FloatArray
    singular_values: FloatArray
    fitted_schedule: FloatArray
    edge_objective: float
    edge_mean_rmse: float
    observation_rmse: float
    functionals: PopulationTruth
    functionally_stable: bool
    near_optimal_starts: int
    q_f_spread: float
    h_f_spread: float
    c_assign_spread: float
    start_objectives: tuple[float, ...]
    start_converged: tuple[bool, ...]
    start_iterations: tuple[int, ...]
    start_q_f: tuple[float, ...]
    start_h_f: tuple[float, ...]
    start_c_assign: tuple[float, ...]


@dataclass(frozen=True)
class BICRankSelectionResult:
    """Exploratory BIC selection over fixed-rank plug-in fits."""

    candidate_ranks: tuple[int, ...]
    bic_values: tuple[float, ...]
    estimates: tuple[LowRankPluginResult, ...]
    selected_rank: int

    @property
    def selected(self) -> LowRankPluginResult:
        return self.estimates[self.candidate_ranks.index(self.selected_rank)]


@dataclass(frozen=True)
class _EdgeData:
    worker_ids: IntArray
    firm_ids: IntArray
    means: FloatArray
    counts: FloatArray
    assignment: FloatArray
    edge_worker: IntArray
    edge_firm: IntArray
    edge_means: FloatArray
    edge_counts: FloatArray
    within_match_sse: float
    sample: LowRankAnalysisSample


@dataclass(frozen=True)
class _StartFit:
    worker_level: FloatArray
    firm_level: FloatArray
    worker_factors: FloatArray
    firm_factors: FloatArray
    singular_values: FloatArray
    objective: float
    converged: bool
    iterations: int


def _largest_support_component(
    mask: NDArray[np.bool_],
    counts: FloatArray,
) -> tuple[IntArray, IntArray]:
    """Return the edge-mass-largest connected bipartite component."""

    n_workers, n_firms = mask.shape
    unseen_workers = set(range(n_workers))
    components: list[tuple[list[int], list[int], float]] = []

    while unseen_workers:
        initial = next(iter(unseen_workers))
        worker_stack = [initial]
        component_workers: set[int] = set()
        component_firms: set[int] = set()

        while worker_stack:
            worker = worker_stack.pop()
            if worker in component_workers:
                continue
            component_workers.add(worker)
            unseen_workers.discard(worker)
            new_firms = np.flatnonzero(mask[worker])
            for firm in new_firms:
                firm_int = int(firm)
                if firm_int in component_firms:
                    continue
                component_firms.add(firm_int)
                linked_workers = np.flatnonzero(mask[:, firm_int])
                worker_stack.extend(int(value) for value in linked_workers)

        worker_index = np.array(sorted(component_workers), dtype=np.int64)
        firm_index = np.array(sorted(component_firms), dtype=np.int64)
        mass = float(counts[np.ix_(worker_index, firm_index)].sum())
        components.append(
            (worker_index.tolist(), firm_index.tolist(), mass)
        )

    selected = max(
        components,
        key=lambda component: (
            component[2],
            len(component[0]) + len(component[1]),
        ),
    )
    return (
        np.asarray(selected[0], dtype=np.int64),
        np.asarray(selected[1], dtype=np.int64),
    )


def _rectangle_count(mask: NDArray[np.bool_]) -> int:
    # The observed worker--firm graph is sparse in the cluster design. A
    # sparse cross-product counts workers shared by each firm pair without
    # performing an O(n_workers * n_firms**2) dense multiplication.
    from scipy import sparse

    incidence = sparse.csr_matrix(mask, dtype=np.int64)
    shared = (incidence.T @ incidence).tocoo()
    upper = shared.data[shared.row < shared.col]
    return int(np.sum(upper * (upper - 1) // 2))


def _prepare_edge_data(panel: PanelData, *, minimum_degree: int) -> _EdgeData:
    if minimum_degree < 1:
        raise ValueError("minimum_degree must be positive.")

    worker_levels, worker_inverse = np.unique(
        panel.worker_id, return_inverse=True
    )
    firm_levels, firm_inverse = np.unique(panel.firm_id, return_inverse=True)
    shape = (worker_levels.size, firm_levels.size)
    counts = np.zeros(shape, dtype=float)
    sums = np.zeros(shape, dtype=float)
    sums_squared = np.zeros(shape, dtype=float)
    np.add.at(counts, (worker_inverse, firm_inverse), 1.0)
    np.add.at(sums, (worker_inverse, firm_inverse), panel.outcome)
    np.add.at(
        sums_squared,
        (worker_inverse, firm_inverse),
        panel.outcome**2,
    )
    mask = counts > 0

    active_workers = np.ones(worker_levels.size, dtype=bool)
    active_firms = np.ones(firm_levels.size, dtype=bool)
    while True:
        worker_degree = np.sum(
            mask[:, active_firms], axis=1
        )
        new_workers = active_workers & (worker_degree >= minimum_degree)
        firm_degree = np.sum(
            mask[new_workers, :], axis=0
        )
        new_firms = active_firms & (firm_degree >= minimum_degree)
        if np.array_equal(new_workers, active_workers) and np.array_equal(
            new_firms, active_firms
        ):
            break
        active_workers = new_workers
        active_firms = new_firms

    worker_index = np.flatnonzero(active_workers)
    firm_index = np.flatnonzero(active_firms)
    if worker_index.size == 0 or firm_index.size == 0:
        raise ValueError(
            "No worker--firm support remains after the minimum-degree "
            f"restriction of {minimum_degree}."
        )

    core_mask = mask[np.ix_(worker_index, firm_index)]
    core_counts = counts[np.ix_(worker_index, firm_index)]
    component_workers, component_firms = _largest_support_component(
        core_mask,
        core_counts,
    )
    worker_index = worker_index[component_workers]
    firm_index = firm_index[component_firms]

    selected_counts = counts[np.ix_(worker_index, firm_index)]
    selected_sums = sums[np.ix_(worker_index, firm_index)]
    selected_sums_squared = sums_squared[np.ix_(worker_index, firm_index)]
    selected_mask = selected_counts > 0
    selected_means = np.zeros_like(selected_sums)
    selected_means[selected_mask] = (
        selected_sums[selected_mask] / selected_counts[selected_mask]
    )
    edge_worker, edge_firm = np.nonzero(selected_mask)
    edge_worker = edge_worker.astype(np.int64, copy=False)
    edge_firm = edge_firm.astype(np.int64, copy=False)
    edge_means = selected_means[edge_worker, edge_firm]
    edge_counts = selected_counts[edge_worker, edge_firm]

    within_match_sse = max(
        0.0,
        float(
            np.sum(
                selected_sums_squared[selected_mask]
                - selected_sums[selected_mask] ** 2
                / selected_counts[selected_mask]
            )
        ),
    )
    observations = int(round(float(selected_counts.sum())))
    assignment = selected_counts / observations
    worker_degree = np.sum(selected_mask, axis=1)
    firm_degree = np.sum(selected_mask, axis=0)

    sample = LowRankAnalysisSample(
        observations=observations,
        edges=int(np.sum(selected_mask)),
        workers=int(worker_index.size),
        firms=int(firm_index.size),
        dropped_workers=int(worker_levels.size - worker_index.size),
        dropped_firms=int(firm_levels.size - firm_index.size),
        min_worker_degree=int(worker_degree.min()),
        min_firm_degree=int(firm_degree.min()),
        rectangles=_rectangle_count(selected_mask),
    )
    return _EdgeData(
        worker_ids=worker_levels[worker_index].astype(np.int64),
        firm_ids=firm_levels[firm_index].astype(np.int64),
        means=selected_means,
        counts=selected_counts,
        assignment=assignment,
        edge_worker=edge_worker,
        edge_firm=edge_firm,
        edge_means=edge_means,
        edge_counts=edge_counts,
        within_match_sse=within_match_sse,
        sample=sample,
    )


def _grouped_weighted_lstsq(
    groups: IntArray,
    n_groups: int,
    design: FloatArray,
    outcome: FloatArray,
    weights: FloatArray,
) -> FloatArray:
    """Solve many small WLS problems without Python loops over groups."""

    n_columns = design.shape[1]
    gram = np.empty((n_groups, n_columns, n_columns), dtype=float)
    for left in range(n_columns):
        for right in range(n_columns):
            gram[:, left, right] = np.bincount(
                groups,
                weights=weights * design[:, left] * design[:, right],
                minlength=n_groups,
            )
    right_hand_side = np.empty((n_groups, n_columns), dtype=float)
    for column in range(n_columns):
        right_hand_side[:, column] = np.bincount(
            groups,
            weights=weights * design[:, column] * outcome,
            minlength=n_groups,
        )

    # A pseudoinverse matches the minimum-norm behavior of np.linalg.lstsq
    # when an individual conditional regression is temporarily collinear.
    return np.einsum(
        "gij,gj->gi",
        np.linalg.pinv(gram, rcond=1e-15),
        right_hand_side,
    )


def _edge_objective(
    data: _EdgeData,
    worker_level: FloatArray,
    firm_level: FloatArray,
    worker_factors: FloatArray | None = None,
    firm_factors: FloatArray | None = None,
) -> float:
    fitted = (
        worker_level[data.edge_worker]
        + firm_level[data.edge_firm]
    )
    if worker_factors is not None and firm_factors is not None:
        fitted = fitted + np.sum(
            worker_factors[data.edge_worker]
            * firm_factors[data.edge_firm],
            axis=1,
        )
    return float(
        np.sum(data.edge_counts * (data.edge_means - fitted) ** 2)
    )


def _fit_additive(
    data: _EdgeData,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[FloatArray, FloatArray, bool, int]:
    n_workers = data.sample.workers
    n_firms = data.sample.firms
    worker_mass = np.bincount(
        data.edge_worker,
        weights=data.edge_counts,
        minlength=n_workers,
    )
    firm_mass = np.bincount(
        data.edge_firm,
        weights=data.edge_counts,
        minlength=n_firms,
    )
    worker_level = np.zeros(n_workers)
    firm_level = np.zeros(n_firms)
    previous = float("inf")
    converged = False

    for iteration in range(1, max_iterations + 1):
        worker_level = np.bincount(
            data.edge_worker,
            weights=(
                data.edge_counts
                * (data.edge_means - firm_level[data.edge_firm])
            ),
            minlength=n_workers,
        ) / worker_mass
        firm_level = np.bincount(
            data.edge_firm,
            weights=(
                data.edge_counts
                * (data.edge_means - worker_level[data.edge_worker])
            ),
            minlength=n_firms,
        ) / firm_mass

        q = firm_mass / firm_mass.sum()
        firm_mean = float(q @ firm_level)
        firm_level -= firm_mean
        worker_level += firm_mean
        objective = _edge_objective(data, worker_level, firm_level)
        if np.isfinite(previous) and abs(
            previous - objective
        ) <= tolerance * max(1.0, previous):
            converged = True
            break
        previous = objective

    return worker_level, firm_level, converged, iteration


def _canonicalize_factors(
    worker_level: FloatArray,
    firm_level: FloatArray,
    worker_factors: FloatArray,
    firm_factors: FloatArray,
    p: FloatArray,
    q: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Center and SVD-normalize factors without changing the fitted schedule."""

    rank = worker_factors.shape[1]
    if rank == 0:
        firm_mean = float(q @ firm_level)
        return (
            worker_level + firm_mean,
            firm_level - firm_mean,
            worker_factors,
            firm_factors,
            np.empty(0),
        )

    worker_shift = p @ worker_factors
    worker_factors = worker_factors - worker_shift
    firm_level = firm_level + firm_factors @ worker_shift

    firm_shift = q @ firm_factors
    firm_factors = firm_factors - firm_shift
    worker_level = worker_level + worker_factors @ firm_shift

    firm_mean = float(q @ firm_level)
    firm_level = firm_level - firm_mean
    worker_level = worker_level + firm_mean

    weighted_worker = np.sqrt(p)[:, np.newaxis] * worker_factors
    weighted_firm = np.sqrt(q)[:, np.newaxis] * firm_factors
    worker_q, worker_r = np.linalg.qr(weighted_worker, mode="reduced")
    firm_q, firm_r = np.linalg.qr(weighted_firm, mode="reduced")
    left, singular_values, right_transpose = np.linalg.svd(
        worker_r @ firm_r.T,
        full_matrices=False,
    )
    root = np.sqrt(np.maximum(singular_values, 0.0))
    worker_factors = (
        worker_q
        @ (left * root[np.newaxis, :])
        / np.sqrt(p)[:, np.newaxis]
    )
    firm_factors = (
        firm_q
        @ (right_transpose.T * root[np.newaxis, :])
        / np.sqrt(q)[:, np.newaxis]
    )
    return (
        worker_level,
        firm_level,
        worker_factors,
        firm_factors,
        singular_values,
    )


def _spectral_initialization(
    data: _EdgeData,
    worker_level: FloatArray,
    firm_level: FloatArray,
    rank: int,
) -> tuple[FloatArray, FloatArray]:
    from scipy import sparse
    from scipy.sparse.linalg import svds

    p = np.bincount(
        data.edge_worker,
        weights=data.edge_counts,
        minlength=data.sample.workers,
    ) / data.sample.observations
    q = np.bincount(
        data.edge_firm,
        weights=data.edge_counts,
        minlength=data.sample.firms,
    ) / data.sample.observations
    residual = (
        data.edge_means
        - worker_level[data.edge_worker]
        - firm_level[data.edge_firm]
    )
    weighted_values = (
        np.sqrt(p[data.edge_worker])
        * residual
        * np.sqrt(q[data.edge_firm])
    )
    weighted = sparse.csr_matrix(
        (
            weighted_values,
            (data.edge_worker, data.edge_firm),
        ),
        shape=(data.sample.workers, data.sample.firms),
    )
    left, singular_values, right_transpose = svds(
        weighted,
        k=rank,
        which="LM",
        v0=np.ones(min(weighted.shape), dtype=float),
    )
    order = np.argsort(singular_values)[::-1]
    singular_values = singular_values[order]
    left = left[:, order]
    right_transpose = right_transpose[order, :]
    root = np.sqrt(np.maximum(singular_values, 0.0))
    worker_factors = (
        left
        * root[np.newaxis, :]
        / np.sqrt(p)[:, np.newaxis]
    )
    firm_factors = (
        right_transpose.T
        * root[np.newaxis, :]
        / np.sqrt(q)[:, np.newaxis]
    )
    return worker_factors, firm_factors


def _fit_start(
    data: _EdgeData,
    *,
    rank: int,
    initial_worker_level: FloatArray,
    initial_firm_level: FloatArray,
    initial_worker_factors: FloatArray,
    initial_firm_factors: FloatArray,
    tolerance: float,
    max_iterations: int,
) -> _StartFit:
    p = np.bincount(
        data.edge_worker,
        weights=data.edge_counts,
        minlength=data.sample.workers,
    ) / data.sample.observations
    q = np.bincount(
        data.edge_firm,
        weights=data.edge_counts,
        minlength=data.sample.firms,
    ) / data.sample.observations
    worker_level = initial_worker_level.copy()
    firm_level = initial_firm_level.copy()
    worker_factors = initial_worker_factors.copy()
    firm_factors = initial_firm_factors.copy()
    previous = float("inf")
    converged = False
    singular_values = np.empty(rank)

    for iteration in range(1, max_iterations + 1):
        worker_design = np.column_stack(
            [
                np.ones(data.sample.edges),
                firm_factors[data.edge_firm],
            ]
        )
        worker_coefficients = _grouped_weighted_lstsq(
            data.edge_worker,
            data.sample.workers,
            worker_design,
            data.edge_means - firm_level[data.edge_firm],
            data.edge_counts,
        )
        worker_level = worker_coefficients[:, 0]
        worker_factors = worker_coefficients[:, 1:]

        firm_design = np.column_stack(
            [
                np.ones(data.sample.edges),
                worker_factors[data.edge_worker],
            ]
        )
        firm_coefficients = _grouped_weighted_lstsq(
            data.edge_firm,
            data.sample.firms,
            firm_design,
            data.edge_means - worker_level[data.edge_worker],
            data.edge_counts,
        )
        firm_level = firm_coefficients[:, 0]
        firm_factors = firm_coefficients[:, 1:]

        (
            worker_level,
            firm_level,
            worker_factors,
            firm_factors,
            singular_values,
        ) = _canonicalize_factors(
            worker_level,
            firm_level,
            worker_factors,
            firm_factors,
            p,
            q,
        )
        objective = _edge_objective(
            data,
            worker_level,
            firm_level,
            worker_factors,
            firm_factors,
        )
        if np.isfinite(previous) and abs(
            previous - objective
        ) <= tolerance * max(1.0, previous):
            converged = True
            break
        previous = objective

    return _StartFit(
        worker_level=worker_level,
        firm_level=firm_level,
        worker_factors=worker_factors,
        firm_factors=firm_factors,
        singular_values=singular_values,
        objective=objective,
        converged=converged,
        iterations=iteration,
    )


def _result_from_starts(
    data: _EdgeData,
    *,
    rank: int,
    starts: list[_StartFit],
) -> LowRankPluginResult:
    best_index = int(
        np.argmin([start.objective for start in starts])
    )
    best = starts[best_index]
    p = data.assignment.sum(axis=1)
    grand_mean = float(p @ best.worker_level)
    worker_main = best.worker_level - grand_mean
    functional_values: list[tuple[float, float, float] | None] = []
    functionals: PopulationTruth | None = None
    fitted_schedule: FloatArray | None = None
    for index, start in enumerate(starts):
        schedule = (
            start.worker_level[:, np.newaxis]
            + start.firm_level[np.newaxis, :]
            + start.worker_factors @ start.firm_factors.T
        )
        try:
            value = compute_population_truth(
                schedule,
                data.assignment,
            )
        except ArithmeticError:
            value = None
        if value is None:
            functional_values.append(None)
            del schedule
        else:
            functional_values.append(
                (value.q_f, value.h_f, value.c_assign)
            )
            if index == best_index:
                functionals = value
                fitted_schedule = schedule
            else:
                del value, schedule
    if functionals is None or fitted_schedule is None:
        raise ValueError(
            "The best weighted-least-squares completion is numerically "
            "unstable. Increase minimum_degree or inspect the support graph."
        )

    objective_cutoff = best.objective + 1e-4 * max(1.0, best.objective)
    near_indices = [
        index
        for index, start in enumerate(starts)
        if start.objective <= objective_cutoff
        and functional_values[index] is not None
    ]
    near_q_f = np.array(
        [functional_values[index][0] for index in near_indices]
    )
    near_h_f = np.array(
        [functional_values[index][1] for index in near_indices]
    )
    near_c_assign = np.array(
        [functional_values[index][2] for index in near_indices]
    )
    q_f_spread = float(np.ptp(near_q_f))
    h_f_spread = float(np.ptp(near_h_f))
    c_assign_spread = float(np.ptp(near_c_assign))
    required_near_starts = 1 if len(starts) == 1 else 2
    functionally_stable = (
        best.converged
        and len(near_indices) >= required_near_starts
        and q_f_spread <= 1e-3 * max(1.0, abs(functionals.q_f))
        and h_f_spread <= 1e-3 * max(1.0, abs(functionals.h_f))
        and c_assign_spread
        <= 1e-3 * max(1.0, abs(functionals.c_assign))
    )
    edge_mean_rmse = np.sqrt(best.objective / data.sample.observations)
    observation_sse = best.objective + data.within_match_sse
    observation_rmse = np.sqrt(
        observation_sse / data.sample.observations
    )

    return LowRankPluginResult(
        label=PLUGIN_LABEL,
        rank=rank,
        converged=best.converged,
        iterations=best.iterations,
        sample=data.sample,
        worker_ids=data.worker_ids,
        firm_ids=data.firm_ids,
        assignment=data.assignment,
        match_means=data.means,
        match_counts=data.counts,
        grand_mean=grand_mean,
        worker_main=worker_main,
        firm_main=best.firm_level,
        worker_factors=best.worker_factors,
        firm_factors=best.firm_factors,
        singular_values=best.singular_values,
        fitted_schedule=fitted_schedule,
        edge_objective=best.objective,
        edge_mean_rmse=float(edge_mean_rmse),
        observation_rmse=float(observation_rmse),
        functionals=functionals,
        functionally_stable=functionally_stable,
        near_optimal_starts=len(near_indices),
        q_f_spread=q_f_spread,
        h_f_spread=h_f_spread,
        c_assign_spread=c_assign_spread,
        start_objectives=tuple(start.objective for start in starts),
        start_converged=tuple(start.converged for start in starts),
        start_iterations=tuple(start.iterations for start in starts),
        start_q_f=tuple(
            value[0] if value is not None else float("nan")
            for value in functional_values
        ),
        start_h_f=tuple(
            value[1] if value is not None else float("nan")
            for value in functional_values
        ),
        start_c_assign=tuple(
            value[2] if value is not None else float("nan")
            for value in functional_values
        ),
    )


def _fit_prepared(
    data: _EdgeData,
    *,
    rank: int,
    n_starts: int,
    tolerance: float,
    max_iterations: int,
    seed: int,
) -> LowRankPluginResult:
    if rank < 0:
        raise ValueError("rank cannot be negative.")
    if rank > min(data.sample.workers - 1, data.sample.firms - 1):
        raise ValueError(
            "rank cannot exceed min(n_workers - 1, n_firms - 1) on the "
            "retained analysis sample."
        )
    if n_starts < 1:
        raise ValueError("n_starts must be positive.")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive.")

    additive_worker, additive_firm, additive_converged, additive_iterations = (
        _fit_additive(
            data,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    )
    if rank == 0:
        start = _StartFit(
            worker_level=additive_worker,
            firm_level=additive_firm,
            worker_factors=np.empty((data.sample.workers, 0)),
            firm_factors=np.empty((data.sample.firms, 0)),
            singular_values=np.empty(0),
            objective=_edge_objective(
                data,
                additive_worker,
                additive_firm,
            ),
            converged=additive_converged,
            iterations=additive_iterations,
        )
        return _result_from_starts(data, rank=rank, starts=[start])

    spectral_worker, spectral_firm = _spectral_initialization(
        data,
        additive_worker,
        additive_firm,
        rank,
    )
    residual_scale = np.sqrt(
        max(
            _edge_objective(
                data,
                additive_worker,
                additive_firm,
            )
            / data.sample.observations,
            np.finfo(float).eps,
        )
    )
    factor_scale = (residual_scale**2 / rank) ** 0.25
    rng = np.random.default_rng(seed)
    starts: list[_StartFit] = []

    for start_index in range(n_starts):
        if start_index == 0:
            initial_worker_factors = spectral_worker
            initial_firm_factors = spectral_firm
        else:
            initial_worker_factors = spectral_worker + rng.normal(
                scale=0.05 * factor_scale,
                size=(data.sample.workers, rank),
            )
            initial_firm_factors = spectral_firm + rng.normal(
                scale=0.05 * factor_scale,
                size=(data.sample.firms, rank),
            )
        starts.append(
            _fit_start(
                data,
                rank=rank,
                initial_worker_level=additive_worker,
                initial_firm_level=additive_firm,
                initial_worker_factors=initial_worker_factors,
                initial_firm_factors=initial_firm_factors,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
        )

    return _result_from_starts(data, rank=rank, starts=starts)


def fit_low_rank_plugin(
    panel: PanelData,
    *,
    rank: int,
    minimum_degree: int | None = None,
    n_starts: int = 3,
    tolerance: float = 1e-6,
    max_iterations: int = 300,
    seed: int = 2026,
) -> LowRankPluginResult:
    """Fit the manuscript's fixed-rank plug-in estimator.

    Period observations are collapsed to worker--firm match means and weighted
    by match counts. By default, the largest connected ``(rank + 2)``-degree
    support core is retained for positive ranks, giving one more match than the
    conditional-regression identification minimum; set ``minimum_degree``
    explicitly to study weaker graph designs. Multiple starts address the
    non-convex factorization problem. ``panel.outcome`` is assumed to have
    already removed observed covariates; the current simulation DGP has no
    nonzero ``X beta`` term.
    """

    if minimum_degree is None:
        minimum_degree = 1 if rank == 0 else rank + 2
    if minimum_degree < rank + 1:
        raise ValueError(
            "minimum_degree must be at least rank + 1 for the conditional "
            "weighted least-squares updates."
        )
    data = _prepare_edge_data(panel, minimum_degree=minimum_degree)
    return _fit_prepared(
        data,
        rank=rank,
        n_starts=n_starts,
        tolerance=tolerance,
        max_iterations=max_iterations,
        seed=seed,
    )


def select_low_rank_bic(
    panel: PanelData,
    *,
    candidate_ranks: tuple[int, ...] = (0, 1, 2),
    minimum_degree: int | None = None,
    n_starts: int = 3,
    tolerance: float = 1e-6,
    max_iterations: int = 300,
    seed: int = 2026,
) -> BICRankSelectionResult:
    """Select rank by an explicitly exploratory observation-level BIC.

    All candidates use the support core required by the largest candidate, so
    their criteria are comparable. This is a simulation convenience, not part
    of the unfinished LOO theory.
    """

    if not candidate_ranks:
        raise ValueError("candidate_ranks cannot be empty.")
    candidates = tuple(sorted(set(candidate_ranks)))
    if candidates[0] < 0:
        raise ValueError("candidate ranks cannot be negative.")
    if minimum_degree is None:
        minimum_degree = max(candidates) + 2
    if minimum_degree < max(candidates) + 1:
        raise ValueError(
            "minimum_degree must be at least max(candidate_ranks) + 1."
        )
    data = _prepare_edge_data(panel, minimum_degree=minimum_degree)
    estimates = tuple(
        _fit_prepared(
            data,
            rank=rank,
            n_starts=n_starts,
            tolerance=tolerance,
            max_iterations=max_iterations,
            seed=seed + 10_000 * rank,
        )
        for rank in candidates
    )

    n_observations = data.sample.observations
    bic_values: list[float] = []
    for estimate in estimates:
        rank = estimate.rank
        degrees_of_freedom = (
            data.sample.workers
            + data.sample.firms
            - 1
            + rank
            * (
                data.sample.workers
                + data.sample.firms
                - 2
                - rank
            )
        )
        residual_sse = (
            estimate.edge_objective + data.within_match_sse
        )
        residual_variance = max(
            residual_sse / n_observations,
            np.finfo(float).eps,
        )
        bic_values.append(
            float(
                n_observations * np.log(residual_variance)
                + degrees_of_freedom * np.log(n_observations)
            )
        )

    selected_index = int(np.argmin(bic_values))
    return BICRankSelectionResult(
        candidate_ranks=candidates,
        bic_values=tuple(bic_values),
        estimates=estimates,
        selected_rank=candidates[selected_index],
    )
