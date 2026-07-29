"""Finite worker--firm panel sampling from a complete simulated economy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .dgp import PopulationDGP


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class PanelData:
    """Long-format arrays for one balanced simulated worker panel."""

    worker_id: IntArray
    firm_id: IntArray
    period: IntArray
    outcome: FloatArray
    systematic_wage: FloatArray
    error: FloatArray

    @property
    def n_observations(self) -> int:
        return int(self.outcome.size)

    @property
    def n_workers(self) -> int:
        return int(np.unique(self.worker_id).size)

    @property
    def n_firms_observed(self) -> int:
        return int(np.unique(self.firm_id).size)

    @property
    def mover_share(self) -> float:
        order = np.lexsort((self.period, self.worker_id))
        workers = self.worker_id[order]
        firms = self.firm_id[order]
        worker_change = workers[1:] != workers[:-1]
        firm_change = firms[1:] != firms[:-1]
        moved_worker_ids = np.unique(workers[1:][firm_change & ~worker_change])
        return float(moved_worker_ids.size / self.n_workers)

    def as_columns(self) -> dict[str, IntArray | FloatArray]:
        """Return columns using the names expected by BipartitePandas."""

        return {
            "i": self.worker_id.copy(),
            "j": self.firm_id.copy(),
            "t": self.period.copy(),
            "y": self.outcome.copy(),
        }


def sample_panel(
    population: PopulationDGP,
    *,
    n_periods: int = 6,
    redraw_probability: float = 0.35,
    error_sd: float = 1.0,
    seed: int = 54321,
) -> PanelData:
    """Sample a balanced short panel from the population assignment law.

    Each worker's initial firm is drawn from the worker-specific conditional
    assignment distribution. In later periods the worker redraws from the
    same distribution with ``redraw_probability`` and otherwise remains at
    the previous firm. A redraw may return the worker to the same firm, so
    the realized mover share is weakly below the redraw probability.
    """

    if n_periods < 2:
        raise ValueError("n_periods must be at least two.")
    if not 0 <= redraw_probability <= 1:
        raise ValueError("redraw_probability must lie in [0, 1].")
    if error_sd < 0:
        raise ValueError("error_sd cannot be negative.")

    rng = np.random.default_rng(seed)
    n_workers, n_firms = population.schedule.shape
    conditionals = (
        population.assignment / population.assignment.sum(axis=1)[:, None]
    )

    firm_history = np.empty((n_workers, n_periods), dtype=np.int64)
    for worker in range(n_workers):
        firm_history[worker, 0] = rng.choice(
            n_firms, p=conditionals[worker]
        )
        for period in range(1, n_periods):
            if rng.random() < redraw_probability:
                firm_history[worker, period] = rng.choice(
                    n_firms, p=conditionals[worker]
                )
            else:
                firm_history[worker, period] = firm_history[worker, period - 1]

    worker_id = np.repeat(np.arange(n_workers, dtype=np.int64), n_periods)
    period = np.tile(np.arange(n_periods, dtype=np.int64), n_workers)
    firm_id = firm_history.reshape(-1)
    systematic_wage = population.schedule[worker_id, firm_id]
    error = rng.normal(scale=error_sd, size=systematic_wage.size)
    outcome = systematic_wage + error

    return PanelData(
        worker_id=worker_id,
        firm_id=firm_id,
        period=period,
        outcome=outcome,
        systematic_wage=systematic_wage,
        error=error,
    )
