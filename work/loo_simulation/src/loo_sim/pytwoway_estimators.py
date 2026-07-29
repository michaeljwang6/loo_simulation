"""Thin, explicit wrappers around the PyTwoWay comparison estimators."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np

from .panel import PanelData


@dataclass(frozen=True)
class AnalysisSample:
    """Size of the panel that remains after an estimator's required cleaning."""

    rows: int
    observations: int
    workers: int
    firms: int


@dataclass(frozen=True)
class FEKSSResult:
    """AKM plug-in and KSS bias-corrected two-way fixed-effect moments."""

    sample: AnalysisSample
    var_y: float
    var_psi_fe: float
    var_psi_ho: float
    var_psi_he: float
    cov_psi_alpha_fe: float
    cov_psi_alpha_ho: float
    cov_psi_alpha_he: float
    var_eps_fe: float
    var_eps_ho: float
    var_eps_he: float
    min_leverage: float
    max_leverage: float


@dataclass(frozen=True)
class BS20Result:
    """Native Borovičková--Shimer wage-type moments."""

    sample: AnalysisSample
    mean_y: float
    var_y: float
    var_worker_type: float
    var_firm_type: float
    covariance: float
    correlation: float


def _load_pytwoway() -> tuple[Any, Any, Any]:
    """Import optional estimator dependencies with a writable plotting cache."""

    project_root = Path(__file__).resolve().parents[2]
    cache = project_root / ".mplconfig"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))

    try:
        import bipartitepandas as bpd
        import pandas as pd
        import pytwoway as tw
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "PyTwoWay comparisons require the optional estimator dependencies. "
            "Install this project with the 'estimators' extra."
        ) from exc
    return pd, bpd, tw


def panel_to_bipartite(panel: PanelData) -> Any:
    """Convert a simulated panel to PyTwoWay's long-format data container."""

    pd, bpd, _ = _load_pytwoway()
    frame = pd.DataFrame(panel.as_columns())
    return bpd.BipartiteDataFrame(frame, track_id_changes=True)


def _sample_size(adata: Any) -> AnalysisSample:
    observations = (
        int(round(float(adata.loc[:, "w"].sum())))
        if "w" in adata.columns
        else int(len(adata))
    )
    return AnalysisSample(
        rows=int(len(adata)),
        observations=observations,
        workers=int(adata.loc[:, "i"].nunique()),
        firms=int(adata.loc[:, "j"].nunique()),
    )


def prepare_fe_sample(panel: PanelData) -> Any:
    """Apply the leave-out-spell cleaning used for FE/KSS estimation."""

    _, bpd, _ = _load_pytwoway()
    adata = panel_to_bipartite(panel)
    params = bpd.clean_params(
        {
            "connectedness": "leave_out_spell",
            "collapse_at_connectedness_measure": True,
            "drop_single_stayers": True,
            "drop_returns": "returners",
            "copy": True,
            "verbose": False,
        }
    )
    cleaned = adata.clean(params)
    if len(cleaned) == 0:
        raise ValueError("FE/KSS cleaning removed the entire simulated panel.")
    return cleaned


def estimate_fe_kss(
    panel: PanelData,
    *,
    seed: int = 2026,
    exact: bool = True,
) -> FEKSSResult:
    """Run PyTwoWay's AKM plug-in, homoskedastic, and KSS-HE corrections.

    ``exact=True`` uses analytical traces and leverages. This is deterministic
    and appropriate for validation pilots; larger Monte Carlo runs can use
    randomized approximations by setting it to ``False``.
    """

    _, _, tw = _load_pytwoway()
    adata = prepare_fe_sample(panel)
    params = tw.fe_params(
        {
            "ho": True,
            "he": True,
            "exact_trace_sigma_2": exact,
            "exact_trace_ho": exact,
            "exact_trace_he": exact,
            "exact_lev_he": exact,
            "ncore": 1,
            "preconditioner": None,
            "progress_bars": False,
            "verbose": False,
        }
    )
    estimator = tw.FEEstimator(adata, params)
    estimator.fit(np.random.default_rng(seed))
    result = estimator.res

    return FEKSSResult(
        sample=_sample_size(adata),
        var_y=float(result["var(y)"]),
        var_psi_fe=float(result["var(psi)_fe"]),
        var_psi_ho=float(result["var(psi)_ho"]),
        var_psi_he=float(result["var(psi)_he"]),
        cov_psi_alpha_fe=float(result["cov(psi, alpha)_fe"]),
        cov_psi_alpha_ho=float(result["cov(psi, alpha)_ho"]),
        cov_psi_alpha_he=float(result["cov(psi, alpha)_he"]),
        var_eps_fe=float(result["var(eps)_fe"]),
        var_eps_ho=float(result["var(eps)_ho"]),
        var_eps_he=float(result["var(eps)_he"]),
        min_leverage=float(result["min_lev"]),
        max_leverage=float(result["max_lev"]),
    )


def prepare_bs20_sample(panel: PanelData) -> Any:
    """Apply the no-return, spell-level cleaning required by BS20."""

    _, bpd, _ = _load_pytwoway()
    adata = panel_to_bipartite(panel)
    params = bpd.clean_params(
        {
            "connectedness": "strongly_connected",
            "drop_single_stayers": True,
            "drop_returns": "returns",
            "copy": True,
            "verbose": False,
        }
    )
    cleaned = adata.clean(params)
    collapsed = cleaned.collapse(level="spell", copy=True)
    joint = collapsed.min_joint_obs_frame(
        threshold_1=2,
        threshold_2=2,
        id_col_1="j",
        id_col_2="i",
        copy=True,
    )
    final = joint.clean(
        bpd.clean_params(
            {
                "drop_returns": "returns",
                "is_sorted": False,
                "copy": True,
                "verbose": False,
            }
        )
    )
    if len(final) == 0:
        raise ValueError("BS20 cleaning removed the entire simulated panel.")
    return final


def estimate_bs20(panel: PanelData) -> BS20Result:
    """Run PyTwoWay's standard weighted Borovičková--Shimer estimator."""

    _, _, tw = _load_pytwoway()
    adata = prepare_bs20_sample(panel)
    sample = _sample_size(adata)
    wages = adata.loc[:, "y"].to_numpy(dtype=float)
    weights = adata.loc[:, "w"].to_numpy(dtype=float)
    mean_y = float(np.average(wages, weights=weights))
    var_y = float(np.average((wages - mean_y) ** 2, weights=weights))

    # BSEstimator temporarily mutates and sorts its input.
    estimator = tw.BSEstimator()
    estimator.fit(adata.copy(), alternative_estimator=False, weighted=True)
    result = estimator.res

    return BS20Result(
        sample=sample,
        mean_y=mean_y,
        # PyTwoWay 0.3.21 computes this diagnostic after internally replacing
        # spell wages by w*y. Recompute the conventional duration-weighted
        # variance here; the native BS type moments below remain PyTwoWay's.
        var_y=var_y,
        var_worker_type=float(result["var(lambda)"]),
        var_firm_type=float(result["var(mu)"]),
        covariance=float(result["cov(lambda, mu)"]),
        correlation=float(result["corr(lambda, mu)"]),
    )
