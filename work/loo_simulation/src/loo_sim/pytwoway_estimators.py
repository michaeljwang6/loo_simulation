"""Thin, explicit wrappers around the PyTwoWay comparison estimators."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .panel import PanelData


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


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


@dataclass(frozen=True)
class BLMAnalysisSample:
    """Two-period sample retained by the BLM preparation pipeline."""

    observations: int
    spells: int
    event_rows: int
    mover_rows: int
    stayer_rows: int
    workers: int
    firms: int
    firm_groups: int
    stayer_group_count: int
    mover_pair_count: int
    first_period: int | None
    second_period: int | None


@dataclass(frozen=True)
class BLMSupportDiagnostics:
    """Observed firm-class support required by PyTwoWay's BLM updates."""

    stayer_groups: tuple[int, ...]
    missing_stayer_groups: tuple[int, ...]
    mover_pairs: tuple[tuple[int, int], ...]
    missing_mover_pairs: tuple[tuple[int, int], ...]

    @property
    def complete(self) -> bool:
        """Whether every stayer class and mover class-pair is observed."""

        return not self.missing_stayer_groups and not self.missing_mover_pairs


class BLMSupportError(ValueError):
    """Raised before BLM fitting when required observed cells are absent."""

    def __init__(
        self,
        reason_code: str,
        diagnostics: BLMSupportDiagnostics,
        sample: BLMAnalysisSample,
    ) -> None:
        self.reason_code = reason_code
        self.diagnostics = diagnostics
        self.sample = sample
        period_label = (
            [sample.first_period, sample.second_period]
            if sample.first_period is not None
            else "legacy_full_history"
        )
        super().__init__(
            f"BLM_SUPPORT[{reason_code}]: "
            f"missing_stayer_groups="
            f"{list(diagnostics.missing_stayer_groups)}; "
            f"missing_mover_pairs={list(diagnostics.missing_mover_pairs)}; "
            f"stayer_rows={sample.stayer_rows}; "
            f"mover_rows={sample.mover_rows}; "
            f"periods={period_label}"
        )


@dataclass(frozen=True)
class PreparedBLMData:
    """Mover/stayer event-study inputs and their sample accounting."""

    variant: str
    jdata: Any
    sdata: Any
    sample: BLMAnalysisSample
    support: BLMSupportDiagnostics


@dataclass(frozen=True)
class BLMResult:
    """Native PyTwoWay BLM grouped-mixture estimates."""

    variant: str
    n_worker_types: int
    n_firm_types: int
    sample: BLMAnalysisSample
    a1: FloatArray
    a2: FloatArray
    s1: FloatArray
    s2: FloatArray
    pk1: FloatArray
    pk0: FloatArray
    mover_log_likelihood: float
    stayer_log_likelihood: float
    connectedness: float
    mover_iterations: int
    stayer_iterations: int
    mover_min_likelihood_change: float
    stayer_min_likelihood_change: float
    mover_likelihood_monotone: bool
    stayer_likelihood_monotone: bool

    @property
    def stationary_cell_means(self) -> FloatArray:
        """Average of BLM's first- and second-period cell means."""

        return 0.5 * (self.a1 + self.a2)


@dataclass(frozen=True)
class BLMCellMeanAlignment:
    """Label-aligned comparison of an estimated and true BLM cell table."""

    aligned_estimate: FloatArray
    worker_permutation: tuple[int, ...]
    firm_permutation: tuple[int, ...]
    rmse: float
    max_absolute_error: float


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
            # PyTwoWay's FE Monte Carlo keeps return spells. Dropping every
            # returner can erase the mover network in longer simulated panels.
            "drop_returns": False,
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


def _two_period_blm_events(
    clustered: Any,
    *,
    first_period: int,
    second_period: int,
    pd: Any,
    bpd: Any,
) -> tuple[Any, Any, Any]:
    """Construct exactly one static-BLM event row per observed worker."""

    paired_source = clustered.loc[
        clustered.loc[:, "t"].isin((first_period, second_period)),
        ["i", "j", "y", "t", "g"],
    ]
    paired = (
        paired_source
        .pivot(index="i", columns="t", values=["j", "y", "g"])
        .dropna()
    )
    if len(paired) == 0:
        raise ValueError(
            "No workers are observed in both requested BLM periods."
        )
    event_frame = pd.DataFrame(
        {
            "i": paired.index.to_numpy(dtype=np.int64),
            "j1": paired.loc[:, ("j", first_period)].to_numpy(
                dtype=np.int64
            ),
            "j2": paired.loc[:, ("j", second_period)].to_numpy(
                dtype=np.int64
            ),
            "y1": paired.loc[:, ("y", first_period)].to_numpy(dtype=float),
            "y2": paired.loc[:, ("y", second_period)].to_numpy(dtype=float),
            "t1": np.full(len(paired), first_period, dtype=np.int64),
            "t2": np.full(len(paired), second_period, dtype=np.int64),
            "g1": paired.loc[:, ("g", first_period)].to_numpy(
                dtype=np.int64
            ),
            "g2": paired.loc[:, ("g", second_period)].to_numpy(
                dtype=np.int64
            ),
        }
    )
    event_frame.loc[:, "m"] = (
        event_frame.loc[:, "j1"].to_numpy()
        != event_frame.loc[:, "j2"].to_numpy()
    ).astype(np.int64)
    event_data = bpd.BipartiteDataFrame(
        event_frame,
        track_id_changes=True,
    )
    mover = event_data.loc[:, "m"].to_numpy(dtype=np.int64) > 0
    return (
        event_data,
        event_data.loc[mover, :].copy(),
        event_data.loc[~mover, :].copy(),
    )


def prepare_blm_data(
    panel: PanelData,
    *,
    n_firm_types: int,
    firm_groups: IntArray | None = None,
    periods: tuple[int, int] | None = (0, 1),
    cdf_resolution: int = 10,
    seed: int = 2026,
) -> PreparedBLMData:
    """Prepare a static two-period mover/stayer sample for BLM.

    BLM's static model classifies a worker using one declared pair of
    periods. A worker is a stayer when the firm is unchanged across that
    pair, even if the worker moves elsewhere in another panel period. This
    function deliberately constructs one event-study row per worker instead
    of collapsing the worker's complete firm history into spells.
    ``periods=None`` is retained only to reproduce the superseded
    full-history support audit; new estimation must declare two periods.
    """

    if n_firm_types < 1:
        raise ValueError("n_firm_types must be positive.")
    if cdf_resolution < 2:
        raise ValueError("cdf_resolution must be at least two.")
    pd, bpd, _ = _load_pytwoway()
    frame = pd.DataFrame(panel.as_columns())
    if periods is None:
        first_period = None
        second_period = None
    else:
        if len(periods) != 2:
            raise ValueError("BLM periods must contain exactly two entries.")
        first_period, second_period = (int(period) for period in periods)
        if first_period >= second_period:
            raise ValueError(
                "BLM periods must be two distinct values in increasing order."
            )
        available_periods = set(
            int(value) for value in frame.loc[:, "t"].unique()
        )
        missing_periods = sorted(
            {first_period, second_period}.difference(available_periods)
        )
        if missing_periods:
            raise ValueError(
                "BLM periods are absent from the panel: "
                f"{missing_periods}."
            )
    if firm_groups is None:
        variant = "estimated_firm_groups"
    else:
        variant = "oracle_firm_groups"
        groups = np.asarray(firm_groups, dtype=np.int64)
        if groups.ndim != 1:
            raise ValueError("firm_groups must be one-dimensional.")
        if panel.firm_id.size and int(panel.firm_id.max()) >= groups.size:
            raise ValueError(
                "firm_groups does not cover every observed firm id."
            )
        observed_firms = frame.loc[:, "j"].to_numpy(dtype=np.int64)
        observed_groups = groups[observed_firms]
        labels = np.unique(observed_groups)
        if labels.size != n_firm_types:
            raise ValueError(
                "The observed panel must contain every requested oracle "
                "firm group."
            )
        frame.loc[:, "g"] = np.searchsorted(labels, observed_groups)

    adata = bpd.BipartiteDataFrame(frame, track_id_changes=True)
    cleaned = adata.clean(
        bpd.clean_params(
            {
                "connectedness": None,
                "drop_returns": (
                    "returners" if periods is None else False
                ),
                "copy": True,
                "verbose": False,
            }
        )
    )
    analysis_data = (
        cleaned.collapse(is_sorted=True, copy=True)
        if periods is None
        else cleaned
    )
    if firm_groups is None:
        os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
        clustered = analysis_data.cluster(
            bpd.cluster_params(
                {
                    "measures": bpd.measures.CDFs(
                        cdf_resolution=cdf_resolution
                    ),
                    "grouping": bpd.grouping.KMeans(
                        n_clusters=n_firm_types,
                        n_init=10,
                    ),
                    "is_sorted": True,
                    "copy": True,
                }
            ),
            rng=np.random.default_rng(seed),
        )
    else:
        clustered = analysis_data

    observed_cluster_count = int(clustered.loc[:, "g"].nunique())
    if observed_cluster_count != n_firm_types:
        raise ValueError(
            "BLM preparation produced "
            f"{observed_cluster_count} firm groups; expected {n_firm_types}."
        )

    if periods is None:
        event_data = clustered.to_eventstudy(is_sorted=True, copy=True)
        mover = event_data.get_worker_m(is_sorted=True)
        jdata = event_data.loc[mover, :]
        sdata = event_data.loc[~mover, :]
        observations = int(round(float(clustered.loc[:, "w"].sum())))
        spells = int(len(clustered))
    else:
        if first_period is None or second_period is None:  # pragma: no cover
            raise RuntimeError("Validated BLM periods unexpectedly missing.")
        event_data, jdata, sdata = _two_period_blm_events(
            clustered,
            first_period=first_period,
            second_period=second_period,
            pd=pd,
            bpd=bpd,
        )
        observations = 2 * int(len(event_data))
        spells = int(len(event_data) + len(jdata))
    support = compute_blm_support_diagnostics(
        mover_origins=jdata.loc[:, "g1"].to_numpy(dtype=np.int64),
        mover_destinations=jdata.loc[:, "g2"].to_numpy(dtype=np.int64),
        stayer_groups=sdata.loc[:, "g1"].to_numpy(dtype=np.int64),
        n_firm_types=n_firm_types,
    )
    sample = BLMAnalysisSample(
        observations=observations,
        spells=spells,
        event_rows=int(len(event_data)),
        mover_rows=int(len(jdata)),
        stayer_rows=int(len(sdata)),
        workers=int(event_data.loc[:, "i"].nunique()),
        firms=int(
            np.unique(
                np.concatenate(
                    [
                        event_data.loc[:, "j1"].to_numpy(dtype=np.int64),
                        event_data.loc[:, "j2"].to_numpy(dtype=np.int64),
                    ]
                )
            ).size
        ),
        firm_groups=observed_cluster_count,
        stayer_group_count=len(support.stayer_groups),
        mover_pair_count=len(support.mover_pairs),
        first_period=first_period,
        second_period=second_period,
    )
    if len(jdata) == 0:
        reason_code = "no_mover_events"
    elif len(sdata) == 0:
        reason_code = "no_stayer_events"
    elif support.missing_stayer_groups and support.missing_mover_pairs:
        reason_code = "missing_stayer_classes_and_mover_pairs"
    elif support.missing_stayer_groups:
        reason_code = "missing_stayer_classes"
    elif support.missing_mover_pairs:
        reason_code = "missing_mover_pairs"
    else:
        reason_code = ""
    if reason_code:
        raise BLMSupportError(reason_code, support, sample)

    return PreparedBLMData(
        variant=variant,
        jdata=jdata,
        sdata=sdata,
        sample=sample,
        support=support,
    )


def compute_blm_support_diagnostics(
    *,
    mover_origins: ArrayLike,
    mover_destinations: ArrayLike,
    stayer_groups: ArrayLike,
    n_firm_types: int,
) -> BLMSupportDiagnostics:
    """Enumerate the observed support used in PyTwoWay BLM probability rows.

    PyTwoWay 0.3.21 reshapes mover counts to ``(nl, nk**2)`` and stayer
    counts to ``(nl, nk)`` without padding absent categories.  Requiring all
    firm-class cells here prevents missing categories from becoming either a
    label-dependent reshape exception or an invalid zero-probability row.
    """

    if n_firm_types < 1:
        raise ValueError("n_firm_types must be positive.")
    origins = np.asarray(mover_origins, dtype=np.int64)
    destinations = np.asarray(mover_destinations, dtype=np.int64)
    stayers = np.asarray(stayer_groups, dtype=np.int64)
    if origins.ndim != 1 or destinations.ndim != 1 or stayers.ndim != 1:
        raise ValueError("BLM support inputs must be one-dimensional.")
    if origins.shape != destinations.shape:
        raise ValueError("Mover origin and destination arrays must align.")
    for name, values in (
        ("mover origins", origins),
        ("mover destinations", destinations),
        ("stayer groups", stayers),
    ):
        if values.size and (values.min() < 0 or values.max() >= n_firm_types):
            raise ValueError(f"{name} contain an out-of-range firm class.")

    expected_groups = set(range(n_firm_types))
    observed_groups = set(int(value) for value in stayers)
    expected_pairs = set(product(range(n_firm_types), repeat=2))
    observed_pairs = set(
        zip(
            (int(value) for value in origins),
            (int(value) for value in destinations),
        )
    )
    return BLMSupportDiagnostics(
        stayer_groups=tuple(sorted(observed_groups)),
        missing_stayer_groups=tuple(sorted(expected_groups - observed_groups)),
        mover_pairs=tuple(sorted(observed_pairs)),
        missing_mover_pairs=tuple(sorted(expected_pairs - observed_pairs)),
    )


def estimate_blm(
    panel: PanelData,
    *,
    n_worker_types: int,
    n_firm_types: int,
    firm_groups: IntArray | None = None,
    periods: tuple[int, int] = (0, 1),
    n_init: int = 4,
    n_best: int = 2,
    n_iterations: int = 250,
    threshold: float = 1e-6,
    cdf_resolution: int = 10,
    seed: int = 2026,
) -> BLMResult:
    """Run PyTwoWay BLM with estimated or supplied oracle firm groups."""

    if n_worker_types < 1:
        raise ValueError("n_worker_types must be positive.")
    if n_init < 1:
        raise ValueError("n_init must be positive.")
    if n_best < 1 or n_best > n_init:
        raise ValueError("n_best must lie between one and n_init.")
    if n_iterations < 1:
        raise ValueError("n_iterations must be positive.")
    if threshold <= 0:
        raise ValueError("threshold must be positive.")

    prepared = prepare_blm_data(
        panel,
        n_firm_types=n_firm_types,
        firm_groups=firm_groups,
        periods=periods,
        cdf_resolution=cdf_resolution,
        seed=seed,
    )
    _, _, tw = _load_pytwoway()
    params = tw.blm_params(
        {
            "nl": n_worker_types,
            "nk": n_firm_types,
            "verbose": 0,
            "n_iters_movers": n_iterations,
            "threshold_movers": threshold,
            "n_iters_stayers": n_iterations,
            "threshold_stayers": threshold,
            "weighted": True,
            "normalize": True,
        }
    )
    estimator = tw.BLMEstimator(params)
    estimator.fit(
        jdata=prepared.jdata,
        sdata=prepared.sdata,
        n_init=n_init,
        n_best=n_best,
        ncore=1,
        rng=np.random.default_rng(seed + 1),
    )
    model = estimator.model
    if model is None:
        raise RuntimeError("PyTwoWay BLM returned no fitted model.")

    mover_path = np.asarray(model.liks1, dtype=float)
    stayer_path = np.asarray(model.liks0, dtype=float)
    mover_changes = np.diff(mover_path)
    stayer_changes = np.diff(stayer_path)
    monotonicity_tolerance = max(1e-4, threshold)
    return BLMResult(
        variant=prepared.variant,
        n_worker_types=n_worker_types,
        n_firm_types=n_firm_types,
        sample=prepared.sample,
        a1=np.asarray(model.A1, dtype=float).copy(),
        a2=np.asarray(model.A2, dtype=float).copy(),
        s1=np.asarray(model.S1, dtype=float).copy(),
        s2=np.asarray(model.S2, dtype=float).copy(),
        pk1=np.asarray(model.pk1, dtype=float).copy(),
        pk0=np.asarray(model.pk0, dtype=float).copy(),
        mover_log_likelihood=float(model.lik1),
        stayer_log_likelihood=float(model.lik0),
        connectedness=float(model.connectedness),
        mover_iterations=int(mover_path.size),
        stayer_iterations=int(stayer_path.size),
        mover_min_likelihood_change=float(
            np.min(mover_changes) if mover_changes.size else 0.0
        ),
        stayer_min_likelihood_change=float(
            np.min(stayer_changes) if stayer_changes.size else 0.0
        ),
        mover_likelihood_monotone=bool(
            mover_path.size < 2
            or np.all(mover_changes >= -monotonicity_tolerance)
        ),
        stayer_likelihood_monotone=bool(
            stayer_path.size < 2
            or np.all(stayer_changes >= -monotonicity_tolerance)
        ),
    )


def align_blm_cell_means(
    estimated: ArrayLike,
    truth: ArrayLike,
    *,
    allow_firm_permutation: bool = True,
) -> BLMCellMeanAlignment:
    """Resolve latent worker and optional firm label permutations by RMSE."""

    estimated_array = np.asarray(estimated, dtype=float)
    truth_array = np.asarray(truth, dtype=float)
    if estimated_array.shape != truth_array.shape:
        raise ValueError("estimated and truth cell tables must have equal shape.")
    if estimated_array.ndim != 2:
        raise ValueError("BLM cell tables must be two-dimensional.")

    worker_permutations = permutations(range(estimated_array.shape[0]))
    firm_permutations = (
        permutations(range(estimated_array.shape[1]))
        if allow_firm_permutation
        else [tuple(range(estimated_array.shape[1]))]
    )
    firm_permutations = list(firm_permutations)
    best_error = float("inf")
    best_aligned: FloatArray | None = None
    best_worker: tuple[int, ...] | None = None
    best_firm: tuple[int, ...] | None = None

    for worker_permutation in worker_permutations:
        for firm_permutation in firm_permutations:
            aligned = estimated_array[
                np.ix_(worker_permutation, firm_permutation)
            ]
            error = float(np.mean((aligned - truth_array) ** 2))
            if error < best_error:
                best_error = error
                best_aligned = aligned.copy()
                best_worker = tuple(worker_permutation)
                best_firm = tuple(firm_permutation)

    if best_aligned is None or best_worker is None or best_firm is None:
        raise RuntimeError("BLM label alignment failed.")
    return BLMCellMeanAlignment(
        aligned_estimate=best_aligned,
        worker_permutation=best_worker,
        firm_permutation=best_firm,
        rmse=float(np.sqrt(best_error)),
        max_absolute_error=float(
            np.max(np.abs(best_aligned - truth_array))
        ),
    )
