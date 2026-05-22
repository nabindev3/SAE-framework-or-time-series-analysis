"""Feature extractors for the difficulty probe.

Three feature families, all computed per forecast window:
  - input_statistics: cheap classical stats of the raw context series. This is
    the baseline that matters - if these alone predict difficulty, the model's
    internals add nothing.
  - aggregate_sequence: pool a (seq, d) activation/code tensor over the time
    axis as concat(mean, max, last). Mean-pooling alone destroys the temporal
    localization (a regime shift at token 400) that makes SAE features
    interesting, so we keep max and last too. The SAME pooling is applied to
    raw activations and SAE codes so the comparison is fair.
"""
import numpy as np
from scipy import signal


def input_statistics(context: np.ndarray, season_length: int = 24) -> np.ndarray:
    """Classical stats of one raw context window. Returns a fixed-length vector.

    Order is fixed (probe relies on it): [variance, volatility, ac_lag1,
    ac_seasonal, spectral_entropy, trend_slope, range, stationarity_stat].
    `stationarity_stat` is the statsmodels ADF p-value if statsmodels is
    installed, else a scipy-only variance-ratio proxy (lower = more
    non-stationary in both conventions, so the probe can use it either way).
    """
    x = np.asarray(context, dtype=np.float64)
    n = len(x)
    var = float(np.var(x))
    volatility = float(np.mean(np.abs(np.diff(x)))) if n > 1 else 0.0

    def autocorr(lag):
        if n <= lag:
            return 0.0
        a, b = x[:-lag] - x.mean(), x[lag:] - x.mean()
        denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
        return float(np.sum(a * b) / denom) if denom > 0 else 0.0

    ac_lag1 = autocorr(1)
    ac_seasonal = autocorr(season_length)

    if n >= 8:
        f, pxx = signal.periodogram(x - x.mean())
        pxx = pxx[1:]
        if pxx.sum() > 0:
            p = pxx / pxx.sum()
            spectral_entropy = float(-np.sum(p * np.log(p + 1e-12)) / np.log(len(p)))
        else:
            spectral_entropy = 0.0
    else:
        spectral_entropy = 0.0

    t = np.arange(n)
    trend_slope = float(np.polyfit(t, x, 1)[0]) if n > 1 else 0.0
    rng = float(x.max() - x.min())

    stationarity_stat = _stationarity(x)

    return np.array([
        var, volatility, ac_lag1, ac_seasonal,
        spectral_entropy, trend_slope, rng, stationarity_stat
    ], dtype=np.float64)


INPUT_STAT_NAMES = [
    "variance", "volatility", "ac_lag1", "ac_seasonal",
    "spectral_entropy", "trend_slope", "range", "stationarity_stat",
]


def _stationarity(x: np.ndarray) -> float:
    try:
        from statsmodels.tsa.stattools import adfuller
        return float(adfuller(x, autolag="AIC")[1])  # ADF p-value
    except Exception:
        # Variance-ratio proxy: var(second half) / var(first half).
        # ~1 for stationary, far from 1 for level/variance drift.
        h = len(x) // 2
        if h < 2:
            return 1.0
        v1 = np.var(x[:h]) + 1e-8
        v2 = np.var(x[h:]) + 1e-8
        return float(v2 / v1)


def aggregate_sequence(seq_tensor: np.ndarray) -> np.ndarray:
    """(N, seq, d) -> (N, 3*d) as concat(mean, max, last) over the time axis."""
    if seq_tensor.ndim != 3:
        raise ValueError(f"expected (N, seq, d), got {seq_tensor.shape}")
    mean = seq_tensor.mean(axis=1)
    mx = seq_tensor.max(axis=1)
    last = seq_tensor[:, -1, :]
    return np.concatenate([mean, mx, last], axis=1)
