"""
Image quality metrics for ReliefLink V5.

All functions accept numpy float32 arrays in [0, 1] with shape H×W×C or H×W.
They return a scalar float.

These are full-reference metrics — both a degraded input and a spatially
aligned clean reference are required.  Do not call them without a reference.
"""

from __future__ import annotations

import math

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# PSNR
# ──────────────────────────────────────────────────────────────────────────────

def psnr(reference: np.ndarray, restored: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio in dB.
    Higher is better.  Typical range: 20–50 dB.
    """
    reference = np.asarray(reference, dtype=np.float64)
    restored  = np.asarray(restored,  dtype=np.float64)
    mse = np.mean((reference - restored) ** 2)
    if mse < 1e-15:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


# ──────────────────────────────────────────────────────────────────────────────
# SSIM  (simplified, pure-numpy implementation)
# ──────────────────────────────────────────────────────────────────────────────

def ssim(
    reference: np.ndarray,
    restored: np.ndarray,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    data_range: float = 1.0,
) -> float:
    """
    Structural Similarity Index Measure.
    Range [−1, 1]; 1 = identical.  Typical good values: > 0.9.
    """
    from scipy.ndimage import uniform_filter

    ref = np.asarray(reference, dtype=np.float64)
    res = np.asarray(restored,  dtype=np.float64)

    if ref.ndim == 3:
        # Average over channels
        scores = [
            _ssim_single(ref[..., c], res[..., c], window_size, k1, k2, data_range)
            for c in range(ref.shape[2])
        ]
        return float(np.mean(scores))
    return _ssim_single(ref, res, window_size, k1, k2, data_range)


def _ssim_single(
    ref: np.ndarray,
    res: np.ndarray,
    window_size: int,
    k1: float,
    k2: float,
    data_range: float,
) -> float:
    from scipy.ndimage import uniform_filter

    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2
    w  = window_size

    mu1    = uniform_filter(ref, w)
    mu2    = uniform_filter(res, w)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12   = mu1 * mu2

    sigma1_sq = uniform_filter(ref * ref, w) - mu1_sq
    sigma2_sq = uniform_filter(res * res, w) - mu2_sq
    sigma12   = uniform_filter(ref * res, w) - mu12

    num = (2.0 * mu12 + C1) * (2.0 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = num / np.maximum(den, 1e-15)
    return float(np.mean(ssim_map))


# ──────────────────────────────────────────────────────────────────────────────
# MAE / RMSE
# ──────────────────────────────────────────────────────────────────────────────

def mae(reference: np.ndarray, restored: np.ndarray) -> float:
    """Mean Absolute Error.  Lower is better."""
    ref = np.asarray(reference, dtype=np.float64)
    res = np.asarray(restored,  dtype=np.float64)
    return float(np.mean(np.abs(ref - res)))


def rmse(reference: np.ndarray, restored: np.ndarray) -> float:
    """Root Mean Squared Error.  Lower is better."""
    ref = np.asarray(reference, dtype=np.float64)
    res = np.asarray(restored,  dtype=np.float64)
    return float(math.sqrt(np.mean((ref - res) ** 2)))


# ──────────────────────────────────────────────────────────────────────────────
# SAM – Spectral Angle Mapper
# ──────────────────────────────────────────────────────────────────────────────

def sam(reference: np.ndarray, restored: np.ndarray) -> float:
    """
    Spectral Angle Mapper (mean per-pixel angle in radians).
    Lower is better.  0 = perfect spectral match.
    Requires multi-channel input (H × W × C).
    """
    ref = np.asarray(reference, dtype=np.float64)
    res = np.asarray(restored,  dtype=np.float64)

    if ref.ndim != 3 or ref.shape[2] < 2:
        raise ValueError("SAM requires a multi-channel (H×W×C) image.")

    dot    = np.sum(ref * res, axis=2)
    norm_r = np.linalg.norm(ref, axis=2)
    norm_s = np.linalg.norm(res, axis=2)

    denom  = norm_r * norm_s
    cos_th = np.where(denom > 1e-8, dot / denom, 1.0)
    cos_th = np.clip(cos_th, -1.0, 1.0)

    return float(np.mean(np.arccos(cos_th)))


# ──────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ──────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    reference: np.ndarray,
    restored: np.ndarray,
) -> dict[str, float]:
    """
    Compute all five quality metrics.

    Parameters
    ----------
    reference : float32 H×W×C, range [0, 1]
    restored  : float32 H×W×C, range [0, 1]

    Returns
    -------
    dict with keys: PSNR_dB, SSIM, MAE, RMSE, SAM
    """
    return {
        "PSNR_dB": psnr(reference, restored),
        "SSIM":    ssim(reference, restored),
        "MAE":     mae(reference,  restored),
        "RMSE":    rmse(reference, restored),
        "SAM":     sam(reference,  restored),
    }
