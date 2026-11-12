"""
Patch-based image processing utilities for ReliefLink V5.

The model expects 256×256 RGB patches with float32 values in [0, 1].
Large images are tiled into overlapping patches; the tiles are blended
back together using a cosine (Hann) weighting window so that seams are
not visible in the output.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from inference.inference import ReliefLinkRestoration

PATCH_SIZE = 256
OVERLAP     = 64          # pixels of overlap on each side
STEP        = PATCH_SIZE - OVERLAP


def _hann_window(size: int) -> np.ndarray:
    """1-D Hann window, then formed into a 2-D weight map (H × W)."""
    w = np.hanning(size).astype(np.float32)
    return np.outer(w, w)


def pil_to_float(image: Image.Image) -> np.ndarray:
    """Convert PIL RGB image → float32 H×W×3 array in [0, 1]."""
    image = image.convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr


def float_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert float32 H×W×3 [0,1] → PIL RGB image."""
    arr = np.clip(arr, 0.0, 1.0)
    uint8 = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(uint8, mode="RGB")


def restore_image_patched(
    restorer: "ReliefLinkRestoration",
    pil_image: Image.Image,
) -> tuple[Image.Image, np.ndarray]:
    """
    Run patch-based inference on an image of arbitrary size.

    Returns
    -------
    restored_pil  : PIL.Image  – restored image at the original resolution
    artifact_logits : np.ndarray – logits from the *last* processed patch
        (shape [1, 7]).  The caller is responsible for converting these to
        per-image probabilities (sigmoid).  For a full image the logits
        are averaged across all patches.
    """
    arr = pil_to_float(pil_image)   # H × W × 3
    H, W = arr.shape[:2]

    # ---------- fast path: image is already ≤ PATCH_SIZE ----------
    if H <= PATCH_SIZE and W <= PATCH_SIZE:
        # Pad up to PATCH_SIZE with reflection to avoid edge artefacts
        pad_h = PATCH_SIZE - H
        pad_w = PATCH_SIZE - W
        if pad_h > 0 or pad_w > 0:
            arr_padded = np.pad(
                arr,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="reflect",
            )
        else:
            arr_padded = arr

        restored_padded, logits = restorer.restore(arr_padded)
        restored = restored_padded[:H, :W, :]
        return float_to_pil(restored), logits

    # ---------- patch-based path ----------
    window = _hann_window(PATCH_SIZE)               # (256, 256)
    window_3ch = window[:, :, np.newaxis]           # (256, 256, 1)

    accumulator  = np.zeros_like(arr, dtype=np.float64)
    weight_map   = np.zeros((H, W, 1), dtype=np.float64)
    all_logits: list[np.ndarray] = []

    # Compute patch start positions so the last patch always reaches the edge
    def _positions(total: int) -> list[int]:
        positions = list(range(0, total - PATCH_SIZE + 1, STEP))
        if not positions or positions[-1] + PATCH_SIZE < total:
            positions.append(max(total - PATCH_SIZE, 0))
        return positions

    ys = _positions(H)
    xs = _positions(W)

    for y in ys:
        for x in xs:
            patch = arr[y : y + PATCH_SIZE, x : x + PATCH_SIZE, :]

            # Pad if the patch is smaller than PATCH_SIZE (edge case)
            ph, pw = patch.shape[:2]
            if ph < PATCH_SIZE or pw < PATCH_SIZE:
                patch = np.pad(
                    patch,
                    ((0, PATCH_SIZE - ph), (0, PATCH_SIZE - pw), (0, 0)),
                    mode="reflect",
                )

            restored_patch, logits = restorer.restore(patch)
            restored_patch = restored_patch[:ph, :pw, :]  # un-pad

            # Build local window for this (possibly smaller) patch
            local_win = window[:ph, :pw, np.newaxis]

            accumulator[y : y + ph, x : x + pw, :] += (
                restored_patch.astype(np.float64) * local_win
            )
            weight_map[y : y + ph, x : x + pw, :] += local_win
            all_logits.append(logits)

    # Blend
    restored_arr = (accumulator / np.maximum(weight_map, 1e-8)).astype(np.float32)

    # Average logits across patches
    avg_logits = np.mean(np.stack(all_logits, axis=0), axis=0)

    return float_to_pil(restored_arr), avg_logits
