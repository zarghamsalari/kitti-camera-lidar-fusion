"""Frustum depth-cleaning methods for removing background contamination.

Each method takes Nx3 LiDAR points (in Velodyne frame, x-forward) and returns
a boolean mask selecting the *retained* inlier points.
"""

from __future__ import annotations

import numpy as np


def clean_raw(points: np.ndarray) -> np.ndarray:
    """No cleaning -- keep all points."""
    return np.ones(points.shape[0], dtype=bool)


def clean_iqr(points: np.ndarray, k: float = 1.5) -> np.ndarray:
    """Remove depth outliers using the interquartile range (IQR) method.

    Points whose depth (x-coordinate) falls outside [Q1 - k*IQR, Q3 + k*IQR]
    are discarded.
    """
    if points.shape[0] < 4:
        return np.ones(points.shape[0], dtype=bool)
    depths = points[:, 0]
    q1, q3 = np.percentile(depths, [25, 75])
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (depths >= lower) & (depths <= upper)


def clean_peak(points: np.ndarray, bin_width: float = 1.0,
               window: float = 2.0) -> np.ndarray:
    """Select points near the dominant depth-histogram peak.

    Builds a histogram with *bin_width* metre bins, finds the peak bin, and
    keeps points within *window* metres of the peak centre.
    """
    if points.shape[0] < 3:
        return np.ones(points.shape[0], dtype=bool)
    depths = points[:, 0]
    lo, hi = float(depths.min()), float(depths.max())
    if hi - lo < bin_width:
        return np.ones(points.shape[0], dtype=bool)
    n_bins = max(int(np.ceil((hi - lo) / bin_width)), 1)
    counts, edges = np.histogram(depths, bins=n_bins)
    peak_idx = int(np.argmax(counts))
    peak_centre = (edges[peak_idx] + edges[peak_idx + 1]) / 2.0
    return np.abs(depths - peak_centre) <= window


def clean_dbscan(points: np.ndarray, eps: float = 1.5,
                 min_samples: int = 5) -> np.ndarray:
    """Cluster frustum points and keep the nearest dense cluster.

    Uses a simple grid-based density approach to avoid sklearn dependency.
    Falls back to IQR if the point cloud is too small.
    """
    if points.shape[0] < min_samples:
        return np.ones(points.shape[0], dtype=bool)

    try:
        from sklearn.cluster import DBSCAN
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points[:, :3])
    except ImportError:
        # Fallback: use IQR cleaning if sklearn not available
        return clean_iqr(points)

    if np.all(labels == -1):
        return np.ones(points.shape[0], dtype=bool)

    # Select cluster whose median depth is closest (smallest x)
    unique_labels = [l for l in np.unique(labels) if l != -1]
    best_label = min(unique_labels, key=lambda l: np.median(points[labels == l, 0]))
    return labels == best_label


# Registry for easy lookup by name
METHODS = {
    "raw": clean_raw,
    "iqr": clean_iqr,
    "peak": clean_peak,
    "dbscan": clean_dbscan,
}


def apply_cleaning(points: np.ndarray, method: str = "raw") -> np.ndarray:
    """Apply a named cleaning method, return boolean inlier mask."""
    fn = METHODS.get(method)
    if fn is None:
        raise ValueError(f"Unknown cleaning method '{method}'. Choose from: {list(METHODS)}")
    return fn(points)
