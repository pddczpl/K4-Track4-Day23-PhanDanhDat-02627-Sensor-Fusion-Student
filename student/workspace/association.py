"""Measurement-to-track association via Mahalanobis gating and greedy matching.

Part F supplies the association stage shown in docs/HUONG_DAN_KY_THUAT.md §2.
Load ``kalman`` with ``load_workspace_module`` for innovation helpers and tracking parameters
for the chi-square gate.
"""

from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np
from scipy.stats import chi2


def _get_kalman():
    """Load the workspace kalman module at call time to avoid stale imports."""
    from fusion_lab.workspace_loader import load_workspace_module
    return load_workspace_module("kalman")


def mahalanobis_distance(track: Any, meas: Any) -> float:
    """Return squared Mahalanobis distance between a track and a measurement.

    Args:
        track: Track with ``x``, ``P``.
        meas: Measurement with ``sensor``.

    Returns:
        Scalar squared Mahalanobis distance.
    """
    kalman = _get_kalman()
    H = meas.sensor.get_H(track.x)
    gamma = kalman.innovation(track.x, meas)
    S = kalman.innovation_covariance(track.P, meas, H)
    dist_sq = float((gamma.T @ np.linalg.inv(S) @ gamma).item())
    return dist_sq


def chi2_gate(mhd_sq: float, sensor: Any) -> bool:
    """Return True if squared Mahalanobis distance lies inside the chi-square gate.

    Args:
        mhd_sq: Squared Mahalanobis distance.
        sensor: Sensor with ``dim_meas``.

    Returns:
        True if inside gate.
    """
    from fusion_lab.workspace_support import get_tracking_params
    params = get_tracking_params()
    threshold = chi2.ppf(params.gating_threshold, sensor.dim_meas)
    return bool(mhd_sq < threshold)


def association_cost_matrix(
    track_list: Sequence[Any], meas_list: Sequence[Any]
) -> np.matrix:
    """Build gated costs, checking each sensor's visibility before projection.

    Args:
        track_list: Active tracks.
        meas_list: Measurements for this sensor pass.

    Returns:
        Cost matrix; ``np.inf`` for invisible tracks or rejected chi-square gates.
        Invisible pairs must never call the Mahalanobis/projection helpers.
    """
    n_tracks = len(track_list)
    n_meas = len(meas_list)
    cost = np.asmatrix(np.full((n_tracks, n_meas), np.inf))

    for i, track in enumerate(track_list):
        for j, meas in enumerate(meas_list):
            # Check FOV before computing Mahalanobis distance
            if not meas.sensor.in_fov(track.x):
                continue
            # Compute Mahalanobis distance and apply chi2 gate
            mhd_sq = mahalanobis_distance(track, meas)
            if chi2_gate(mhd_sq, meas.sensor):
                cost[i, j] = mhd_sq

    return cost


def pick_next_pair(
    association_matrix: np.matrix,
    unassigned_tracks: Sequence[Any],
    unassigned_meas: Sequence[Any],
) -> tuple[Any, Any, np.matrix, list[Any], list[Any]]:
    """Pick the minimum-cost track/measurement pair and shrink the association problem.

    Args:
        association_matrix: Current cost matrix.
        unassigned_tracks: Track objects still free.
        unassigned_meas: Measurement objects still free.

    Returns:
        Tuple (track, meas, new_matrix, remaining_tracks, remaining_meas).
        If no finite pair exists, return np.nan for track and meas and retain both lists.
    """
    # Find minimum finite value
    if association_matrix.size == 0 or not np.isfinite(association_matrix).any():
        return np.nan, np.nan, association_matrix, list(unassigned_tracks), list(unassigned_meas)

    # Find position of minimum value
    idx = np.argmin(association_matrix)
    row, col = np.unravel_index(idx, association_matrix.shape)
    min_val = association_matrix[row, col]

    if not np.isfinite(min_val):
        return np.nan, np.nan, association_matrix, list(unassigned_tracks), list(unassigned_meas)

    track = unassigned_tracks[row]
    meas = unassigned_meas[col]

    # Remove selected row and column from the matrix and lists
    remaining_tracks = [t for i, t in enumerate(unassigned_tracks) if i != row]
    remaining_meas = [m for j, m in enumerate(unassigned_meas) if j != col]

    new_matrix = np.delete(np.delete(association_matrix, row, axis=0), col, axis=1)

    return track, meas, new_matrix, remaining_tracks, remaining_meas


def associate_and_update(
    manager: Any,
    meas_list: Sequence[Any],
    filter_obj: Any,
    sensor: Any,
) -> None:
    """Greedy association loop with EKF updates and track management.

    Args:
        manager: Track manager (``track_list``, ``manage_tracks``, ...).
        meas_list: Lidar or camera measurements for this frame pass.
        filter_obj: Filter with ``predict`` / ``update``.
        sensor: Explicit lidar/camera pass sensor, including empty measurement frames.

    Returns:
        None; updates tracks in place and always finishes the lifecycle pass.
        Visibility is handled in the cost matrix, before pair removal. Camera
        updates refine state only; lidar hits alone increase existence scores.
    """
    unassigned_tracks = list(manager.track_list)
    unassigned_meas = list(meas_list)

    if unassigned_tracks and unassigned_meas:
        cost_matrix = association_cost_matrix(unassigned_tracks, unassigned_meas)

        while True:
            track, meas, cost_matrix, unassigned_tracks, unassigned_meas = pick_next_pair(
                cost_matrix, unassigned_tracks, unassigned_meas
            )
            if (isinstance(track, (float, np.floating)) and np.isnan(track)) or (
                isinstance(meas, (float, np.floating)) and np.isnan(meas)
            ):
                break
            # Update EKF with the matched measurement
            filter_obj.update(track, meas)
            manager.handle_updated_track(track, sensor)

    # Always call manage_tracks at the end of the pass (even with empty lists)
    manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)
