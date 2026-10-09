"""Track initialization, scoring, and deletion helpers.

Part H supplies lidar-driven existence decisions (docs/HUONG_DAN_KY_THUAT.md §2).
Use tracking parameters for the score window, thresholds, and covariance limit.
"""

from __future__ import annotations

from typing import Any

from fusion_lab.workspace_support import get_tracking_params
import numpy as np


def init_track_state_from_meas(meas: Any) -> dict[str, Any]:
    """Initialize track state, covariance, lifecycle state, and score from a measurement.

    Args:
        meas: Lidar measurement with ``z``, ``R``, ``sensor``.

    Returns:
        Dict with keys ``x``, ``P``, ``state``, ``score`` (matrices as ``np.matrix``).
    """
    params = get_tracking_params()

    # Convert measurement position to vehicle frame using sens_to_veh transform
    transform = np.asarray(meas.sensor.sens_to_veh)
    meas_pos = np.asarray(meas.z).reshape(-1)
    pos_veh = transform[:3, :3] @ meas_pos + transform[:3, 3]

    # State: [px, py, pz, vx, vy, vz] — initialize velocity to zero
    x = np.asmatrix(np.r_[pos_veh, [0.0, 0.0, 0.0]]).T

    # Position covariance: rotate lidar R into vehicle frame
    R_lidar = np.asarray(meas.R)
    rot = transform[:3, :3]
    P_pos = rot @ R_lidar @ rot.T

    # Velocity covariance: broad initial uncertainty
    P_vel = np.diag([params.sigma_p44**2, params.sigma_p55**2, params.sigma_p66**2])

    # Assemble 6x6 covariance matrix
    P = np.asmatrix(np.zeros((6, 6)))
    P[:3, :3] = P_pos
    P[3:, 3:] = P_vel

    # Initial score and state
    score = 1.0 / params.window
    state = "initialized"

    return {"x": x, "P": P, "state": state, "score": score}


def update_track_score(track: dict[str, Any], associated: bool) -> dict[str, Any]:
    """Update existence once per lidar frame; camera passes never call this helper.

    A hit adds 1/window, capped at one; an in-FOV miss subtracts 1/window.
    Confirm above confirmed_threshold, and preserve confirmed state after misses.

    Args:
        track: Dict-like track with ``score``, ``state``.
        associated: True for a lidar hit; False for a lidar miss within the lidar FOV.

    Returns:
        Updated track dict.
    """
    params = get_tracking_params()
    score = track["score"]
    state = track["state"]

    if associated:
        # Hit: add 1/window, cap at 1
        score = min(1.0, score + 1.0 / params.window)
    else:
        # Miss within FOV: subtract 1/window
        score = score - 1.0 / params.window

    # Update state: confirm if score exceeds threshold
    if score > params.confirmed_threshold:
        state = "confirmed"
    elif state == "confirmed":
        # Preserve confirmed state after misses (don't downgrade)
        pass
    elif state == "initialized" and associated:
        state = "tentative"
    elif state == "initialized":
        # initialized miss stays initialized
        pass
    else:
        state = "tentative"

    # Mutate in place and also return
    track["score"] = score
    track["state"] = state
    return track


def should_delete_track(track: dict[str, Any]) -> bool:
    """Return whether a lidar lifecycle pass should remove this track.

    Delete if either horizontal variance exceeds max_P, or if a confirmed
    track has score < delete_threshold, or an unconfirmed track has score <= 0.
    Camera passes never trigger deletion.

    Args:
        track: Dict with ``score``, ``state``, ``P``.

    Returns:
        True if track should be removed.
    """
    params = get_tracking_params()
    score = track["score"]
    state = track["state"]
    P = track["P"]

    # Condition 1: horizontal variance exceeds max_P (Pxx or Pyy > max_P)
    if float(P[0, 0]) > params.max_P or float(P[1, 1]) > params.max_P:
        return True

    # Condition 2: confirmed track with score below delete_threshold
    if state == "confirmed" and score < params.delete_threshold:
        return True

    # Condition 3: unconfirmed track with score <= 0
    if state != "confirmed" and score <= 0:
        return True

    return False
