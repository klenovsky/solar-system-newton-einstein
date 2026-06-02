#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit web application: Newtonian vs. pairwise 1PN Solar-System model.

Run locally:
    streamlit run app.py

Deploy:
    Put app.py and requirements.txt into a GitHub repository and deploy the
    repository on Streamlit Community Cloud.

Model summary
-------------
Units:
    length = AU, time = Julian year, mass = solar mass.

The left panel integrates Newtonian N-body gravity.  The right panel integrates
Newtonian gravity plus a pairwise two-body first post-Newtonian (1PN) correction.
The 1PN part is intended for visualization of weak relativistic corrections. It
is not a full Einstein-Infeld-Hoffmann many-body ephemeris and it is not a JPL
Horizons replacement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import math
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    # The app still works in manual-frame mode if the optional package is absent.
    HAS_AUTOREFRESH = False



# =============================================================================
# Physical constants and default body data
# =============================================================================

G_MODEL = 4.0 * math.pi * math.pi       # AU^3 / (M_sun yr^2)
C_REAL_AU_PER_YR = 63241.07708426628    # physical speed of light in AU/year
SOFTENING_AU = 1.0e-6                   # purely numerical softening
PLOT_UIREVISION = "solar-system-fixed-user-view"  # keep manual Plotly zoom/camera across reruns
DAYS_PER_YEAR = 365.25
SUN_RADIUS_KM = 696_340.0
MSUN_KG = 1.98847e30


@dataclass(frozen=True)
class BodyData:
    name: str
    mass_msun: float
    radius_km: float
    semi_major_au: float
    inclination_deg: float
    phase_deg: float
    color: str


# Masses/radii are standard rounded planetary physical parameters.  The orbital
# elements are simplified mean/semi-major-axis values used only to create
# didactic initial conditions, not a date-specific ephemeris.
PLANET_BODIES: tuple[BodyData, ...] = (
    BodyData("Sun", 1.0, SUN_RADIUS_KM, 0.0, 0.0, 0.0, "gold"),
    BodyData("Mercury", 0.330103e24 / MSUN_KG, 2439.4, 0.38709927, 7.00497902, 252.25032350, "dimgray"),
    BodyData("Venus", 4.86731e24 / MSUN_KG, 6051.8, 0.72333566, 3.39467605, 181.97909950, "orange"),
    BodyData("Earth", 5.97217e24 / MSUN_KG, 6371.0084, 1.00000261, -0.00001531, 100.46457166, "royalblue"),
    BodyData("Mars", 0.641691e24 / MSUN_KG, 3389.50, 1.52371034, 1.84969142, -4.55343205, "red"),
    BodyData("Jupiter", 1898.125e24 / MSUN_KG, 69911.0, 5.20288700, 1.30439695, 34.39644051, "sienna"),
    BodyData("Saturn", 568.317e24 / MSUN_KG, 58232.0, 9.53667594, 2.48599187, 49.95424423, "peru"),
    BodyData("Uranus", 86.8099e24 / MSUN_KG, 25362.0, 19.18916464, 0.77263783, 313.23810451, "cyan"),
    BodyData("Neptune", 102.4092e24 / MSUN_KG, 24622.0, 30.06992276, 1.77004347, -55.12002969, "purple"),
)

# Optional non-planet objects.  They are included in the state vector only when
# the corresponding checkbox is enabled.  The radii are used only for visual
# marker scaling; the trajectories are point-mass trajectories.
VOYAGER_DRY_MASS_KG = 721.9
VOYAGER_LAUNCH_MASS_KG = 815.0
SL9_DEFAULT_MASS_KG = 1.0e13
EXTRA_BODIES: tuple[BodyData, ...] = (
    BodyData("Voyager 1-like probe", VOYAGER_DRY_MASS_KG / MSUN_KG, 0.005, 0.0, 0.0, 0.0, "magenta"),
    BodyData("SL9-like Jupiter-impact comet", SL9_DEFAULT_MASS_KG / MSUN_KG, 0.5, 0.0, 0.0, 0.0, "lime"),
)

BODIES: tuple[BodyData, ...] = PLANET_BODIES + EXTRA_BODIES
EARTH_IDX = 3
JUPITER_IDX = 5
VOYAGER_IDX = len(PLANET_BODIES)
SL9_IDX = len(PLANET_BODIES) + 1
SL9_START_OFFSET_AU = 1.0
PLANET_NAMES = tuple(body.name for body in PLANET_BODIES[1:])
BODY_NAMES = tuple(body.name for body in BODIES)
PLANET_MAX_RADIUS_KM = max(body.radius_km for body in PLANET_BODIES[1:])


# =============================================================================
# Low-level mechanics
# =============================================================================

def rotation_x(angle_rad: float) -> np.ndarray:
    """Rotation matrix around x; used to tilt simplified circular orbits."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)), dtype=float)


def barycentric_transform(pos: np.ndarray, vel: np.ndarray, masses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Move positions and velocities into the barycentric frame."""
    total_mass = float(np.sum(masses))
    r_cm = np.sum(pos * masses[:, None], axis=0) / total_mass
    v_cm = np.sum(vel * masses[:, None], axis=0) / total_mass
    return pos - r_cm[None, :], vel - v_cm[None, :]


def build_initial_conditions(
    sun_mass_log10: float,
    planet_mass_log10: Sequence[float],
    planet_distance_scale: Sequence[float],
    include_voyager: bool,
    voyager_mass_log10kg: float,
    voyager_velocity_earth_frame: Sequence[float],
    include_sl9: bool,
    sl9_mass_log10kg: float,
    sl9_velocity_jupiter_frame: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create simplified initial conditions for Sun + planets + optional objects.

    Planet positions start on tilted circular orbits with radii equal to
    semi_major_axis * distance_scale.  Velocities are circular Keplerian speeds
    for the scaled Sun mass and scaled planet mass.  This gives a clean didactic
    model, not a high-precision ephemeris for a specific date.

    Optional objects are deliberately simplified:
    - the Voyager 1-like probe starts near Earth, with a small numerical offset
      from Earth's center to avoid a singular initial separation, and receives a
      user-set velocity relative to Earth;
    - the SL9-like comet starts outside Jupiter's orbit and receives a user-set
      velocity relative to Jupiter, by default aimed toward Jupiter.
    """
    n = len(BODIES)
    masses = np.zeros(n, dtype=float)
    pos = np.zeros((n, 3), dtype=float)
    vel = np.zeros((n, 3), dtype=float)
    active = np.zeros(n, dtype=bool)
    active[: len(PLANET_BODIES)] = True

    sun_factor = 10.0 ** float(sun_mass_log10)
    masses[0] = max(sun_factor, 1.0e-15) * PLANET_BODIES[0].mass_msun

    for i, body in enumerate(PLANET_BODIES[1:], start=1):
        m_factor = 10.0 ** float(planet_mass_log10[i - 1])
        d_factor = max(float(planet_distance_scale[i - 1]), 1.0e-4)
        masses[i] = max(m_factor, 0.0) * body.mass_msun

        r = max(body.semi_major_au * d_factor, 1.0e-6)
        phase = math.radians(body.phase_deg)
        inc = math.radians(body.inclination_deg)

        # Circular orbit in the local orbital plane.
        local_pos = np.array((r * math.cos(phase), r * math.sin(phase), 0.0), dtype=float)
        speed = math.sqrt(G_MODEL * (masses[0] + masses[i]) / r)
        local_vel = np.array((-speed * math.sin(phase), speed * math.cos(phase), 0.0), dtype=float)

        rot = rotation_x(inc)
        pos[i] = rot @ local_pos
        vel[i] = rot @ local_vel

    # Insertion points for optional small bodies.  This is a didactic setup, not
    # a reconstructed Voyager launch/flyby trajectory and not a reconstructed
    # Shoemaker-Levy 9 pre-impact ephemeris.
    epos = pos[EARTH_IDX].copy()
    evel = vel[EARTH_IDX].copy()
    enorm = float(np.linalg.norm(epos))
    rhat_e = epos / enorm if enorm > 0.0 else np.array((1.0, 0.0, 0.0), dtype=float)

    jpos = pos[JUPITER_IDX].copy()
    jvel = vel[JUPITER_IDX].copy()
    jnorm = float(np.linalg.norm(jpos))
    rhat_j = jpos / jnorm if jnorm > 0.0 else np.array((1.0, 0.0, 0.0), dtype=float)

    if include_voyager:
        masses[VOYAGER_IDX] = (10.0 ** float(voyager_mass_log10kg)) / MSUN_KG
        active[VOYAGER_IDX] = True
        # Start essentially at Earth.  A small 0.002 AU offset avoids starting
        # exactly at Earth's center, which would create a numerical singularity
        # for a point-mass Earth in this simple model.
        pos[VOYAGER_IDX] = epos + 0.002 * rhat_e
        vel[VOYAGER_IDX] = evel + np.asarray(voyager_velocity_earth_frame, dtype=float)

    if include_sl9:
        masses[SL9_IDX] = (10.0 ** float(sl9_mass_log10kg)) / MSUN_KG
        active[SL9_IDX] = True
        # Start outside Jupiter's orbit and aim inward toward Jupiter by default.
        # Since the velocity is added to Jupiter's velocity, an exactly radial
        # default relative velocity gives a simple visual Jupiter-encounter path.
        pos[SL9_IDX] = jpos + SL9_START_OFFSET_AU * rhat_j
        vel[SL9_IDX] = jvel + np.asarray(sl9_velocity_jupiter_frame, dtype=float)

    pos, vel = barycentric_transform(pos, vel, masses)
    return pos, vel, masses, active


def acceleration_newton(pos: np.ndarray, masses: np.ndarray, active_mask: np.ndarray | None = None) -> np.ndarray:
    """Newtonian N-body acceleration with a tiny numerical softening.

    This vectorized implementation is numerically equivalent to the explicit
    double loop for the point-mass model used here, but it is noticeably faster
    on Streamlit Cloud when many RK4 steps are requested.
    """
    n = len(masses)
    acc = np.zeros_like(pos)
    if active_mask is None:
        active_mask = np.ones(n, dtype=bool)

    active_idx = np.where(active_mask & (masses > 0.0))[0]
    if active_idx.size <= 1:
        return acc

    p = pos[active_idx]
    m = masses[active_idx]
    dr = p[:, None, :] - p[None, :, :]
    r2 = np.einsum("ijk,ijk->ij", dr, dr) + SOFTENING_AU * SOFTENING_AU
    np.fill_diagonal(r2, np.inf)
    inv_r3 = 1.0 / (r2 * np.sqrt(r2))
    acc_active = -G_MODEL * np.sum(dr * inv_r3[:, :, None] * m[None, :, None], axis=1)
    acc[active_idx] = acc_active
    return acc


def acceleration_pairwise_1pn(
    pos: np.ndarray,
    vel: np.ndarray,
    masses: np.ndarray,
    c_au_per_year: float,
    pn_multiplier: float,
    active_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Newtonian N-body acceleration plus pairwise two-body 1PN corrections.

    For each active pair i,j the standard relative two-body 1PN correction is
    computed in harmonic-coordinate form and split between the two bodies so
    that the pair center-of-mass acceleration remains zero.

    This is useful pedagogically, especially for a Sun-dominated system, but it
    is not the full Einstein-Infeld-Hoffmann N-body equation because the genuine
    1PN three-body terms are not included.
    """
    n_bodies = len(masses)
    if active_mask is None:
        active_mask = np.ones(n_bodies, dtype=bool)
    acc = acceleration_newton(pos, masses, active_mask)
    if pn_multiplier == 0.0:
        return acc

    c2 = float(c_au_per_year) ** 2
    if c2 <= 0.0:
        return acc

    for i in range(n_bodies):
        if not bool(active_mask[i]):
            continue
        for j in range(i + 1, n_bodies):
            if not bool(active_mask[j]):
                continue
            mi = masses[i]
            mj = masses[j]
            mtot = mi + mj
            if mtot <= 0.0:
                continue

            dr = pos[i] - pos[j]
            r2 = float(np.dot(dr, dr)) + SOFTENING_AU * SOFTENING_AU
            r = math.sqrt(r2)
            nvec = dr / r
            vrel = vel[i] - vel[j]
            v2 = float(np.dot(vrel, vrel))
            rdot = float(np.dot(nvec, vrel))
            eta = (mi * mj) / (mtot * mtot)

            bracket = (
                nvec * ((4.0 + 2.0 * eta) * G_MODEL * mtot / r - (1.0 + 3.0 * eta) * v2 + 1.5 * eta * rdot * rdot)
                + (4.0 - 2.0 * eta) * rdot * vrel
            )
            a_rel_corr = (G_MODEL * mtot / (c2 * r2)) * bracket
            a_rel_corr *= pn_multiplier

            # Split the relative correction a_i - a_j while keeping
            # mi*a_i_corr + mj*a_j_corr = 0.
            acc[i] += (mj / mtot) * a_rel_corr
            acc[j] += -(mi / mtot) * a_rel_corr

    return acc


def rhs(
    state: np.ndarray,
    masses: np.ndarray,
    active_mask: np.ndarray,
    model: str,
    c_value: float,
    pn_multiplier: float,
) -> np.ndarray:
    """Right-hand side of the first-order ODE system."""
    n = len(masses)
    pos = state[: 3 * n].reshape((n, 3))
    vel = state[3 * n :].reshape((n, 3))
    if model == "newton":
        acc = acceleration_newton(pos, masses, active_mask)
    elif model == "1pn":
        acc = acceleration_pairwise_1pn(pos, vel, masses, c_value, pn_multiplier, active_mask)
    else:
        raise ValueError(f"unknown model: {model}")
    # Inactive optional objects stay fixed and do not contaminate diagnostics.
    if active_mask is not None:
        vel = vel.copy()
        acc = acc.copy()
        vel[~active_mask] = 0.0
        acc[~active_mask] = 0.0
    return np.concatenate((vel.reshape(-1), acc.reshape(-1)))


def rk4_step(
    state: np.ndarray,
    dt: float,
    masses: np.ndarray,
    active_mask: np.ndarray,
    model: str,
    c_value: float,
    pn_multiplier: float,
) -> np.ndarray:
    """One fourth-order Runge-Kutta time step."""
    k1 = rhs(state, masses, active_mask, model, c_value, pn_multiplier)
    k2 = rhs(state + 0.5 * dt * k1, masses, active_mask, model, c_value, pn_multiplier)
    k3 = rhs(state + 0.5 * dt * k2, masses, active_mask, model, c_value, pn_multiplier)
    k4 = rhs(state + dt * k3, masses, active_mask, model, c_value, pn_multiplier)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def diagnostics(pos: np.ndarray, vel: np.ndarray, masses: np.ndarray, active_mask: np.ndarray, c_value: float) -> dict[str, float]:
    """Return simple 1PN validity diagnostics for active bodies only."""
    if not np.any(active_mask):
        return {"max_v_over_c": 0.0, "max_GM_over_rc2": 0.0}

    speeds = np.linalg.norm(vel[active_mask], axis=1)
    max_v_over_c = float(np.max(speeds) / max(c_value, 1.0e-30))

    max_compactness = 0.0
    n = len(masses)
    for i in range(n):
        if not bool(active_mask[i]):
            continue
        for j in range(i + 1, n):
            if not bool(active_mask[j]):
                continue
            dr = pos[i] - pos[j]
            r = math.sqrt(float(np.dot(dr, dr)) + SOFTENING_AU * SOFTENING_AU)
            value_i = G_MODEL * masses[i] / (r * c_value * c_value)
            value_j = G_MODEL * masses[j] / (r * c_value * c_value)
            max_compactness = max(max_compactness, value_i, value_j)

    return {"max_v_over_c": max_v_over_c, "max_GM_over_rc2": float(max_compactness)}


@st.cache_data(show_spinner=False)
def simulate_cached(
    total_years: float,
    dt_days: float,
    frame_stride: int,
    sun_mass_log10: float,
    planet_mass_log10: tuple[float, ...],
    planet_distance_scale: tuple[float, ...],
    include_voyager: bool,
    voyager_mass_log10kg: float,
    voyager_vx: float,
    voyager_vy: float,
    voyager_vz: float,
    include_sl9: bool,
    sl9_mass_log10kg: float,
    sl9_vx: float,
    sl9_vy: float,
    sl9_vz: float,
    c_value: float,
    pn_log10: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float], dict[str, float]]:
    """Integrate Newton and 1PN models and return downsampled frames."""
    pos0, vel0, masses, active_mask = build_initial_conditions(
        sun_mass_log10,
        planet_mass_log10,
        planet_distance_scale,
        include_voyager,
        voyager_mass_log10kg,
        (voyager_vx, voyager_vy, voyager_vz),
        include_sl9,
        sl9_mass_log10kg,
        (sl9_vx, sl9_vy, sl9_vz),
    )
    n = len(masses)
    state_n = np.concatenate((pos0.reshape(-1), vel0.reshape(-1)))
    state_p = state_n.copy()

    dt = float(dt_days) / DAYS_PER_YEAR
    n_steps = int(math.ceil(float(total_years) / dt))
    frame_stride = max(int(frame_stride), 1)
    pn_multiplier = 10.0 ** float(pn_log10)

    times = []
    frames_n = []
    frames_p = []

    def store(step: int) -> None:
        times.append(step * dt)
        frames_n.append(state_n[: 3 * n].reshape((n, 3)).copy())
        frames_p.append(state_p[: 3 * n].reshape((n, 3)).copy())

    store(0)
    for step in range(1, n_steps + 1):
        state_n = rk4_step(state_n, dt, masses, active_mask, "newton", c_value, pn_multiplier)
        state_p = rk4_step(state_p, dt, masses, active_mask, "1pn", c_value, pn_multiplier)
        if step % frame_stride == 0 or step == n_steps:
            store(step)

    diag_n = diagnostics(frames_n[-1], state_n[3 * n :].reshape((n, 3)), masses, active_mask, c_value)
    diag_p = diagnostics(frames_p[-1], state_p[3 * n :].reshape((n, 3)), masses, active_mask, c_value)
    return (
        np.asarray(times),
        np.asarray(frames_n),
        np.asarray(frames_p),
        masses,
        active_mask,
        diag_n,
        diag_p,
    )


# =============================================================================
# Plot helpers
# =============================================================================

def visible_body_indices(view: str, include_voyager: bool, include_sl9: bool) -> list[int]:
    if view == "Inner planets":
        indices = list(range(0, 5))       # Sun through Mars
    elif view == "To Jupiter":
        indices = list(range(0, 6))       # Sun through Jupiter
    else:
        indices = list(range(0, len(PLANET_BODIES)))

    # Optional non-planet bodies are shown whenever enabled, even if the current
    # planet-region selection is otherwise limited to the inner planets.
    if include_voyager:
        indices.append(VOYAGER_IDX)
    if include_sl9:
        indices.append(SL9_IDX)
    return indices


def marker_sizes(
    indices: Iterable[int],
    gamma: float,
    sun_size: float,
    planet_min_size: float,
    planet_max_size: float,
) -> list[float]:
    sizes = []
    for idx in indices:
        body = BODIES[idx]
        if idx == 0:
            sizes.append(float(sun_size))
        else:
            normalized = max(body.radius_km / PLANET_MAX_RADIUS_KM, 1.0e-12)
            sizes.append(float(planet_min_size + (planet_max_size - planet_min_size) * normalized ** gamma))
    return sizes


def planet_indices_for_view(view: str) -> list[int]:
    """Planet-only indices that define the fixed plotting box for each view."""
    if view == "Inner planets":
        return list(range(0, 5))       # Sun through Mars
    if view == "To Jupiter":
        return list(range(0, 6))       # Sun through Jupiter
    return list(range(0, len(PLANET_BODIES)))


def fixed_axis_range_for_view(view: str, planet_distance_scale: Sequence[float]) -> tuple[float, float]:
    """Return a fixed AU range for the selected planet region.

    The optional Voyager/comet bodies are intentionally ignored in this mode.
    Otherwise a fast escaping probe or an incoming comet could enlarge the two
    3D boxes and make the planets appear to shrink. Users can still inspect
    objects outside the fixed box by manually zooming/panning the Plotly view.
    """
    indices = planet_indices_for_view(view)
    max_r = 1.0
    for idx in indices:
        if idx == 0:
            continue
        scale = float(planet_distance_scale[idx - 1]) if idx - 1 < len(planet_distance_scale) else 1.0
        max_r = max(max_r, PLANET_BODIES[idx].semi_major_au * max(scale, 1.0e-4))
    margin = 1.22 * max_r
    return -margin, margin


def symmetric_axis_range_from_points(points: np.ndarray, minimum_half_range: float = 1.0) -> tuple[float, float]:
    """Return a symmetric axis range enclosing an array of 3D points."""
    if points.size == 0:
        half = max(float(minimum_half_range), 1.0)
        return -half, half
    max_abs = float(np.nanmax(np.abs(points)))
    if not np.isfinite(max_abs):
        max_abs = float(minimum_half_range)
    half = max(1.15 * max_abs, float(minimum_half_range), 1.0)
    return -half, half


def axis_range_for_mode(
    mode: str,
    view: str,
    planet_distance_scale: Sequence[float],
    frames_n: np.ndarray,
    frames_p: np.ndarray,
    visible_indices: Sequence[int],
    frame_index: int,
    trail_frames: int,
) -> tuple[float, float]:
    """Return the common 3D axis range according to the selected scaling mode.

    Modes:
    - Fixed by selected region: use only the selected planet region; Voyager and
      the comet do not enlarge the box.
    - Fit full computed trajectory: one constant box enclosing all computed
      visible trajectories.
    - Dynamic auto-fit during playback: recompute the box from the current
      frame and visible trail.
    """
    fixed_range = fixed_axis_range_for_view(view, planet_distance_scale)
    min_half = max(abs(fixed_range[0]), abs(fixed_range[1]))
    if mode == "Fit full computed trajectory":
        pts = np.concatenate(
            (
                frames_n[:, visible_indices, :].reshape((-1, 3)),
                frames_p[:, visible_indices, :].reshape((-1, 3)),
            ),
            axis=0,
        )
        return symmetric_axis_range_from_points(pts, minimum_half_range=min_half)
    if mode == "Dynamic auto-fit during playback":
        fidx = int(np.clip(frame_index, 0, len(frames_n) - 1))
        sl = trail_slice(fidx, trail_frames)
        pts = np.concatenate(
            (
                frames_n[sl, visible_indices, :].reshape((-1, 3)),
                frames_p[sl, visible_indices, :].reshape((-1, 3)),
            ),
            axis=0,
        )
        return symmetric_axis_range_from_points(pts, minimum_half_range=min_half)
    return fixed_range


def axis_template_from_range(axis_range: tuple[float, float], dynamic: bool = False) -> dict:
    axis_min, axis_max = axis_range
    axis_common = dict(
        autorange=False,
        range=[float(axis_min), float(axis_max)],
        showspikes=False,
    )
    template = dict(
        xaxis=dict(title="x [AU]", **axis_common),
        yaxis=dict(title="y [AU]", **axis_common),
        zaxis=dict(title="z [AU]", **axis_common),
        aspectmode="cube",
    )
    if not dynamic:
        template["uirevision"] = PLOT_UIREVISION
    return template


def trail_slice(frame: int, trail_frames: int) -> slice:
    """Return only the already travelled part of a trajectory.

    The complete trajectory is precomputed after pressing Apply and recompute,
    but plotting uses this slice so the visible path grows behind the moving
    body.  No future part of the orbit is drawn ahead of the current time.
    ``trail_frames`` is a maximum trail length in stored/displayed frames.
    """
    fidx = max(int(frame), 0)
    start = max(0, fidx - max(int(trail_frames), 1) + 1)
    return slice(start, fidx + 1)



# =============================================================================
# User-interface language helpers
# =============================================================================

LANGUAGE_OPTIONS = ("English", "Čeština")

BODY_NAME_CS = {
    "Sun": "Slunce",
    "Mercury": "Merkur",
    "Venus": "Venuše",
    "Earth": "Země",
    "Mars": "Mars",
    "Jupiter": "Jupiter",
    "Saturn": "Saturn",
    "Uranus": "Uran",
    "Neptune": "Neptun",
    "Voyager 1-like probe": "Sonda podobná Voyageru 1",
    "SL9-like Jupiter-impact comet": "Kometa typu SL9 dopadající na Jupiter",
}

VIEW_LABELS = {
    "en": {
        "Inner planets": "Inner planets",
        "To Jupiter": "To Jupiter",
        "All planets": "All planets",
    },
    "cs": {
        "Inner planets": "Vnitřní planety",
        "To Jupiter": "Po Jupiter",
        "All planets": "Všechny planety",
    },
}

UI_TEXT = {
    "en": {
        "title": "Solar-System motion: Newton gravity vs. Einstein GTR 1PN approximation",
        "presets": "Presets",
        "reset_initial": "Reset to initial values",
        "what": "What this app computes",
        "global": "Global controls",
        "displayed_region": "Displayed region",
        "sim_time": "Simulated time [yr]",
        "rk4_dt": "RK4 time step [days]",
        "stride": "Integration steps per displayed frame",
        "trail": "Trail length [displayed frames]",
        "axis_scaling": "View-box scaling mode",
        "axis_fixed": "Fixed by selected region",
        "axis_full": "Fit full computed trajectory",
        "axis_dynamic": "Dynamic auto-fit during playback",
        "pn_params": "1PN parameters",
        "c_caption": "c = {c:,.1f} AU/yr; physical c ≈ {cphys:,.1f} AU/yr",
        "pn_caption": "1PN multiplier = {val:.3g}",
        "display_sizes": "Display sizes",
        "gamma": "Planet size compression gamma",
        "sun_marker": "Sun marker diameter [px]",
        "planet_min": "Minimum planet diameter [px]",
        "planet_max": "Largest planet diameter [px]",
        "mass_scaling": "Mass scaling",
        "sun_mass": "Sun: log10(M/M_real)",
        "planet_masses": "Individual planet masses",
        "planet_distances": "Individual planet distances",
        "optional": "Optional spacecraft / comet",
        "show_voyager": "Show Voyager 1-like probe",
        "voyager_title": "Voyager 1-like probe",
        "voyager_mass": "Voyager mass: log10(m [kg])",
        "voyager_caption": "Initial position: near Earth, with a small numerical offset from Earth's center. Velocity components are relative to Earth, in AU/yr.",
        "voyager_vx": "Voyager vx rel. Earth [AU/yr]",
        "voyager_vy": "Voyager vy rel. Earth [AU/yr]",
        "voyager_vz": "Voyager vz rel. Earth [AU/yr]",
        "show_sl9": "Show Jupiter-impact comet (Shoemaker–Levy 9-like)",
        "sl9_title": "Jupiter-impact comet / SL9-like body",
        "comet_mass": "Comet mass: log10(m [kg])",
        "sl9_caption": "Initial position: outside Jupiter's orbit, aimed toward Jupiter by default. Velocity components are relative to Jupiter, in AU/yr.",
        "comet_vx": "Comet vx rel. Jupiter [AU/yr]",
        "comet_vy": "Comet vy rel. Jupiter [AU/yr]",
        "comet_vz": "Comet vz rel. Jupiter [AU/yr]",
        "playback": "Playback",
        "step_caption": "Internal RK4 steps: {steps:,}; displayed frames: about {frames:,}",
        "live_refresh": "Live playback refresh [ms]",
        "frames_refresh": "Frames advanced per refresh",
        "loop": "Loop live playback",
        "plotly_play": "Also create Plotly chart Play button",
        "max_plotly": "Max Plotly animation frames",
        "apply_recompute": "Apply and recompute",
        "apply_help": "Change sliders freely; the trajectories are recomputed only after pressing Apply and recompute.",
        "too_many_steps": "The selected time span and time step would require more than 20,000 RK4 steps. Increase the time step, shorten the simulated time, or increase steps per displayed frame.",
        "spinner": "Integrating Newton and 1PN trajectories...",
        "live_playback": "Live playback",
        "start": "▶ Start",
        "pause": "⏸ Pause",
        "reset": "↺ Reset",
        "running": "running",
        "paused": "paused",
        "status": "Status: {status}; frame {frame}/{total}; t = {time:.2f} yr",
        "need_autorefresh": "Live playback requires the optional package streamlit-autorefresh. Install it or use the Plotly Play button in the chart.",
        "displayed_frame": "Displayed time frame",
        "axes_caption": "View-box mode: {mode}. In the fixed mode, the 3D axis range is set by the selected planet region and is not enlarged by Voyager/comet motion. In the full-trajectory and dynamic modes, the box may include the optional bodies. Use Plotly zoom/pan/rotate controls for manual viewing.",
        "progressive_caption": "The trajectories are precomputed after Apply and recompute, but the plotted trails are progressive: only the already travelled path up to the current time is drawn. Future orbit segments are not shown ahead of the moving bodies.",
        "displayed_time": "Displayed time",
        "sun_mass_scale": "Sun mass scale",
        "onepn_multiplier": "1PN multiplier",
        "diagnostics": "Approximation diagnostics",
        "warn_validity": "The chosen parameters push the system outside the comfortable weak-field / slow-motion 1PN regime. The visualization may still be interesting, but it should not be interpreted as a quantitatively valid relativistic model.",
        "current_params": "Current body parameters",
        "body": "body",
        "active": "active",
        "mass_value": "mass scale / value",
        "distance_start": "distance / start",
        "model_mass": "model mass [M_sun]",
        "center": "center",
        "near_earth": "near Earth",
        "outside_jupiter": "outside Jupiter orbit",
        "marker_caption": "Marker diameters are visually compressed and are not plotted on the same linear AU scale as the orbital distances. The compression preserves the ordering of body radii but is chosen so that both the Sun and the planets remain visible.",
        "newton_title": "Newton gravity",
        "pn_title": "Einstein GTR 1PN approximation",
        "fig_title": "Solar-System model",
        "play": "Play",
        "plot_pause": "Pause",
        "trail_word": "trail",
        "bodies_word": "bodies",
    },
    "cs": {
        "title": "Pohyb ve Sluneční soustavě: Newtonova gravitace vs. Einsteinova OTR 1PN aproximace",
        "presets": "Přednastavení",
        "reset_initial": "Obnovit výchozí hodnoty",
        "what": "Co tato aplikace počítá",
        "global": "Globální ovládání",
        "displayed_region": "Zobrazená oblast",
        "sim_time": "Simulovaný čas [rok]",
        "rk4_dt": "Časový krok RK4 [dny]",
        "stride": "Integračních kroků na zobrazený snímek",
        "trail": "Délka stopy [zobrazené snímky]",
        "axis_scaling": "Režim škálování boxu",
        "axis_fixed": "Pevně podle vybrané oblasti",
        "axis_full": "Přizpůsobit celé spočtené trajektorii",
        "axis_dynamic": "Dynamický auto-fit během přehrávání",
        "pn_params": "Parametry 1PN",
        "c_caption": "c = {c:,.1f} AU/rok; fyzikální c ≈ {cphys:,.1f} AU/rok",
        "pn_caption": "Násobek 1PN = {val:.3g}",
        "display_sizes": "Velikosti zobrazení",
        "gamma": "Komprese velikostí planet gamma",
        "sun_marker": "Průměr značky Slunce [px]",
        "planet_min": "Minimální průměr planety [px]",
        "planet_max": "Průměr největší planety [px]",
        "mass_scaling": "Škálování hmotností",
        "sun_mass": "Slunce: log10(M/M_real)",
        "planet_masses": "Hmotnosti jednotlivých planet",
        "planet_distances": "Vzdálenosti jednotlivých planet",
        "optional": "Volitelná sonda / kometa",
        "show_voyager": "Zobrazit sondu podobnou Voyageru 1",
        "voyager_title": "Sonda podobná Voyageru 1",
        "voyager_mass": "Hmotnost Voyageru: log10(m [kg])",
        "voyager_caption": "Počáteční poloha: blízko Země, s malým numerickým posunem od středu Země. Složky rychlosti jsou vztažené k Zemi, v AU/rok.",
        "voyager_vx": "Voyager vx vůči Zemi [AU/rok]",
        "voyager_vy": "Voyager vy vůči Zemi [AU/rok]",
        "voyager_vz": "Voyager vz vůči Zemi [AU/rok]",
        "show_sl9": "Zobrazit kometu dopadající na Jupiter (typ Shoemaker–Levy 9)",
        "sl9_title": "Kometa dopadající na Jupiter / těleso typu SL9",
        "comet_mass": "Hmotnost komety: log10(m [kg])",
        "sl9_caption": "Počáteční poloha: vně Jupiterovy dráhy, výchozí rychlost míří k Jupiteru. Složky rychlosti jsou vztažené k Jupiteru, v AU/rok.",
        "comet_vx": "Kometa vx vůči Jupiteru [AU/rok]",
        "comet_vy": "Kometa vy vůči Jupiteru [AU/rok]",
        "comet_vz": "Kometa vz vůči Jupiteru [AU/rok]",
        "playback": "Přehrávání",
        "step_caption": "Vnitřní kroky RK4: {steps:,}; zobrazených snímků přibližně: {frames:,}",
        "live_refresh": "Obnova živého přehrávání [ms]",
        "frames_refresh": "Snímků posunutých při jedné obnově",
        "loop": "Opakovat živé přehrávání ve smyčce",
        "plotly_play": "Vytvořit také tlačítko Play přímo v grafu Plotly",
        "max_plotly": "Maximální počet snímků animace Plotly",
        "apply_recompute": "Použít a přepočítat",
        "apply_help": "Slidery lze měnit libovolně; trajektorie se přepočítají až po stisku Použít a přepočítat.",
        "too_many_steps": "Zvolený časový rozsah a časový krok by vyžadovaly více než 20 000 kroků RK4. Zvětšete časový krok, zkraťte simulovaný čas nebo zvětšete počet kroků na zobrazený snímek.",
        "spinner": "Integruji trajektorie Newtonova a 1PN modelu...",
        "live_playback": "Živé přehrávání",
        "start": "▶ Spustit",
        "pause": "⏸ Pozastavit",
        "reset": "↺ Reset času",
        "running": "běží",
        "paused": "pozastaveno",
        "status": "Stav: {status}; snímek {frame}/{total}; t = {time:.2f} roku",
        "need_autorefresh": "Živé přehrávání vyžaduje volitelný balíček streamlit-autorefresh. Nainstalujte jej nebo použijte tlačítko Plotly Play v grafu.",
        "displayed_frame": "Zobrazený časový snímek",
        "axes_caption": "Režim boxu: {mode}. V pevném režimu je rozsah 3D os určen vybranou oblastí planet a nezvětšuje se pohybem Voyageru/komety. V režimu celé trajektorie a dynamického auto-fitu může box zahrnovat i volitelná tělesa. Viditelnou oblast lze měnit ručně pomocí zoomu/panu/rotace Plotly.",
        "displayed_time": "Zobrazený čas",
        "sun_mass_scale": "Škálování hmotnosti Slunce",
        "onepn_multiplier": "Násobek 1PN",
        "diagnostics": "Diagnostika aproximace",
        "warn_validity": "Zvolené parametry posouvají systém mimo pohodlný slabopolní / pomalý 1PN režim. Vizualizace může být stále zajímavá, ale neměla by být interpretována jako kvantitativně platný relativistický model.",
        "current_params": "Aktuální parametry těles",
        "body": "těleso",
        "active": "aktivní",
        "mass_value": "škálování hmotnosti / hodnota",
        "distance_start": "vzdálenost / start",
        "model_mass": "modelová hmotnost [M_sun]",
        "center": "střed",
        "near_earth": "blízko Země",
        "outside_jupiter": "vně Jupiterovy dráhy",
        "marker_caption": "Průměry značek jsou vizuálně komprimované a nejsou kreslené ve stejném lineárním měřítku AU jako orbitální vzdálenosti. Komprese zachovává pořadí poloměrů těles, ale je zvolena tak, aby bylo vidět Slunce i planety.",
        "newton_title": "Newtonova gravitace",
        "pn_title": "Einsteinova OTR 1PN aproximace",
        "fig_title": "Model Sluneční soustavy",
        "play": "Spustit",
        "plot_pause": "Pozastavit",
        "trail_word": "stopa",
        "bodies_word": "tělesa",
    },
}


def lang_code() -> str:
    return "cs" if st.session_state.get("language") == "Čeština" else "en"


def tr(lang: str, key: str, **kwargs: object) -> str:
    text = UI_TEXT[lang][key]
    return text.format(**kwargs) if kwargs else text


def view_label(view: str, lang: str) -> str:
    return VIEW_LABELS[lang].get(view, view)


def axis_mode_label(mode: str, lang: str) -> str:
    mapping = {
        "Fixed by selected region": tr(lang, "axis_fixed"),
        "Fit full computed trajectory": tr(lang, "axis_full"),
        "Dynamic auto-fit during playback": tr(lang, "axis_dynamic"),
    }
    return mapping.get(mode, mode)


def body_display_name(body: BodyData | str, lang: str) -> str:
    name = body.name if isinstance(body, BodyData) else str(body)
    return BODY_NAME_CS.get(name, name) if lang == "cs" else name

def make_figure(
    times: np.ndarray,
    frames_n: np.ndarray,
    frames_p: np.ndarray,
    frame_index: int,
    visible_indices: Sequence[int],
    trail_frames: int,
    sizes: Sequence[float],
    animate: bool,
    max_animation_frames: int,
    axis_scaling_mode: str,
    view: str,
    planet_distance_scale: Sequence[float],
    lang: str = "en",
) -> go.Figure:
    """Build static or animated Plotly figure."""
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(tr(lang, "newton_title"), tr(lang, "pn_title")),
        horizontal_spacing=0.02,
    )

    colors = [BODIES[i].color for i in visible_indices]
    names = [body_display_name(BODIES[i], lang) for i in visible_indices]
    # Only the optional small bodies get a visible black marker outline.
    # Regular Solar-System bodies are intentionally left without outlines.
    outline_colors = [
        "black" if i in (VOYAGER_IDX, SL9_IDX) else "rgba(0,0,0,0)"
        for i in visible_indices
    ]

    def add_model_traces(frames: np.ndarray, scene_col: int, model_prefix: str, initial_frame: int) -> None:
        sl = trail_slice(initial_frame, trail_frames)
        for idx in visible_indices:
            body = BODIES[idx]
            xyz = frames[sl, idx, :]
            fig.add_trace(
                go.Scatter3d(
                    x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                    mode="lines",
                    line=dict(width=2, color=body.color),
                    name=f"{model_prefix} {body_display_name(body, lang)} {tr(lang, 'trail_word')}",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1, col=scene_col,
            )
        pts = frames[initial_frame, visible_indices, :]
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers+text",
                marker=dict(size=list(sizes), color=colors, opacity=0.98, sizemode="diameter", line=dict(color=outline_colors, width=1)),
                text=names,
                textposition="top center",
                name=f"{model_prefix} {tr(lang, 'bodies_word')}",
                hovertemplate="%{text}<br>x=%{x:.3f} AU<br>y=%{y:.3f} AU<br>z=%{z:.3f} AU<extra></extra>",
                showlegend=False,
            ),
            row=1, col=scene_col,
        )

    frame_index = int(np.clip(frame_index, 0, len(times) - 1))
    add_model_traces(frames_n, 1, "Newton", frame_index)
    add_model_traces(frames_p, 2, "1PN", frame_index)

    dynamic_axes = axis_scaling_mode == "Dynamic auto-fit during playback"
    initial_axis_range = axis_range_for_mode(
        axis_scaling_mode, view, planet_distance_scale, frames_n, frames_p,
        visible_indices, frame_index, trail_frames
    )
    axis_template = axis_template_from_range(initial_axis_range, dynamic=dynamic_axes)
    layout_kwargs = dict(
        scene=axis_template,
        scene2=axis_template,
        height=760,
        margin=dict(l=5, r=5, t=70, b=5),
        title=tr(lang, "fig_title"),
        transition=dict(duration=0),
    )
    if not dynamic_axes:
        layout_kwargs["uirevision"] = PLOT_UIREVISION
    fig.update_layout(**layout_kwargs)

    if animate:
        n_total_frames = len(times)
        if n_total_frames <= max_animation_frames:
            selected_animation_frames = list(range(n_total_frames))
        else:
            selected_animation_frames = np.linspace(0, n_total_frames - 1, max_animation_frames).astype(int).tolist()
            selected_animation_frames = sorted(set(selected_animation_frames))

        trace_count_per_model = len(visible_indices) + 1
        total_trace_count = 2 * trace_count_per_model
        animation_frames = []
        for fidx in selected_animation_frames:
            frame_data = []
            for frames in (frames_n, frames_p):
                sl = trail_slice(fidx, trail_frames)
                for idx in visible_indices:
                    xyz = frames[sl, idx, :]
                    frame_data.append(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2]))
                pts = frames[fidx, visible_indices, :]
                frame_data.append(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], text=names))
            frame_axis_template = axis_template
            frame_layout_kwargs = {}
            if dynamic_axes:
                frame_axis_range = axis_range_for_mode(
                    axis_scaling_mode, view, planet_distance_scale, frames_n, frames_p,
                    visible_indices, fidx, trail_frames
                )
                frame_axis_template = axis_template_from_range(frame_axis_range, dynamic=True)
            else:
                frame_layout_kwargs["uirevision"] = PLOT_UIREVISION
            frame_layout = go.Layout(
                scene=frame_axis_template,
                scene2=frame_axis_template,
                **frame_layout_kwargs,
            )
            animation_frames.append(
                go.Frame(
                    data=frame_data,
                    traces=list(range(total_trace_count)),
                    name=str(fidx),
                    layout=frame_layout,
                )
            )

        fig.frames = animation_frames
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.02,
                    y=1.08,
                    xanchor="left",
                    yanchor="top",
                    buttons=[
                        dict(label=tr(lang, "play"), method="animate", args=[None, {"frame": {"duration": 70, "redraw": True}, "transition": {"duration": 0}, "fromcurrent": True}]),
                        dict(label=tr(lang, "plot_pause"), method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
                    ],
                )
            ],
            sliders=[
                dict(
                    active=0,
                    x=0.1,
                    y=0.01,
                    len=0.8,
                    steps=[
                        dict(
                            method="animate",
                            label=f"{times[fidx]:.1f}",
                            args=[[str(fidx)], {"frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}, "mode": "immediate"}],
                        )
                        for fidx in selected_animation_frames
                    ],
                )
            ],
        )

    return fig



# =============================================================================
# UI defaults and reset handling
# =============================================================================

VIEW_OPTIONS = ("Inner planets", "To Jupiter", "All planets")
AXIS_SCALING_OPTIONS = ("Fixed by selected region", "Fit full computed trajectory", "Dynamic auto-fit during playback")



def default_orbit_unit_vectors(body_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Return radial and tangential unit vectors for a default circular orbit."""
    body = PLANET_BODIES[body_index]
    phase = math.radians(body.phase_deg)
    inc = math.radians(body.inclination_deg)
    local_rhat = np.array((math.cos(phase), math.sin(phase), 0.0), dtype=float)
    local_that = np.array((-math.sin(phase), math.cos(phase), 0.0), dtype=float)
    rot = rotation_x(inc)
    rhat = rot @ local_rhat
    that = rot @ local_that
    rnorm = float(np.linalg.norm(rhat))
    tnorm = float(np.linalg.norm(that))
    return (
        rhat / rnorm if rnorm > 0.0 else np.array((1.0, 0.0, 0.0), dtype=float),
        that / tnorm if tnorm > 0.0 else np.array((0.0, 1.0, 0.0), dtype=float),
    )


DEFAULT_EARTH_RHAT, DEFAULT_EARTH_THAT = default_orbit_unit_vectors(EARTH_IDX)
DEFAULT_JUPITER_RHAT, DEFAULT_JUPITER_THAT = default_orbit_unit_vectors(JUPITER_IDX)
# Hohmann-like prograde excess velocity from Earth toward the outer Solar System.
DEFAULT_VOYAGER_REL_V = 1.9 * DEFAULT_EARTH_THAT
# A simple inward approach from outside Jupiter's orbit.
DEFAULT_SL9_REL_V = -1.2 * DEFAULT_JUPITER_RHAT


DEFAULT_UI_VALUES: dict[str, object] = {
    "language": "English",
    "view": "To Jupiter",
    "total_years": 12.0,
    "dt_days": 5.0,
    "frame_stride": 4,
    "trail_frames": 80,
    "axis_scaling_mode": "Fixed by selected region",
    "log10_c": math.log10(C_REAL_AU_PER_YR),
    "pn_log10": 0.0,
    "size_gamma": 0.25,
    "sun_marker": 7.0,
    "planet_min": 7.0,
    "planet_max": 13.0,
    "sun_mass_log10": 0.0,
    "live_interval_ms": 250,
    "frames_per_refresh": 2,
    "loop_playback": True,
    "use_animation": True,
    "max_animation_frames": 120,
    "include_voyager": False,
    "voyager_mass_log10kg": math.log10(VOYAGER_DRY_MASS_KG),
    "voyager_vx": float(DEFAULT_VOYAGER_REL_V[0]),
    "voyager_vy": float(DEFAULT_VOYAGER_REL_V[1]),
    "voyager_vz": float(DEFAULT_VOYAGER_REL_V[2]),
    "include_sl9": False,
    "sl9_mass_log10kg": math.log10(SL9_DEFAULT_MASS_KG),
    "sl9_vx": float(DEFAULT_SL9_REL_V[0]),
    "sl9_vy": float(DEFAULT_SL9_REL_V[1]),
    "sl9_vz": float(DEFAULT_SL9_REL_V[2]),
}

for _planet_name in PLANET_NAMES:
    DEFAULT_UI_VALUES[f"mass_{_planet_name}"] = 0.0
    DEFAULT_UI_VALUES[f"dist_{_planet_name}"] = 1.0


def ensure_default_session_state() -> None:
    """Initialize missing Streamlit widget keys from the app defaults."""
    for key, value in DEFAULT_UI_VALUES.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("live_frame", 0)
    st.session_state.setdefault("running", False)


def reset_to_initial_values() -> None:
    """Reset controls and playback state while preserving the selected language.

    This function must be called before keyed widgets are instantiated in the
    current Streamlit run. Streamlit raises StreamlitAPIException if the value
    of an already-created widget is modified through st.session_state.

    The language selector is intentionally preserved, so a user working in Czech
    remains in Czech after pressing "Reset to initial values" /
    "Obnovit výchozí hodnoty". All physical, visual, and playback controls are
    reset to their defaults.
    """
    current_language = st.session_state.get("language", DEFAULT_UI_VALUES["language"])
    for key, value in DEFAULT_UI_VALUES.items():
        if key == "language":
            continue
        st.session_state[key] = value
    st.session_state["language"] = current_language
    st.session_state["live_frame"] = 0
    st.session_state["running"] = False
    st.session_state.pop("last_parameter_signature", None)


def request_reset_to_initial_values() -> None:
    """Schedule a reset for the next run.

    The reset button is clicked after some widgets may already exist in the
    current run. Therefore the button only sets a non-widget flag and triggers
    a rerun. At the very beginning of the next run, before widgets are created,
    reset_to_initial_values() safely writes the widget keys.
    """
    st.session_state["_reset_requested"] = True


def apply_pending_reset_if_requested() -> None:
    """Apply a scheduled reset before any keyed widgets are created."""
    if st.session_state.pop("_reset_requested", False):
        reset_to_initial_values()



# =============================================================================
# Export helpers
# =============================================================================

def render_solar_system_gif(
    times: np.ndarray,
    frames_n: np.ndarray,
    frames_p: np.ndarray,
    visible_indices: Sequence[int],
    sizes: Sequence[float],
    axis_scaling_mode: str,
    view: str,
    planet_distance_scale: Sequence[float],
    trail_frames: int,
    gif_frame_count: int,
    gif_fps: int,
    lang: str,
) -> bytes:
    """Render the current Newton/1PN simulation as an animated GIF.

    The GIF export is intentionally separate from the interactive Plotly chart.
    Plotly is used for browser interaction, while Matplotlib/Pillow are used to
    create a downloadable animation file on demand.  This avoids requiring
    browser-side screen recording and avoids an ffmpeg dependency on Streamlit
    Cloud.
    """
    if len(times) == 0:
        raise ValueError("no frames available for GIF export")

    gif_frame_count = int(max(2, min(gif_frame_count, len(times))))
    gif_fps = int(max(1, gif_fps))
    selected = np.linspace(0, len(times) - 1, gif_frame_count).astype(int)
    selected = np.unique(selected)

    title_left = "Newton gravity" if lang == "en" else "Newtonova gravitace"
    title_right = "Einstein GTR 1PN approximation" if lang == "en" else "Einsteinova OTR 1PN aproximace"
    main_title = "Solar-System model" if lang == "en" else "Model Sluneční soustavy"
    time_label = "t" if lang == "en" else "t"

    fig = plt.figure(figsize=(12.5, 6.2), dpi=110)
    ax_n = fig.add_subplot(1, 2, 1, projection="3d")
    ax_p = fig.add_subplot(1, 2, 2, projection="3d")
    axes = (ax_n, ax_p)

    dynamic_axes = axis_scaling_mode == "Dynamic auto-fit during playback"
    initial_axis_min, initial_axis_max = axis_range_for_mode(
        axis_scaling_mode, view, planet_distance_scale, frames_n, frames_p,
        visible_indices, int(selected[0]), trail_frames
    )
    for ax, title in zip(axes, (title_left, title_right)):
        ax.set_title(title)
        ax.set_xlim(initial_axis_min, initial_axis_max)
        ax.set_ylim(initial_axis_min, initial_axis_max)
        ax.set_zlim(initial_axis_min, initial_axis_max)
        ax.set_xlabel("x [AU]")
        ax.set_ylabel("y [AU]")
        ax.set_zlabel("z [AU]")
        ax.view_init(elev=22.0, azim=45.0)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

    line_sets = []
    marker_sets = []
    label_sets = []
    body_labels = [body_display_name(BODIES[idx], lang) for idx in visible_indices]
    for ax in axes:
        model_lines = []
        model_markers = []
        model_labels = []
        for label, size, idx in zip(body_labels, sizes, visible_indices):
            color = BODIES[idx].color
            edge = "black" if idx in (VOYAGER_IDX, SL9_IDX) else color
            marker_size = max(2.0, float(size) * 0.80)
            (line,) = ax.plot([], [], [], color=color, linewidth=1.2, alpha=0.80)
            (marker,) = ax.plot(
                [], [], [], marker="o", linestyle="None", color=color,
                markersize=marker_size, markeredgecolor=edge, markeredgewidth=0.7,
            )
            # Matplotlib's GIF export is separate from the interactive Plotly chart.
            # The Plotly chart already shows body names via markers+text, but GIF
            # frames need explicit 3D text artists.  A tiny screen-space-like
            # offset keeps labels from sitting exactly on top of the markers.
            text = ax.text(0.0, 0.0, 0.0, label, fontsize=7, color="black", ha="left", va="bottom")
            model_lines.append(line)
            model_markers.append(marker)
            model_labels.append(text)
        line_sets.append(model_lines)
        marker_sets.append(model_markers)
        label_sets.append(model_labels)

    time_text = fig.text(0.5, 0.965, "", ha="center", va="top", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    def update(k: int):
        fidx = int(selected[k])
        sl = trail_slice(fidx, trail_frames)
        if dynamic_axes:
            axis_min, axis_max = axis_range_for_mode(
                axis_scaling_mode, view, planet_distance_scale, frames_n, frames_p,
                visible_indices, fidx, trail_frames
            )
            for ax in axes:
                ax.set_xlim(axis_min, axis_max)
                ax.set_ylim(axis_min, axis_max)
                ax.set_zlim(axis_min, axis_max)
        for model_index, frames in enumerate((frames_n, frames_p)):
            for local_i, idx in enumerate(visible_indices):
                xyz = frames[sl, idx, :]
                line_sets[model_index][local_i].set_data_3d(xyz[:, 0], xyz[:, 1], xyz[:, 2])
                pnt = frames[fidx, idx, :]
                marker_sets[model_index][local_i].set_data_3d([pnt[0]], [pnt[1]], [pnt[2]])

                # Update the 3D label position.  The offset is proportional to
                # the current axis half-width so labels remain readable in the
                # fixed, fitted and dynamic view-box modes.
                axis_min_now, axis_max_now = ax_n.get_xlim3d() if model_index == 0 else ax_p.get_xlim3d()
                label_offset = 0.012 * max(abs(axis_min_now), abs(axis_max_now), 1.0)
                label = label_sets[model_index][local_i]
                label.set_position((pnt[0] + label_offset, pnt[1] + label_offset))
                label.set_3d_properties(pnt[2] + label_offset, zdir="z")
        time_text.set_text(f"{main_title}: {time_label} = {times[fidx]:.2f} yr")
        artists = []
        for subset in (line_sets + marker_sets + label_sets):
            artists.extend(subset)
        return artists + [time_text]

    anim = FuncAnimation(fig, update, frames=len(selected), interval=1000.0 / gif_fps, blit=False)
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=True) as tmp:
        writer = PillowWriter(fps=gif_fps)
        anim.save(tmp.name, writer=writer)
        tmp.seek(0)
        data = tmp.read()
    plt.close(fig)
    return data

# =============================================================================
# Streamlit UI
# =============================================================================

st.set_page_config(page_title="Solar System: Newton vs 1PN", layout="wide")
ensure_default_session_state()
apply_pending_reset_if_requested()

st.sidebar.selectbox("Language / Jazyk", LANGUAGE_OPTIONS, key="language")
LANG = lang_code()

st.title(tr(LANG, "title"))
st.caption("Build: v7 reset preserves language")
st.sidebar.caption("Build: v7 reset preserves language")

st.sidebar.header(tr(LANG, "presets"))
if st.sidebar.button(tr(LANG, "reset_initial"), use_container_width=True, on_click=request_reset_to_initial_values):
    st.rerun()


def render_model_description(lang: str) -> None:
    """Render bilingual description of the numerical model."""
    if lang == "cs":
        st.markdown(
            """
Tato aplikace integruje zjednodušený trojrozměrný model Sluneční soustavy v soustavě jednotek

- délka: astronomická jednotka, AU,
- čas: juliánský rok,
- hmotnost: hmotnost Slunce, `M_sun`.

V těchto jednotkách je gravitační konstanta
            """
        )
        st.latex(r"G = 4\pi^2\;{\rm AU^3}\,{\rm M_\odot^{-1}}\,{\rm yr^{-2}}")
        st.markdown("Fyzikální rychlost světla je přibližně")
        st.latex(r"c_{\rm phys}\simeq 63241.077\;{\rm AU\,yr^{-1}}")
        st.markdown(
            """
Numerická data těles vycházejí ze standardních zaokrouhlených hmotností, poloměrů a jednoduchých orbitálních prvků planet. Hmotnosti a poloměry odpovídají tabulce planetárních fyzikálních parametrů NASA/JPL Solar System Dynamics. Orbitální poloměry, sklony a počáteční fáze se zde používají jen ke konstrukci čistých didaktických počátečních podmínek; nejde o efemeridu pro konkrétní datum. JPL výslovně rozlišuje přibližné výpočty z Keplerových elementů od vysoce přesných efemerid Horizons.

### Počáteční podmínky

Každá planeta začíná na zjednodušené kruhové dráze s poloměrem
            """
        )
        st.latex(r"r_i = a_i\,s_i")
        st.markdown("kde `a_i` je referenční hlavní poloosa a `s_i` je uživatelem nastavené škálování vzdálenosti `a/a_real`. Odpovídající tečná kruhová rychlost je")
        st.latex(r"v_i = \sqrt{\frac{G\left(M_\odot^{\ast}+m_i^{\ast}\right)}{r_i}}")
        st.markdown(
            """
kde hvězdička značí hmotnosti po uživatelském škálování. Rovina dráhy se poté nakloní o uvedený sklon. Nakonec se polohy a rychlosti převedou do barycentrické soustavy, takže počáteční těžiště je v klidu.

### Volitelná sonda a kometa

Volitelná sonda podobná Voyageru 1 a volitelná kometa dopadající na Jupiter jsou přidané bodové hmotnosti. Sonda podobná Voyageru začíná poblíž Země, zatímco kometa typu Shoemaker--Levy 9 začíná vně Jupiterovy dráhy a ve výchozím nastavení míří k Jupiteru. Jejich počáteční složky rychlosti nastavuje uživatel v lokálním startovním/encounter rámci,
            """
        )
        st.latex(r"\mathbf v_{\rm Voyager}(0)=\mathbf v_{\rm Earth}(0)+\Delta\mathbf v_{\rm slider},\qquad \mathbf v_{\rm comet}(0)=\mathbf v_{\rm Jupiter}(0)+\Delta\mathbf v_{\rm slider}")
        st.markdown(
            """
Výchozí hmotnost Voyageru vychází z hodnoty suché hmotnosti sondy uváděné NASA. NASA popisuje Voyager 1 jako misi s průletem kolem Jupiteru a Saturnu a uvádí hmotnost sondy 721.9 kg. FAQ NASA k Voyageru také uvádí, že Voyager 1 uniká ze Sluneční soustavy rychlostí přibližně 3.5 AU/rok; zde se tato hodnota používá jen jako orientační řád pro výchozí rychlostní posuvník, nikoli jako přesná historická trajektorie.

Volba komety je označena jako objekt typu Shoemaker--Levy 9 dopadající na Jupiter. Slavný pozorovaný dopad na Jupiter způsobila právě kometa Shoemaker--Levy 9, jejíž fragmenty dopadaly do Jupiteru mezi 16. a 22. červencem 1994. Aplikace nemodeluje vstup do atmosféry, fragmentaci, ablaci ani fyziku dopadu; integruje pouze gravitační bodovou trajektorii před takovým setkáním.

### Newtonovský panel

Levý panel řeší změkčené Newtonovy rovnice N těles
            """
        )
        st.latex(r"\dot{\mathbf r}_i=\mathbf v_i")
        st.latex(r"\dot{\mathbf v}_i=-\sum_{j\ne i}Gm_j\frac{\mathbf r_i-\mathbf r_j}{\left(|\mathbf r_i-\mathbf r_j|^2+\epsilon^2\right)^{3/2}}")
        st.markdown(
            """
Malá hodnota `epsilon = 1e-6 AU` je numerická změkčovací délka. Zabraňuje singularitám zrychlení při nerealisticky blízkých setkáních vytvořených extrémními hodnotami posuvníků. Není to nová fyzikální síla.

### Panel Einsteinovy OTR 1PN aproximace

Pravý panel řeší Newtonovu gravitaci plus párovou první post-newtonovskou korekci. Pro dvojici `i,j` kód používá standardní relativní dvoutělesovou 1PN korekci v harmonických souřadnicích,
            """
        )
    else:
        st.markdown(
            """
This app integrates a simplified three-dimensional Solar-System model in the unit system

- length: astronomical unit, AU,
- time: Julian year,
- mass: solar mass, `M_sun`.

In these units the gravitational constant is
            """
        )
        st.latex(r"G = 4\pi^2\;{\rm AU^3}\,{\rm M_\odot^{-1}}\,{\rm yr^{-2}}")
        st.markdown("The physical speed of light is approximately")
        st.latex(r"c_{\rm phys}\simeq 63241.077\;{\rm AU\,yr^{-1}}")
        st.markdown(
            """
The numerical body data are based on standard rounded planetary masses, radii and simple orbital elements. The masses and radii follow the NASA/JPL Solar System Dynamics table of planetary physical parameters. The orbital radii, inclinations and phases are used only to construct clean didactic initial conditions; the model is not a date-specific ephemeris. JPL explicitly separates such approximate Keplerian-element calculations from high-precision Horizons ephemerides.

### Initial conditions

For each planet the app starts from a simplified circular orbit with radius
            """
        )
        st.latex(r"r_i = a_i\,s_i")
        st.markdown("where `a_i` is the reference semi-major axis and `s_i` is the user-controlled `a/a_real` distance scale. The corresponding tangential circular speed is")
        st.latex(r"v_i = \sqrt{\frac{G\left(M_\odot^{\ast}+m_i^{\ast}\right)}{r_i}}")
        st.markdown(
            """
where starred quantities denote the masses after user scaling. The orbital plane is then tilted by the listed inclination. Finally, positions and velocities are transformed to the barycentric frame, so that the initial center of mass is at rest.

### Optional spacecraft and comet models

The optional Voyager 1-like probe and the optional Jupiter-impact comet are additional point masses. The Voyager-like probe starts near Earth, while the Shoemaker--Levy 9-like comet starts outside Jupiter's orbit and is aimed toward Jupiter by default. Their initial velocity components are user-controlled in the local launch/encounter frame,
            """
        )
        st.latex(r"\mathbf v_{\rm Voyager}(0)=\mathbf v_{\rm Earth}(0)+\Delta\mathbf v_{\rm slider},\qquad \mathbf v_{\rm comet}(0)=\mathbf v_{\rm Jupiter}(0)+\Delta\mathbf v_{\rm slider}")
        st.markdown(
            """
The Voyager default mass follows the NASA mission value for the spacecraft dry mass. NASA lists Voyager 1 as a Jupiter/Saturn flyby mission and gives a spacecraft mass of 721.9 kg. NASA's Voyager FAQ also states that Voyager 1 is escaping the Solar System at about 3.5 AU/yr; this is used only as a convenient order-of-magnitude default for the outward velocity slider, not as a precise historical trajectory.

The comet option is labelled as a Shoemaker--Levy 9-like Jupiter-impact object. The famous observed Jupiter impact was Comet Shoemaker--Levy 9, whose fragments hit Jupiter between 16 and 22 July 1994. The app does not model atmospheric entry, fragmentation, ablation or impact physics; it only integrates the gravitational point-mass trajectory before such an encounter.

### Newtonian panel

The left panel solves the softened Newtonian N-body equations
            """
        )
        st.latex(r"\dot{\mathbf r}_i=\mathbf v_i")
        st.latex(r"\dot{\mathbf v}_i=-\sum_{j\ne i}Gm_j\frac{\mathbf r_i-\mathbf r_j}{\left(|\mathbf r_i-\mathbf r_j|^2+\epsilon^2\right)^{3/2}}")
        st.markdown(
            """
The small value `epsilon = 1e-6 AU` is a numerical softening length. It avoids singular accelerations during unrealistically close encounters created by extreme slider settings. It is not a new physical force.

### Einstein GTR 1PN approximation panel

The right panel solves Newtonian gravity plus a pairwise first post-Newtonian correction. For a pair `i,j` the code uses the standard relative two-body 1PN correction in harmonic-coordinate form,
            """
        )
    st.latex(r"\mathbf a_{ij}^{\rm rel}=\mathbf a_{ij}^{\rm N}+\lambda_{\rm 1PN}\,\mathbf a_{ij}^{\rm 1PN}")
    st.latex(r"\mathbf a_{ij}^{\rm 1PN}=\frac{G M}{c^2 r^2}\left\{\mathbf n\left[(4+2\eta)\frac{GM}{r}-(1+3\eta)v^2+\frac{3}{2}\eta\dot r^2\right]+(4-2\eta)\dot r\,\mathbf v\right\}")
    st.markdown("s" if lang == "cs" else "with")
    st.latex(r"M=m_i+m_j,\qquad \eta=\frac{m_i m_j}{M^2},\qquad \mathbf n=\frac{\mathbf r_i-\mathbf r_j}{r},\qquad \mathbf v=\mathbf v_i-\mathbf v_j,\qquad \dot r=\mathbf n\cdot\mathbf v")
    if lang == "cs":
        st.markdown(
            """
Párová relativní korekce se rozdělí mezi obě tělesa tak, aby zrychlení těžiště dvojice zůstalo nulové,
            """
        )
    else:
        st.markdown(
            """
The pairwise relative correction is split between the two bodies so that the pair center-of-mass acceleration remains zero,
            """
        )
    st.latex(r"\mathbf a_i^{\rm corr}=\frac{m_j}{m_i+m_j}\mathbf a_{ij}^{\rm 1PN},\qquad \mathbf a_j^{\rm corr}=-\frac{m_i}{m_i+m_j}\mathbf a_{ij}^{\rm 1PN}")
    if lang == "cs":
        st.markdown(
            """
Posuvník `1PN multiplier` nastavuje `lambda_1PN`. Fyzikálně přirozená hodnota je `lambda_1PN = 1`. Větší hodnoty záměrně zvětšují relativistický člen, aby byl rozdíl oproti Newtonovu pohybu lépe viditelný.

Nejde o plnou Einstein-Infeld-Hoffmannovu N-tělesovou integraci. Přesné 1PN rovnice více těles obsahují dodatečné tří-tělesové křížové členy. Pravý panel je proto vhodné číst jako vizualizaci párových dvoutělesových 1PN efektů, nikoli jako přesnou relativistickou efemeridu.

### Časová integrace

Oba panely se vyvíjejí klasickou Runge-Kuttovou metodou čtvrtého řádu. Pro soustavu prvního řádu `dy/dt = f(y)` je jeden časový krok
            """
        )
    else:
        st.markdown(
            """
The slider `1PN multiplier` sets `lambda_1PN`. The physically natural value is `lambda_1PN = 1`. Larger values intentionally magnify the relativistic term so that the difference from Newtonian motion is easier to see.

This is not a full Einstein-Infeld-Hoffmann N-body integration. The exact 1PN many-body equations contain additional three-body cross terms. Therefore the right panel should be read as a visualization of pairwise two-body 1PN effects, not as a precision relativistic ephemeris.

### Time integration

Both panels are advanced with the classical fourth-order Runge-Kutta method. For the first-order system `dy/dt = f(y)`, one time step is
            """
        )
    st.latex(r"\mathbf y_{n+1}=\mathbf y_n+\frac{\Delta t}{6}\left(\mathbf k_1+2\mathbf k_2+2\mathbf k_3+\mathbf k_4\right)")
    st.latex(r"\mathbf k_1=f(\mathbf y_n),\quad \mathbf k_2=f(\mathbf y_n+\tfrac{\Delta t}{2}\mathbf k_1),\quad \mathbf k_3=f(\mathbf y_n+\tfrac{\Delta t}{2}\mathbf k_2),\quad \mathbf k_4=f(\mathbf y_n+\Delta t\mathbf k_3)")
    if lang == "cs":
        st.markdown(
            """
RK4 je pohodlná a přesná metoda pro krátké výukové integrace, ale není symplektická. Velmi dlouhé integrace nebo velmi blízká setkání proto nemají být interpretovány jako vysoce přesná dynamika Sluneční soustavy.

### Diagnostika platnosti

Zobrazená diagnostika odhaduje dva malé parametry, které by měly zůstat malé pro 1PN interpretaci:
            """
        )
    else:
        st.markdown(
            """
RK4 is convenient and accurate for short educational integrations, but it is not symplectic. Very long integrations or very close encounters should therefore not be interpreted as high-precision Solar-System dynamics.

### Validity diagnostics

The displayed diagnostics estimate the two small parameters that should remain small for the 1PN interpretation:
            """
        )
    st.latex(r"\max_i\frac{|\mathbf v_i|}{c}\ll 1,\qquad \max_{i<j}\frac{Gm_i}{r_{ij}c^2}\ll 1")
    if lang == "cs":
        st.markdown(
            """
Pokud jsou tato čísla příliš velká, animace může být stále zajímavá, ale už nejde o kvantitativně spolehlivý slabopolní a pomalý relativistický model.

### Jak aplikaci používat

1. V levém panelu zvolte jazyk a základní oblast zobrazení: vnitřní planety, oblast po Jupiter, nebo celou Sluneční soustavu.
2. V části **Režim škálování boxu** určete, zda má být 3D box pevný podle vybrané oblasti, jednorázově přizpůsoben celé spočtené trajektorii, nebo dynamicky měněn během přehrávání. Pevný režim je nejlepší, pokud chcete stabilní zoom a porovnání levého a pravého panelu.
3. Upravte délku simulace, časový krok RK4 a délku zobrazených stop. Menší časový krok je přesnější, ale výpočetně pomalejší.
4. V části **Parametry 1PN** lze změnit efektivní rychlost světla a násobek 1PN korekce. Fyzikálně nejčistší volba je skutečná hodnota `c` a násobek 1. Větší násobek 1PN je pouze vizuální lupa na relativistickou korekci.
5. V částech hmotností a vzdáleností lze škálovat hmotnost Slunce, hmotnosti jednotlivých planet a jejich počáteční vzdálenosti od Slunce.
6. Volitelně lze přidat sondu podobnou Voyageru 1 a kometu typu Shoemaker--Levy 9. Jejich hmotnost a počáteční rychlostní složky se nastavují samostatně.
7. Po změně parametrů stiskněte **Použít a přepočítat**. Aplikace díky tomu nepřepočítává dráhy při každém pohybu sliderem, ale až na vyžádání.
8. Poté použijte **Spustit**, **Pauza** a **Reset** pro živé přehrávání. Alternativně lze použít také tlačítko Play přímo v Plotly grafu.
9. Graf lze ručně otáčet, přibližovat a posouvat. V pevném režimu se ruční zoom během přehrávání nemá přepisovat automatickým škálováním os.
10. V části **Export a stažení** lze vytvořit animovaný GIF aktuálně spočteného průběhu. Větší počet snímků dává hladší video, ale generování je pomalejší.
11. Trajektorie se kvůli efektivitě předpočítají, ale viditelné stopy se kreslí progresivně: v čase \(t\) graf ukazuje jen dráhu proletěnou do tohoto času, nikoli budoucí část orbity.

### Reference

- NASA/JPL Solar System Dynamics, [Planetary Physical Parameters](https://ssd.jpl.nasa.gov/planets/phys_par.html).
- NASA/JPL Solar System Dynamics, [Approximate Positions of the Planets](https://ssd.jpl.nasa.gov/planets/approx_pos.html).
- NASA, [Voyager 1 mission page](https://science.nasa.gov/mission/voyager/voyager-1/) and [Voyager FAQ](https://science.nasa.gov/mission/voyager/frequently-asked-questions/).
- NASA, [Comet Shoemaker--Levy 9](https://science.nasa.gov/solar-system/comets/p-shoemaker-levy-9/), k dopadům na Jupiter v červenci 1994.
- A. Einstein, L. Infeld and B. Hoffmann, *The Gravitational Equations and the Problem of Motion*, Annals of Mathematics **39**, 65--100 (1938), [DOI: 10.2307/1968714](https://doi.org/10.2307/1968714).
- L. Blanchet, *Gravitational Radiation from Post-Newtonian Sources and Inspiralling Compact Binaries*, Living Reviews in Relativity **17**, 2 (2014), [DOI: 10.12942/lrr-2014-2](https://doi.org/10.12942/lrr-2014-2).
- J. C. Butcher, *Numerical methods for ordinary differential equations in the 20th century*, Journal of Computational and Applied Mathematics **125**, 1--29 (2000), [DOI: 10.1016/S0377-0427(00)00455-6](https://doi.org/10.1016/S0377-0427(00)00455-6).
- W. Dehnen, *Towards optimal softening in three-dimensional N-body codes -- I. Minimizing the force error*, MNRAS **324**, 273--291 (2001), [DOI: 10.1046/j.1365-8711.2001.04237.x](https://doi.org/10.1046/j.1365-8711.2001.04237.x).
            """
        )
    else:
        st.markdown(
            """
If these numbers become too large, the animation can still be interesting, but it is no longer a quantitatively reliable weak-field, slow-motion relativistic model.

### How to use the app

1. Use the left sidebar to choose the language and the displayed region: inner planets, out to Jupiter, or all planets.
2. In **View-box scaling mode**, choose whether the 3D box should remain fixed by the selected region, fit the full computed trajectory once, or dynamically auto-fit during playback. The fixed mode is best for stable zooming and for comparing the left and right panels.
3. Set the simulated time, RK4 time step, and trail length. A smaller RK4 time step is more accurate but slower.
4. In **1PN parameters**, tune the effective speed of light and the 1PN multiplier. The physically clean choice is the real value of `c` and multiplier 1. A larger 1PN multiplier is only a visual magnifier for the relativistic correction.
5. In the mass and distance controls, rescale the Sun mass, individual planet masses, and individual initial distances from the Sun.
6. Optionally add a Voyager 1-like probe and a Shoemaker--Levy 9-like comet. Their mass and initial velocity components are controlled separately.
7. After changing parameters, press **Apply and recompute**. This prevents the app from recomputing the trajectories after every slider movement.
8. Then use **Start**, **Pause**, and **Reset** for live playback. Alternatively, use the Plotly Play button inside the chart.
9. The 3D chart can be manually rotated, zoomed, and panned. In fixed view-box mode, the manual zoom should not be overwritten by automatic axis rescaling during playback.
10. In **Export and downloads**, generate an animated GIF of the currently computed simulation. More GIF frames give a smoother video, but rendering takes longer.
11. The trajectories are precomputed for numerical efficiency, but the visible trails are drawn progressively: at time \(t\) the plot shows only the path already travelled up to that time, not the future orbit.

### References

- NASA/JPL Solar System Dynamics, [Planetary Physical Parameters](https://ssd.jpl.nasa.gov/planets/phys_par.html).
- NASA/JPL Solar System Dynamics, [Approximate Positions of the Planets](https://ssd.jpl.nasa.gov/planets/approx_pos.html).
- NASA, [Voyager 1 mission page](https://science.nasa.gov/mission/voyager/voyager-1/) and [Voyager FAQ](https://science.nasa.gov/mission/voyager/frequently-asked-questions/).
- NASA, [Comet Shoemaker--Levy 9](https://science.nasa.gov/solar-system/comets/p-shoemaker-levy-9/), documenting the July 1994 Jupiter impacts.
- A. Einstein, L. Infeld and B. Hoffmann, *The Gravitational Equations and the Problem of Motion*, Annals of Mathematics **39**, 65--100 (1938), [DOI: 10.2307/1968714](https://doi.org/10.2307/1968714).
- L. Blanchet, *Gravitational Radiation from Post-Newtonian Sources and Inspiralling Compact Binaries*, Living Reviews in Relativity **17**, 2 (2014), [DOI: 10.12942/lrr-2014-2](https://doi.org/10.12942/lrr-2014-2).
- J. C. Butcher, *Numerical methods for ordinary differential equations in the 20th century*, Journal of Computational and Applied Mathematics **125**, 1--29 (2000), [DOI: 10.1016/S0377-0427(00)00455-6](https://doi.org/10.1016/S0377-0427(00)00455-6).
- W. Dehnen, *Towards optimal softening in three-dimensional N-body codes -- I. Minimizing the force error*, MNRAS **324**, 273--291 (2001), [DOI: 10.1046/j.1365-8711.2001.04237.x](https://doi.org/10.1046/j.1365-8711.2001.04237.x).
            """
        )


with st.expander(tr(LANG, "what"), expanded=False):
    render_model_description(LANG)

with st.sidebar.form("solar_controls_form"):
    st.caption(tr(LANG, "apply_help"))
    st.header(tr(LANG, "global"))
    view = st.selectbox(tr(LANG, "displayed_region"), VIEW_OPTIONS, key="view", format_func=lambda v: view_label(v, LANG))
    st.markdown("**View-box / 3D axis scaling**" if LANG == "en" else "**Škálování 3D boxu / os**")
    axis_scaling_mode = st.selectbox(
        tr(LANG, "axis_scaling"),
        AXIS_SCALING_OPTIONS,
        key="axis_scaling_mode",
        format_func=lambda m: axis_mode_label(m, LANG),
    )
    st.caption(axis_mode_label(axis_scaling_mode, LANG))
    total_years = st.slider(tr(LANG, "sim_time"), min_value=1.0, max_value=250.0, step=1.0, key="total_years")
    dt_days = st.slider(tr(LANG, "rk4_dt"), min_value=1.0, max_value=30.0, step=1.0, key="dt_days")
    frame_stride = st.slider(tr(LANG, "stride"), min_value=1, max_value=50, step=1, key="frame_stride")
    trail_frames = st.slider(tr(LANG, "trail"), min_value=5, max_value=300, step=5, key="trail_frames")

    st.header(tr(LANG, "pn_params"))
    log10_c = st.slider("log10(c [AU/yr])" if LANG == "en" else "log10(c [AU/rok])", min_value=1.0, max_value=6.0, step=0.05, key="log10_c")
    c_value = 10.0 ** log10_c
    st.caption(tr(LANG, "c_caption", c=c_value, cphys=C_REAL_AU_PER_YR))
    pn_log10 = st.slider("log10(1PN multiplier)" if LANG == "en" else "log10(násobku 1PN)", min_value=-3.0, max_value=6.0, step=0.1, key="pn_log10")
    st.caption(tr(LANG, "pn_caption", val=10.0 ** pn_log10))


    st.header(tr(LANG, "mass_scaling"))
    sun_mass_log10 = st.slider(tr(LANG, "sun_mass"), -3.0, 3.0, step=0.1, key="sun_mass_log10")
    planet_mass_log10 = []
    with st.expander(tr(LANG, "planet_masses"), expanded=False):
        for name in PLANET_NAMES:
            planet_mass_log10.append(st.slider(f"{body_display_name(name, LANG)}: log10(M/M_real)", -3.0, 6.0, step=0.1, key=f"mass_{name}"))

    planet_distance_scale = []
    with st.expander(tr(LANG, "planet_distances"), expanded=False):
        for name in PLANET_NAMES:
            planet_distance_scale.append(st.slider(f"{body_display_name(name, LANG)}: a/a_real", 0.10, 5.00, step=0.05, key=f"dist_{name}"))

    st.header(tr(LANG, "optional"))
    include_voyager = st.checkbox(tr(LANG, "show_voyager"), key="include_voyager")
    with st.expander(tr(LANG, "voyager_title"), expanded=False):
        voyager_mass_log10kg = st.slider(tr(LANG, "voyager_mass"), 0.0, 30.0, step=0.1, key="voyager_mass_log10kg")
        st.caption(tr(LANG, "voyager_caption"))
        voyager_vx = st.slider(tr(LANG, "voyager_vx"), -10.0, 10.0, step=0.1, key="voyager_vx")
        voyager_vy = st.slider(tr(LANG, "voyager_vy"), -10.0, 10.0, step=0.1, key="voyager_vy")
        voyager_vz = st.slider(tr(LANG, "voyager_vz"), -10.0, 10.0, step=0.1, key="voyager_vz")

    include_sl9 = st.checkbox(tr(LANG, "show_sl9"), key="include_sl9")
    with st.expander(tr(LANG, "sl9_title"), expanded=False):
        sl9_mass_log10kg = st.slider(tr(LANG, "comet_mass"), 0.0, 30.0, step=0.1, key="sl9_mass_log10kg")
        st.caption(tr(LANG, "sl9_caption"))
        sl9_vx = st.slider(tr(LANG, "comet_vx"), -10.0, 10.0, step=0.1, key="sl9_vx")
        sl9_vy = st.slider(tr(LANG, "comet_vy"), -10.0, 10.0, step=0.1, key="sl9_vy")
        sl9_vz = st.slider(tr(LANG, "comet_vz"), -10.0, 10.0, step=0.1, key="sl9_vz")

    st.header(tr(LANG, "playback"))
    frame_estimate = int(math.ceil(total_years / (dt_days / DAYS_PER_YEAR) / max(frame_stride, 1))) + 1
    n_step_estimate = int(math.ceil(total_years / (dt_days / DAYS_PER_YEAR)))
    st.caption(tr(LANG, "step_caption", steps=n_step_estimate, frames=frame_estimate))

    live_interval_ms = st.slider(tr(LANG, "live_refresh"), 100, 2000, step=50, key="live_interval_ms")
    frames_per_refresh = st.slider(tr(LANG, "frames_refresh"), 1, 20, step=1, key="frames_per_refresh")
    loop_playback = st.checkbox(tr(LANG, "loop"), key="loop_playback")
    use_animation = st.checkbox(tr(LANG, "plotly_play"), key="use_animation")
    max_animation_frames = st.slider(tr(LANG, "max_plotly"), 20, 250, step=10, key="max_animation_frames")
    apply_submitted = st.form_submit_button(tr(LANG, "apply_recompute"), use_container_width=True)

# Visual-only controls are intentionally outside the Apply/recompute form.
# Changing marker diameters should not trigger a new numerical integration.
st.sidebar.header(tr(LANG, "display_sizes"))
st.sidebar.caption("Visual only: these controls redraw the figure but do not recompute trajectories." if LANG == "en" else "Pouze vzhled: tyto volby překreslí obrázek, ale nepřepočítávají trajektorie.")
size_gamma = st.sidebar.slider(tr(LANG, "gamma"), 0.05, 0.80, step=0.05, key="size_gamma")
sun_marker = st.sidebar.slider(tr(LANG, "sun_marker"), 2.0, 20.0, step=0.5, key="sun_marker")
planet_min = st.sidebar.slider(tr(LANG, "planet_min"), 3.0, 14.0, step=0.5, key="planet_min")
planet_max = st.sidebar.slider(tr(LANG, "planet_max"), 5.0, 25.0, step=0.5, key="planet_max")

if apply_submitted:
    st.session_state.live_frame = 0
    st.session_state.running = False

if n_step_estimate > 20_000:
    st.error(tr(LANG, "too_many_steps"))
    st.stop()

with st.spinner(tr(LANG, "spinner")):
    times, frames_n, frames_p, masses, active_mask, diag_n, diag_p = simulate_cached(
        total_years=float(total_years),
        dt_days=float(dt_days),
        frame_stride=int(frame_stride),
        sun_mass_log10=float(sun_mass_log10),
        planet_mass_log10=tuple(float(x) for x in planet_mass_log10),
        planet_distance_scale=tuple(float(x) for x in planet_distance_scale),
        include_voyager=bool(include_voyager),
        voyager_mass_log10kg=float(voyager_mass_log10kg),
        voyager_vx=float(voyager_vx),
        voyager_vy=float(voyager_vy),
        voyager_vz=float(voyager_vz),
        include_sl9=bool(include_sl9),
        sl9_mass_log10kg=float(sl9_mass_log10kg),
        sl9_vx=float(sl9_vx),
        sl9_vy=float(sl9_vy),
        sl9_vz=float(sl9_vz),
        c_value=float(c_value),
        pn_log10=float(pn_log10),
    )

visible_indices = visible_body_indices(view, bool(include_voyager), bool(include_sl9))

parameter_signature = repr((
    view, total_years, dt_days, frame_stride, trail_frames, axis_scaling_mode,
    log10_c, pn_log10,
    sun_mass_log10, tuple(planet_mass_log10), tuple(planet_distance_scale),
    include_voyager, voyager_mass_log10kg, voyager_vx, voyager_vy, voyager_vz,
    include_sl9, sl9_mass_log10kg, sl9_vx, sl9_vy, sl9_vz,
))
if st.session_state.get("last_parameter_signature") != parameter_signature:
    st.session_state.last_parameter_signature = parameter_signature
    st.session_state.live_frame = 0
    st.session_state.running = False

st.subheader(tr(LANG, "live_playback"))
if "live_frame" not in st.session_state:
    st.session_state.live_frame = 0
if "running" not in st.session_state:
    st.session_state.running = False

st.session_state.live_frame = int(np.clip(st.session_state.live_frame, 0, len(times) - 1))

pb1, pb2, pb3, pb4 = st.columns([1, 1, 1, 3])
with pb1:
    if st.button(tr(LANG, "start"), use_container_width=True):
        st.session_state.running = True
with pb2:
    if st.button(tr(LANG, "pause"), use_container_width=True):
        st.session_state.running = False
with pb3:
    if st.button(tr(LANG, "reset"), use_container_width=True):
        st.session_state.live_frame = 0
        st.session_state.running = False
with pb4:
    status = tr(LANG, "running") if st.session_state.running else tr(LANG, "paused")
    st.write(tr(LANG, "status", status=status, frame=st.session_state.live_frame + 1, total=len(times), time=times[st.session_state.live_frame]))

if st.session_state.running:
    if HAS_AUTOREFRESH:
        st_autorefresh(interval=int(live_interval_ms), key="solar_live_autorefresh")
        next_frame = st.session_state.live_frame + int(frames_per_refresh)
        if next_frame >= len(times):
            if loop_playback:
                next_frame = next_frame % len(times)
            else:
                next_frame = len(times) - 1
                st.session_state.running = False
        st.session_state.live_frame = int(next_frame)
    else:
        st.warning(tr(LANG, "need_autorefresh"))
else:
    st.session_state.live_frame = st.slider(
        tr(LANG, "displayed_frame"),
        0,
        len(times) - 1,
        int(st.session_state.live_frame),
    )

current_frame = int(st.session_state.live_frame)

sizes = marker_sizes(visible_indices, size_gamma, sun_marker, planet_min, planet_max)
fig = make_figure(
    times=times,
    frames_n=frames_n,
    frames_p=frames_p,
    frame_index=current_frame,
    visible_indices=visible_indices,
    trail_frames=trail_frames,
    sizes=sizes,
    animate=use_animation,
    max_animation_frames=max_animation_frames,
    axis_scaling_mode=axis_scaling_mode,
    view=view,
    planet_distance_scale=planet_distance_scale,
    lang=LANG,
)
st.plotly_chart(
    fig,
    use_container_width=True,
    key="solar_system_fixed_axis_plot",
    config={"responsive": True, "scrollZoom": True},
)
st.caption(tr(LANG, "axes_caption", mode=axis_mode_label(axis_scaling_mode, LANG)))
st.caption(tr(LANG, "progressive_caption"))

st.subheader("Export and downloads" if LANG == "en" else "Export a stažení")
export_help = (
    "Generate a downloadable animated GIF from the currently computed simulation. "
    "GIF export is rendered on the server and may take some time; use a modest number of frames first."
    if LANG == "en" else
    "Vygeneruje stažitelný animovaný GIF z aktuálně spočtené simulace. "
    "Export GIFu se renderuje na serveru a může chvíli trvat; nejdříve použijte menší počet snímků."
)
st.caption(export_help)
exp_col1, exp_col2, exp_col3 = st.columns([1, 1, 2])
with exp_col1:
    gif_frames = st.slider(
        "Animated GIF frames" if LANG == "en" else "Počet snímků GIFu",
        min_value=20,
        max_value=180,
        value=80,
        step=10,
    )
with exp_col2:
    gif_fps = st.slider(
        "GIF frame rate [fps]" if LANG == "en" else "Snímková frekvence GIFu [fps]",
        min_value=5,
        max_value=30,
        value=15,
        step=1,
    )
with exp_col3:
    st.write("" if LANG == "en" else "")
    make_gif = st.button(
        "Generate downloadable GIF video" if LANG == "en" else "Vygenerovat stažitelný GIF",
        use_container_width=True,
    )

if make_gif:
    with st.spinner("Rendering GIF..." if LANG == "en" else "Renderuji GIF..."):
        gif_bytes = render_solar_system_gif(
            times=times,
            frames_n=frames_n,
            frames_p=frames_p,
            visible_indices=visible_indices,
            sizes=sizes,
            axis_scaling_mode=axis_scaling_mode,
            view=view,
            planet_distance_scale=planet_distance_scale,
            trail_frames=trail_frames,
            gif_frame_count=int(gif_frames),
            gif_fps=int(gif_fps),
            lang=LANG,
        )
    st.download_button(
        "Download GIF video" if LANG == "en" else "Stáhnout GIF video",
        data=gif_bytes,
        file_name="solar_system_newton_1pn.gif",
        mime="image/gif",
        use_container_width=True,
    )

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(tr(LANG, "displayed_time"), f"{times[current_frame]:.2f} yr")
with col2:
    st.metric(tr(LANG, "sun_mass_scale"), f"{10.0 ** sun_mass_log10:.3g}×")
with col3:
    st.metric(tr(LANG, "onepn_multiplier"), f"{10.0 ** pn_log10:.3g}×")
with col4:
    st.metric("c", f"{c_value:,.0f} AU/yr")

st.subheader(tr(LANG, "diagnostics"))
d1, d2, d3, d4 = st.columns(4)
d1.metric("Newton max v/c", f"{diag_n['max_v_over_c']:.3e}")
d2.metric("Newton max GM/(rc²)", f"{diag_n['max_GM_over_rc2']:.3e}")
d3.metric("1PN max v/c", f"{diag_p['max_v_over_c']:.3e}")
d4.metric("1PN max GM/(rc²)", f"{diag_p['max_GM_over_rc2']:.3e}")

if max(diag_n["max_v_over_c"], diag_p["max_v_over_c"]) > 0.3 or max(diag_n["max_GM_over_rc2"], diag_p["max_GM_over_rc2"]) > 0.1:
    st.warning(tr(LANG, "warn_validity"))

st.subheader(tr(LANG, "current_params"))
rows = []
rows.append({tr(LANG, "body"): body_display_name("Sun", LANG), tr(LANG, "active"): True, tr(LANG, "mass_value"): f"{10.0 ** sun_mass_log10:.3g}×", tr(LANG, "distance_start"): tr(LANG, "center"), tr(LANG, "model_mass"): masses[0]})
for i, body in enumerate(PLANET_BODIES[1:], start=1):
    rows.append(
        {
            tr(LANG, "body"): body_display_name(body, LANG),
            tr(LANG, "active"): True,
            tr(LANG, "mass_value"): f"{10.0 ** planet_mass_log10[i - 1]:.3g}×",
            tr(LANG, "distance_start"): f"{planet_distance_scale[i - 1]:.3g} × a_real",
            tr(LANG, "model_mass"): masses[i],
        }
    )
if include_voyager:
    rows.append(
        {
            tr(LANG, "body"): body_display_name(BODIES[VOYAGER_IDX], LANG),
            tr(LANG, "active"): True,
            tr(LANG, "mass_value"): f"10^{voyager_mass_log10kg:.2f} kg",
            tr(LANG, "distance_start"): tr(LANG, "near_earth"),
            tr(LANG, "model_mass"): masses[VOYAGER_IDX],
        }
    )
if include_sl9:
    rows.append(
        {
            tr(LANG, "body"): body_display_name(BODIES[SL9_IDX], LANG),
            tr(LANG, "active"): True,
            tr(LANG, "mass_value"): f"10^{sl9_mass_log10kg:.2f} kg",
            tr(LANG, "distance_start"): tr(LANG, "outside_jupiter"),
            tr(LANG, "model_mass"): masses[SL9_IDX],
        }
    )
st.dataframe(rows, hide_index=True, use_container_width=True)

st.caption(tr(LANG, "marker_caption"))
