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

import numpy as np
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
    """Newtonian N-body acceleration with a tiny numerical softening."""
    n = len(masses)
    acc = np.zeros_like(pos)
    if active_mask is None:
        active_mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not bool(active_mask[i]):
            continue
        for j in range(n):
            if i == j or not bool(active_mask[j]) or masses[j] <= 0.0:
                continue
            dr = pos[i] - pos[j]
            r2 = float(np.dot(dr, dr)) + SOFTENING_AU * SOFTENING_AU
            inv_r3 = 1.0 / (r2 * math.sqrt(r2))
            acc[i] += -G_MODEL * masses[j] * dr * inv_r3
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

    The optional Voyager/comet bodies are intentionally ignored here.  Otherwise
    a fast escaping probe or an incoming comet would continuously enlarge the
    two 3D boxes and make the planets appear to shrink.  Users can still inspect
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


def trail_slice(frame: int, trail_frames: int) -> slice:
    start = max(0, frame - max(int(trail_frames), 1) + 1)
    return slice(start, frame + 1)


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
    fixed_axis_range: tuple[float, float],
) -> go.Figure:
    """Build static or animated Plotly figure."""
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Newton gravity", "Einstein GTR 1PN approximation"),
        horizontal_spacing=0.02,
    )

    colors = [BODIES[i].color for i in visible_indices]
    names = [BODIES[i].name for i in visible_indices]

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
                    name=f"{model_prefix} {body.name} trail",
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
                marker=dict(size=list(sizes), color=colors, opacity=0.98, sizemode="diameter", line=dict(color="black", width=1)),
                text=names,
                textposition="top center",
                name=f"{model_prefix} bodies",
                hovertemplate="%{text}<br>x=%{x:.3f} AU<br>y=%{y:.3f} AU<br>z=%{z:.3f} AU<extra></extra>",
                showlegend=False,
            ),
            row=1, col=scene_col,
        )

    frame_index = int(np.clip(frame_index, 0, len(times) - 1))
    add_model_traces(frames_n, 1, "Newton", frame_index)
    add_model_traces(frames_p, 2, "1PN", frame_index)

    axis_min, axis_max = fixed_axis_range
    # The axis limits are fixed by the selected planet region, not by the current
    # positions of optional bodies.  This prevents Voyager/comet trajectories
    # from stretching the 3D boxes during playback.  Users can still change the
    # view manually with Plotly zoom/pan/rotate controls.
    axis_common = dict(
        autorange=False,
        range=[axis_min, axis_max],
        showspikes=False,
    )
    axis_template = dict(
        xaxis=dict(title="x [AU]", **axis_common),
        yaxis=dict(title="y [AU]", **axis_common),
        zaxis=dict(title="z [AU]", **axis_common),
        aspectmode="cube",
        uirevision=PLOT_UIREVISION,
    )
    fig.update_layout(
        scene=axis_template,
        scene2=axis_template,
        height=760,
        margin=dict(l=5, r=5, t=70, b=5),
        title="Solar-System model",
        uirevision=PLOT_UIREVISION,
        transition=dict(duration=0),
    )

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
            frame_layout = go.Layout(
                scene=axis_template,
                scene2=axis_template,
                uirevision=PLOT_UIREVISION,
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
                        dict(label="Play", method="animate", args=[None, {"frame": {"duration": 70, "redraw": True}, "transition": {"duration": 0}, "fromcurrent": True}]),
                        dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
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
    "view": "To Jupiter",
    "total_years": 12.0,
    "dt_days": 5.0,
    "frame_stride": 4,
    "trail_frames": 80,
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
    """Reset all visible controls and playback state to their default values."""
    for key, value in DEFAULT_UI_VALUES.items():
        st.session_state[key] = value
    st.session_state["live_frame"] = 0
    st.session_state["running"] = False
    st.session_state.pop("last_parameter_signature", None)


# =============================================================================
# Streamlit UI
# =============================================================================

st.set_page_config(page_title="Solar System: Newton vs 1PN", layout="wide")
st.title("Solar-System motion: Newton gravity vs. Einstein GTR 1PN approximation")

ensure_default_session_state()

st.sidebar.header("Presets")
if st.sidebar.button("Reset to initial values", use_container_width=True):
    reset_to_initial_values()
    st.rerun()

with st.expander("What this app computes", expanded=False):
    st.markdown(
        """
This app integrates a simplified three-dimensional Solar-System model in the
unit system

- length: astronomical unit, AU,
- time: Julian year,
- mass: solar mass, `M_sun`.

In these units the gravitational constant is
        """
    )
    st.latex(r"G = 4\pi^2\;{\rm AU^3}\,{\rm M_\odot^{-1}}\,{\rm yr^{-2}}")
    st.markdown(
        """
The physical speed of light is approximately
        """
    )
    st.latex(r"c_{\rm phys}\simeq 63241.077\;{\rm AU\,yr^{-1}}")
    st.markdown(
        """
The numerical body data are based on standard rounded planetary masses, radii
and simple orbital elements.  The masses and radii follow the NASA/JPL Solar
System Dynamics table of planetary physical parameters.  The orbital radii,
inclinations and phases are used only to construct clean didactic initial
conditions; the model is not a date-specific ephemeris.  JPL explicitly
separates such approximate Keplerian-element calculations from high-precision
Horizons ephemerides.

### Initial conditions

For each planet the app starts from a simplified circular orbit with radius
        """
    )
    st.latex(r"r_i = a_i\,s_i")
    st.markdown(
        """
where `a_i` is the reference semi-major axis and `s_i` is the user-controlled
`a/a_real` distance scale.  The corresponding tangential circular speed is
        """
    )
    st.latex(r"v_i = \sqrt{\frac{G\left(M_\odot^{\ast}+m_i^{\ast}\right)}{r_i}}")
    st.markdown(
        """
where starred quantities denote the masses after user scaling.  The orbital
plane is then tilted by the listed inclination.  Finally, positions and
velocities are transformed to the barycentric frame, so that the initial center
of mass is at rest.

### Optional spacecraft and comet models

The optional Voyager 1-like probe and the optional Jupiter-impact comet are
additional point masses.  The Voyager-like probe starts near Earth, while the
Shoemaker--Levy 9-like comet starts outside Jupiter's orbit and is aimed toward
Jupiter by default.  Their initial velocity components are user-controlled in
the local launch/encounter frame,
        """
    )
    st.latex(r"\mathbf v_{\rm Voyager}(0)=\mathbf v_{\rm Earth}(0)+\Delta\mathbf v_{\rm slider},\qquad \mathbf v_{\rm comet}(0)=\mathbf v_{\rm Jupiter}(0)+\Delta\mathbf v_{\rm slider}")
    st.markdown(
        """
The Voyager default mass follows the NASA mission value for the spacecraft dry
mass.  NASA lists Voyager 1 as a Jupiter/Saturn flyby mission and gives a
spacecraft mass of 721.9 kg.  NASA's Voyager FAQ also states that Voyager 1 is
escaping the Solar System at about 3.5 AU/yr; this is used only as a convenient
order-of-magnitude default for the outward velocity slider, not as a precise
historical trajectory.

The comet option is labelled as a Shoemaker--Levy 9-like Jupiter-impact object.
This is intentional: Comet Halley did not hit Jupiter.  The famous observed
Jupiter impact was Comet Shoemaker--Levy 9, whose fragments hit Jupiter between
16 and 22 July 1994.  The app does not model atmospheric entry, fragmentation,
ablation or impact physics; it only integrates the gravitational point-mass
trajectory before such an encounter.

### Newtonian panel

The left panel solves the softened Newtonian N-body equations
        """
    )
    st.latex(r"\dot{\mathbf r}_i=\mathbf v_i")
    st.latex(r"\dot{\mathbf v}_i=-\sum_{j\ne i}Gm_j\frac{\mathbf r_i-\mathbf r_j}{\left(|\mathbf r_i-\mathbf r_j|^2+\epsilon^2\right)^{3/2}}")
    st.markdown(
        """
The small value `epsilon = 1e-6 AU` is a numerical softening length.  It avoids
singular accelerations during unrealistically close encounters created by
extreme slider settings.  It is not a new physical force.

### Einstein GTR 1PN approximation panel

The right panel solves Newtonian gravity plus a pairwise first
post-Newtonian correction.  For a pair `i,j` the code uses the standard
relative two-body 1PN correction in harmonic-coordinate form,
        """
    )
    st.latex(r"\mathbf a_{ij}^{\rm rel}=\mathbf a_{ij}^{\rm N}+\lambda_{\rm 1PN}\,\mathbf a_{ij}^{\rm 1PN}")
    st.latex(r"\mathbf a_{ij}^{\rm 1PN}=\frac{G M}{c^2 r^2}\left\{\mathbf n\left[(4+2\eta)\frac{GM}{r}-(1+3\eta)v^2+\frac{3}{2}\eta\dot r^2\right]+(4-2\eta)\dot r\,\mathbf v\right\}")
    st.markdown(
        """
with
        """
    )
    st.latex(r"M=m_i+m_j,\qquad \eta=\frac{m_i m_j}{M^2},\qquad \mathbf n=\frac{\mathbf r_i-\mathbf r_j}{r},\qquad \mathbf v=\mathbf v_i-\mathbf v_j,\qquad \dot r=\mathbf n\cdot\mathbf v")
    st.markdown(
        """
The pairwise relative correction is split between the two bodies so that the
pair center-of-mass acceleration remains zero,
        """
    )
    st.latex(r"\mathbf a_i^{\rm corr}=\frac{m_j}{m_i+m_j}\mathbf a_{ij}^{\rm 1PN},\qquad \mathbf a_j^{\rm corr}=-\frac{m_i}{m_i+m_j}\mathbf a_{ij}^{\rm 1PN}")
    st.markdown(
        """
The slider `1PN multiplier` sets `lambda_1PN`.  The physically natural value is
`lambda_1PN = 1`.  Larger values intentionally magnify the relativistic term so
that the difference from Newtonian motion is easier to see.

This is not a full Einstein-Infeld-Hoffmann N-body integration.  The exact 1PN
many-body equations contain additional three-body cross terms.  Therefore the
right panel should be read as a visualization of pairwise two-body 1PN effects,
not as a precision relativistic ephemeris.

### Time integration

Both panels are advanced with the classical fourth-order Runge-Kutta method.
For the first-order system `dy/dt = f(y)`, one time step is
        """
    )
    st.latex(r"\mathbf y_{n+1}=\mathbf y_n+\frac{\Delta t}{6}\left(\mathbf k_1+2\mathbf k_2+2\mathbf k_3+\mathbf k_4\right)")
    st.latex(r"\mathbf k_1=f(\mathbf y_n),\quad \mathbf k_2=f(\mathbf y_n+\tfrac{\Delta t}{2}\mathbf k_1),\quad \mathbf k_3=f(\mathbf y_n+\tfrac{\Delta t}{2}\mathbf k_2),\quad \mathbf k_4=f(\mathbf y_n+\Delta t\mathbf k_3)")
    st.markdown(
        """
RK4 is convenient and accurate for short educational integrations, but it is not
symplectic.  Very long integrations or very close encounters should therefore
not be interpreted as high-precision Solar-System dynamics.

### Validity diagnostics

The displayed diagnostics estimate the two small parameters that should remain
small for the 1PN interpretation:
        """
    )
    st.latex(r"\max_i\frac{|\mathbf v_i|}{c}\ll 1,\qquad \max_{i<j}\frac{Gm_i}{r_{ij}c^2}\ll 1")
    st.markdown(
        """
If these numbers become too large, the animation can still be interesting, but
it is no longer a quantitatively reliable weak-field, slow-motion relativistic
model.

### References

- NASA/JPL Solar System Dynamics, [Planetary Physical Parameters](https://ssd.jpl.nasa.gov/planets/phys_par.html).
- NASA/JPL Solar System Dynamics, [Approximate Positions of the Planets](https://ssd.jpl.nasa.gov/planets/approx_pos.html).
- NASA, [Voyager 1 mission page](https://science.nasa.gov/mission/voyager/voyager-1/) and [Voyager FAQ](https://science.nasa.gov/mission/voyager/frequently-asked-questions/).
- NASA, [Comet Shoemaker--Levy 9](https://science.nasa.gov/solar-system/comets/p-shoemaker-levy-9/), documenting the July 1994 Jupiter impacts.
- NASA, [1P/Halley](https://science.nasa.gov/solar-system/comets/1p-halley/), included to distinguish Halley from the Jupiter-impact comet.
- A. Einstein, L. Infeld and B. Hoffmann, *The Gravitational Equations and the Problem of Motion*, Annals of Mathematics **39**, 65--100 (1938), [DOI: 10.2307/1968714](https://doi.org/10.2307/1968714).
- L. Blanchet, *Gravitational Radiation from Post-Newtonian Sources and Inspiralling Compact Binaries*, Living Reviews in Relativity **17**, 2 (2014), [DOI: 10.12942/lrr-2014-2](https://doi.org/10.12942/lrr-2014-2).
- J. C. Butcher, *Numerical methods for ordinary differential equations in the 20th century*, Journal of Computational and Applied Mathematics **125**, 1--29 (2000), [DOI: 10.1016/S0377-0427(00)00455-6](https://doi.org/10.1016/S0377-0427(00)00455-6).
- W. Dehnen, *Towards optimal softening in three-dimensional N-body codes -- I. Minimizing the force error*, MNRAS **324**, 273--291 (2001), [DOI: 10.1046/j.1365-8711.2001.04237.x](https://doi.org/10.1046/j.1365-8711.2001.04237.x).
        """
    )

st.sidebar.header("Global controls")
view = st.sidebar.selectbox("Displayed region", VIEW_OPTIONS, key="view")
total_years = st.sidebar.slider("Simulated time [yr]", min_value=1.0, max_value=250.0, step=1.0, key="total_years")
dt_days = st.sidebar.slider("RK4 time step [days]", min_value=1.0, max_value=30.0, step=1.0, key="dt_days")
frame_stride = st.sidebar.slider("Integration steps per displayed frame", min_value=1, max_value=50, step=1, key="frame_stride")
trail_frames = st.sidebar.slider("Trail length [displayed frames]", min_value=5, max_value=300, step=5, key="trail_frames")

st.sidebar.header("1PN parameters")
log10_c = st.sidebar.slider("log10(c [AU/yr])", min_value=1.0, max_value=6.0, step=0.05, key="log10_c")
c_value = 10.0 ** log10_c
st.sidebar.caption(f"c = {c_value:,.1f} AU/yr; physical c ≈ {C_REAL_AU_PER_YR:,.1f} AU/yr")
pn_log10 = st.sidebar.slider("log10(1PN multiplier)", min_value=-3.0, max_value=6.0, step=0.1, key="pn_log10")
st.sidebar.caption(f"1PN multiplier = {10.0 ** pn_log10:.3g}")

st.sidebar.header("Display sizes")
size_gamma = st.sidebar.slider("Planet size compression gamma", 0.05, 0.80, step=0.05, key="size_gamma")
sun_marker = st.sidebar.slider("Sun marker diameter [px]", 2.0, 20.0, step=0.5, key="sun_marker")
planet_min = st.sidebar.slider("Minimum planet diameter [px]", 3.0, 14.0, step=0.5, key="planet_min")
planet_max = st.sidebar.slider("Largest planet diameter [px]", 5.0, 25.0, step=0.5, key="planet_max")

st.sidebar.header("Mass scaling")
sun_mass_log10 = st.sidebar.slider("Sun: log10(M/M_real)", -3.0, 3.0, step=0.1, key="sun_mass_log10")
planet_mass_log10 = []
with st.sidebar.expander("Individual planet masses", expanded=False):
    for name in PLANET_NAMES:
        planet_mass_log10.append(st.slider(f"{name}: log10(M/M_real)", -3.0, 6.0, step=0.1, key=f"mass_{name}"))

planet_distance_scale = []
with st.sidebar.expander("Individual planet distances", expanded=False):
    for name in PLANET_NAMES:
        planet_distance_scale.append(st.slider(f"{name}: a/a_real", 0.10, 5.00, step=0.05, key=f"dist_{name}"))

st.sidebar.header("Optional spacecraft / comet")
include_voyager = st.sidebar.checkbox("Show Voyager 1-like probe", key="include_voyager")
with st.sidebar.expander("Voyager 1-like probe", expanded=False):
    voyager_mass_log10kg = st.slider("Voyager mass: log10(m [kg])", 0.0, 30.0, step=0.1, key="voyager_mass_log10kg")
    st.caption("Initial position: near Earth, with a small numerical offset from Earth's center. Velocity components are relative to Earth, in AU/yr.")
    voyager_vx = st.slider("Voyager vx rel. Earth [AU/yr]", -10.0, 10.0, step=0.1, key="voyager_vx")
    voyager_vy = st.slider("Voyager vy rel. Earth [AU/yr]", -10.0, 10.0, step=0.1, key="voyager_vy")
    voyager_vz = st.slider("Voyager vz rel. Earth [AU/yr]", -10.0, 10.0, step=0.1, key="voyager_vz")

include_sl9 = st.sidebar.checkbox("Show Jupiter-impact comet (Shoemaker–Levy 9-like)", key="include_sl9")
with st.sidebar.expander("Jupiter-impact comet / SL9-like body", expanded=False):
    sl9_mass_log10kg = st.slider("Comet mass: log10(m [kg])", 0.0, 30.0, step=0.1, key="sl9_mass_log10kg")
    st.caption("Initial position: outside Jupiter's orbit, aimed toward Jupiter by default. Velocity components are relative to Jupiter, in AU/yr.")
    sl9_vx = st.slider("Comet vx rel. Jupiter [AU/yr]", -10.0, 10.0, step=0.1, key="sl9_vx")
    sl9_vy = st.slider("Comet vy rel. Jupiter [AU/yr]", -10.0, 10.0, step=0.1, key="sl9_vy")
    sl9_vz = st.slider("Comet vz rel. Jupiter [AU/yr]", -10.0, 10.0, step=0.1, key="sl9_vz")

st.sidebar.header("Playback")
frame_estimate = int(math.ceil(total_years / (dt_days / DAYS_PER_YEAR) / max(frame_stride, 1))) + 1
n_step_estimate = int(math.ceil(total_years / (dt_days / DAYS_PER_YEAR)))
st.sidebar.caption(f"Internal RK4 steps: {n_step_estimate:,}; displayed frames: about {frame_estimate:,}")

# Two playback modes are provided:
# 1. Streamlit live playback: Start/Pause buttons advance the displayed frame by
#    rerunning the app on a timer. This is the most visible option for web use.
# 2. Plotly animation: a Play button is embedded directly inside the Plotly chart.
live_interval_ms = st.sidebar.slider("Live playback refresh [ms]", 100, 2000, step=50, key="live_interval_ms")
frames_per_refresh = st.sidebar.slider("Frames advanced per refresh", 1, 20, step=1, key="frames_per_refresh")
loop_playback = st.sidebar.checkbox("Loop live playback", key="loop_playback")
use_animation = st.sidebar.checkbox("Also create Plotly chart Play button", key="use_animation")
max_animation_frames = st.sidebar.slider("Max Plotly animation frames", 20, 250, step=10, key="max_animation_frames")

if n_step_estimate > 20_000:
    st.error(
        "The selected time span and time step would require more than 20,000 RK4 steps. "
        "Increase the time step, shorten the simulated time, or increase steps per displayed frame."
    )
    st.stop()

with st.spinner("Integrating Newton and 1PN trajectories..."):
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

# Reset live playback whenever a physical or display parameter is changed.
parameter_signature = repr((
    view, total_years, dt_days, frame_stride, trail_frames,
    log10_c, pn_log10, size_gamma, sun_marker, planet_min, planet_max,
    sun_mass_log10, tuple(planet_mass_log10), tuple(planet_distance_scale),
    include_voyager, voyager_mass_log10kg, voyager_vx, voyager_vy, voyager_vz,
    include_sl9, sl9_mass_log10kg, sl9_vx, sl9_vy, sl9_vz,
))
if st.session_state.get("last_parameter_signature") != parameter_signature:
    st.session_state.last_parameter_signature = parameter_signature
    st.session_state.live_frame = 0
    st.session_state.running = False

st.subheader("Live playback")
if "live_frame" not in st.session_state:
    st.session_state.live_frame = 0
if "running" not in st.session_state:
    st.session_state.running = False

st.session_state.live_frame = int(np.clip(st.session_state.live_frame, 0, len(times) - 1))

pb1, pb2, pb3, pb4 = st.columns([1, 1, 1, 3])
with pb1:
    if st.button("▶ Start", use_container_width=True):
        st.session_state.running = True
with pb2:
    if st.button("⏸ Pause", use_container_width=True):
        st.session_state.running = False
with pb3:
    if st.button("↺ Reset", use_container_width=True):
        st.session_state.live_frame = 0
        st.session_state.running = False
with pb4:
    st.write(
        f"Status: {'running' if st.session_state.running else 'paused'}; "
        f"frame {st.session_state.live_frame + 1}/{len(times)}; "
        f"t = {times[st.session_state.live_frame]:.2f} yr"
    )

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
        st.warning(
            "Live playback requires the optional package streamlit-autorefresh. "
            "Install it or use the Plotly Play button in the chart."
        )
else:
    st.session_state.live_frame = st.slider(
        "Displayed time frame",
        0,
        len(times) - 1,
        int(st.session_state.live_frame),
    )

current_frame = int(st.session_state.live_frame)

sizes = marker_sizes(visible_indices, size_gamma, sun_marker, planet_min, planet_max)
fixed_axis_range = fixed_axis_range_for_view(view, planet_distance_scale)
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
    fixed_axis_range=fixed_axis_range,
)
st.plotly_chart(
    fig,
    use_container_width=True,
    key="solar_system_fixed_axis_plot",
    config={"responsive": True, "scrollZoom": True},
)
st.caption(
    "The displayed 3D axis ranges are locked by the selected planet region and are not enlarged by Voyager/comet motion. "
    "Use Plotly zoom/pan/rotate controls to change the visible region manually."
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Displayed time", f"{times[current_frame]:.2f} yr")
with col2:
    st.metric("Sun mass scale", f"{10.0 ** sun_mass_log10:.3g}×")
with col3:
    st.metric("1PN multiplier", f"{10.0 ** pn_log10:.3g}×")
with col4:
    st.metric("c", f"{c_value:,.0f} AU/yr")

st.subheader("Approximation diagnostics")
d1, d2, d3, d4 = st.columns(4)
d1.metric("Newton max v/c", f"{diag_n['max_v_over_c']:.3e}")
d2.metric("Newton max GM/(rc²)", f"{diag_n['max_GM_over_rc2']:.3e}")
d3.metric("1PN max v/c", f"{diag_p['max_v_over_c']:.3e}")
d4.metric("1PN max GM/(rc²)", f"{diag_p['max_GM_over_rc2']:.3e}")

if max(diag_n["max_v_over_c"], diag_p["max_v_over_c"]) > 0.3 or max(diag_n["max_GM_over_rc2"], diag_p["max_GM_over_rc2"]) > 0.1:
    st.warning(
        "The chosen parameters push the system outside the comfortable weak-field / slow-motion 1PN regime. "
        "The visualization may still be interesting, but it should not be interpreted as a quantitatively valid relativistic model."
    )

st.subheader("Current body parameters")
rows = []
rows.append({"body": "Sun", "active": True, "mass scale / value": f"{10.0 ** sun_mass_log10:.3g}×", "distance / start": "center", "model mass [M_sun]": masses[0]})
for i, body in enumerate(PLANET_BODIES[1:], start=1):
    rows.append(
        {
            "body": body.name,
            "active": True,
            "mass scale / value": f"{10.0 ** planet_mass_log10[i - 1]:.3g}×",
            "distance / start": f"{planet_distance_scale[i - 1]:.3g} × a_real",
            "model mass [M_sun]": masses[i],
        }
    )
if include_voyager:
    rows.append(
        {
            "body": BODIES[VOYAGER_IDX].name,
            "active": True,
            "mass scale / value": f"10^{voyager_mass_log10kg:.2f} kg",
            "distance / start": "near Earth",
            "model mass [M_sun]": masses[VOYAGER_IDX],
        }
    )
if include_sl9:
    rows.append(
        {
            "body": BODIES[SL9_IDX].name,
            "active": True,
            "mass scale / value": f"10^{sl9_mass_log10kg:.2f} kg",
            "distance / start": "outside Jupiter orbit",
            "model mass [M_sun]": masses[SL9_IDX],
        }
    )
st.dataframe(rows, hide_index=True, use_container_width=True)

st.caption(
    "Marker diameters are visually compressed and are not plotted on the same linear AU scale as the orbital distances. "
    "The compression preserves the ordering of body radii but is chosen so that both the Sun and the planets remain visible."
)
