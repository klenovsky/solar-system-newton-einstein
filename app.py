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


# =============================================================================
# Physical constants and default body data
# =============================================================================

G_MODEL = 4.0 * math.pi * math.pi       # AU^3 / (M_sun yr^2)
C_REAL_AU_PER_YR = 63241.07708426628    # physical speed of light in AU/year
SOFTENING_AU = 1.0e-6                   # purely numerical softening
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
BODIES: tuple[BodyData, ...] = (
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

PLANET_NAMES = tuple(body.name for body in BODIES[1:])
BODY_NAMES = tuple(body.name for body in BODIES)
PLANET_MAX_RADIUS_KM = max(body.radius_km for body in BODIES[1:])


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create simplified initial conditions for Sun + eight planets.

    Planet positions start on tilted circular orbits with radii equal to
    semi_major_axis * distance_scale.  Velocities are circular Keplerian speeds
    for the scaled Sun mass and scaled planet mass.  This gives a clean didactic
    model, not a high-precision ephemeris for a specific date.
    """
    n = len(BODIES)
    masses = np.zeros(n, dtype=float)
    pos = np.zeros((n, 3), dtype=float)
    vel = np.zeros((n, 3), dtype=float)

    sun_factor = 10.0 ** float(sun_mass_log10)
    masses[0] = max(sun_factor, 1.0e-15) * BODIES[0].mass_msun

    for i, body in enumerate(BODIES[1:], start=1):
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

    pos, vel = barycentric_transform(pos, vel, masses)
    return pos, vel, masses


def acceleration_newton(pos: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Newtonian N-body acceleration with a tiny numerical softening."""
    n = len(masses)
    acc = np.zeros_like(pos)
    for i in range(n):
        for j in range(n):
            if i == j:
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
) -> np.ndarray:
    """Newtonian N-body acceleration plus pairwise two-body 1PN corrections.

    For each pair i,j the standard relative two-body 1PN correction is computed
    in harmonic-coordinate form and split between the two bodies so that the
    pair center-of-mass acceleration remains zero.

    This is useful pedagogically, especially for a Sun-dominated system, but it
    is not the full Einstein-Infeld-Hoffmann N-body equation because the genuine
    1PN three-body terms are not included.
    """
    acc = acceleration_newton(pos, masses)
    if pn_multiplier == 0.0:
        return acc

    c2 = float(c_au_per_year) ** 2
    if c2 <= 0.0:
        return acc

    n_bodies = len(masses)
    for i in range(n_bodies):
        for j in range(i + 1, n_bodies):
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


def rhs(state: np.ndarray, masses: np.ndarray, model: str, c_value: float, pn_multiplier: float) -> np.ndarray:
    """Right-hand side of the first-order ODE system."""
    n = len(masses)
    pos = state[: 3 * n].reshape((n, 3))
    vel = state[3 * n :].reshape((n, 3))
    if model == "newton":
        acc = acceleration_newton(pos, masses)
    elif model == "1pn":
        acc = acceleration_pairwise_1pn(pos, vel, masses, c_value, pn_multiplier)
    else:
        raise ValueError(f"unknown model: {model}")
    return np.concatenate((vel.reshape(-1), acc.reshape(-1)))


def rk4_step(state: np.ndarray, dt: float, masses: np.ndarray, model: str, c_value: float, pn_multiplier: float) -> np.ndarray:
    """One fourth-order Runge-Kutta time step."""
    k1 = rhs(state, masses, model, c_value, pn_multiplier)
    k2 = rhs(state + 0.5 * dt * k1, masses, model, c_value, pn_multiplier)
    k3 = rhs(state + 0.5 * dt * k2, masses, model, c_value, pn_multiplier)
    k4 = rhs(state + dt * k3, masses, model, c_value, pn_multiplier)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def diagnostics(pos: np.ndarray, vel: np.ndarray, masses: np.ndarray, c_value: float) -> dict[str, float]:
    """Return simple 1PN validity diagnostics."""
    speeds = np.linalg.norm(vel, axis=1)
    max_v_over_c = float(np.max(speeds) / max(c_value, 1.0e-30))

    max_compactness = 0.0
    n = len(masses)
    for i in range(n):
        for j in range(i + 1, n):
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
    c_value: float,
    pn_log10: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float], dict[str, float]]:
    """Integrate Newton and 1PN models and return downsampled frames."""
    pos0, vel0, masses = build_initial_conditions(sun_mass_log10, planet_mass_log10, planet_distance_scale)
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
        state_n = rk4_step(state_n, dt, masses, "newton", c_value, pn_multiplier)
        state_p = rk4_step(state_p, dt, masses, "1pn", c_value, pn_multiplier)
        if step % frame_stride == 0 or step == n_steps:
            store(step)

    diag_n = diagnostics(frames_n[-1], state_n[3 * n :].reshape((n, 3)), masses, c_value)
    diag_p = diagnostics(frames_p[-1], state_p[3 * n :].reshape((n, 3)), masses, c_value)
    return (
        np.asarray(times),
        np.asarray(frames_n),
        np.asarray(frames_p),
        masses,
        diag_n,
        diag_p,
    )


# =============================================================================
# Plot helpers
# =============================================================================

def visible_body_indices(view: str) -> list[int]:
    if view == "Inner planets":
        return list(range(0, 5))       # Sun through Mars
    if view == "To Jupiter":
        return list(range(0, 6))       # Sun through Jupiter
    return list(range(0, len(BODIES)))


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


def axis_range_for(frames_a: np.ndarray, frames_b: np.ndarray, indices: Sequence[int]) -> tuple[float, float]:
    selected = np.concatenate((frames_a[:, indices, :].reshape((-1, 3)), frames_b[:, indices, :].reshape((-1, 3))), axis=0)
    max_abs = float(np.max(np.abs(selected)))
    if not np.isfinite(max_abs) or max_abs < 0.5:
        max_abs = 1.0
    margin = 1.12 * max_abs
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
                marker=dict(size=list(sizes), color=colors, opacity=0.95, sizemode="diameter"),
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

    axis_min, axis_max = axis_range_for(frames_n, frames_p, visible_indices)
    axis_template = dict(
        xaxis=dict(title="x [AU]", range=[axis_min, axis_max]),
        yaxis=dict(title="y [AU]", range=[axis_min, axis_max]),
        zaxis=dict(title="z [AU]", range=[axis_min, axis_max]),
        aspectmode="cube",
    )
    fig.update_layout(
        scene=axis_template,
        scene2=axis_template,
        height=760,
        margin=dict(l=5, r=5, t=70, b=5),
        title=f"Solar-System model: t = {times[frame_index]:.2f} yr",
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
            animation_frames.append(go.Frame(data=frame_data, traces=list(range(total_trace_count)), name=str(fidx)))

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
                        dict(label="Play", method="animate", args=[None, {"frame": {"duration": 70, "redraw": True}, "fromcurrent": True}]),
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
                            args=[[str(fidx)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        )
                        for fidx in selected_animation_frames
                    ],
                )
            ],
        )

    return fig


# =============================================================================
# Streamlit UI
# =============================================================================

st.set_page_config(page_title="Solar System: Newton vs 1PN", layout="wide")
st.title("Solar-System motion: Newton gravity vs. Einstein GTR 1PN approximation")

with st.expander("What this app computes", expanded=False):
    st.markdown(
        """
The app integrates a simplified Solar-System model in astronomical units.  The
left panel solves Newtonian N-body gravity.  The right panel solves Newtonian
gravity plus a pairwise two-body 1PN correction.  The 1PN correction is a
weak-field, slow-motion approximation inspired by general relativity; it is not
a full Einstein-Infeld-Hoffmann many-body ephemeris and not a JPL Horizons
replacement.
        """
    )
    st.latex(r"\ddot{\mathbf r}_i=-\sum_{j\ne i}Gm_j\frac{\mathbf r_i-\mathbf r_j}{\left(|\mathbf r_i-\mathbf r_j|^2+\epsilon^2\right)^{3/2}}")
    st.latex(r"\mathbf a_i=\mathbf a_i^{\rm Newton}+\lambda_{\rm 1PN}\sum_{j\ne i}\mathbf a_{ij}^{\rm pairwise\;1PN}")

st.sidebar.header("Global controls")
view = st.sidebar.selectbox("Displayed region", ("Inner planets", "To Jupiter", "All planets"), index=1)
total_years = st.sidebar.slider("Simulated time [yr]", min_value=1.0, max_value=250.0, value=12.0, step=1.0)
dt_days = st.sidebar.slider("RK4 time step [days]", min_value=1.0, max_value=30.0, value=5.0, step=1.0)
frame_stride = st.sidebar.slider("Integration steps per displayed frame", min_value=1, max_value=50, value=4, step=1)
trail_frames = st.sidebar.slider("Trail length [displayed frames]", min_value=5, max_value=300, value=80, step=5)

st.sidebar.header("1PN parameters")
log10_c = st.sidebar.slider("log10(c [AU/yr])", min_value=1.0, max_value=6.0, value=math.log10(C_REAL_AU_PER_YR), step=0.05)
c_value = 10.0 ** log10_c
st.sidebar.caption(f"c = {c_value:,.1f} AU/yr; physical c ≈ {C_REAL_AU_PER_YR:,.1f} AU/yr")
pn_log10 = st.sidebar.slider("log10(1PN multiplier)", min_value=-3.0, max_value=6.0, value=0.0, step=0.1)
st.sidebar.caption(f"1PN multiplier = {10.0 ** pn_log10:.3g}")

st.sidebar.header("Display sizes")
size_gamma = st.sidebar.slider("Planet size compression gamma", 0.05, 0.80, 0.25, 0.05)
sun_marker = st.sidebar.slider("Sun marker diameter [px]", 2.0, 20.0, 7.0, 0.5)
planet_min = st.sidebar.slider("Minimum planet diameter [px]", 3.0, 14.0, 7.0, 0.5)
planet_max = st.sidebar.slider("Largest planet diameter [px]", 5.0, 25.0, 13.0, 0.5)

st.sidebar.header("Mass scaling")
sun_mass_log10 = st.sidebar.slider("Sun: log10(M/M_real)", -3.0, 3.0, 0.0, 0.1)
planet_mass_log10 = []
with st.sidebar.expander("Individual planet masses", expanded=False):
    for name in PLANET_NAMES:
        planet_mass_log10.append(st.slider(f"{name}: log10(M/M_real)", -3.0, 6.0, 0.0, 0.1, key=f"mass_{name}"))

planet_distance_scale = []
with st.sidebar.expander("Individual planet distances", expanded=False):
    for name in PLANET_NAMES:
        planet_distance_scale.append(st.slider(f"{name}: a/a_real", 0.10, 5.00, 1.00, 0.05, key=f"dist_{name}"))

st.sidebar.header("Animation")
frame_estimate = int(math.ceil(total_years / (dt_days / DAYS_PER_YEAR) / max(frame_stride, 1))) + 1
n_step_estimate = int(math.ceil(total_years / (dt_days / DAYS_PER_YEAR)))
st.sidebar.caption(f"Internal RK4 steps: {n_step_estimate:,}; displayed frames: about {frame_estimate:,}")
use_animation = st.sidebar.checkbox("Create Plotly Play animation", value=False)
max_animation_frames = st.sidebar.slider("Max animation frames", 20, 250, 120, 10)

if n_step_estimate > 20_000:
    st.error(
        "The selected time span and time step would require more than 20,000 RK4 steps. "
        "Increase the time step, shorten the simulated time, or increase steps per displayed frame."
    )
    st.stop()

with st.spinner("Integrating Newton and 1PN trajectories..."):
    times, frames_n, frames_p, masses, diag_n, diag_p = simulate_cached(
        total_years=float(total_years),
        dt_days=float(dt_days),
        frame_stride=int(frame_stride),
        sun_mass_log10=float(sun_mass_log10),
        planet_mass_log10=tuple(float(x) for x in planet_mass_log10),
        planet_distance_scale=tuple(float(x) for x in planet_distance_scale),
        c_value=float(c_value),
        pn_log10=float(pn_log10),
    )

visible_indices = visible_body_indices(view)
current_frame = st.slider("Displayed time frame", 0, len(times) - 1, min(len(times) - 1, len(times) // 2))

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
)
st.plotly_chart(fig, use_container_width=True)

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
rows.append({"body": "Sun", "mass scale": 10.0 ** sun_mass_log10, "distance scale": 0.0, "model mass [M_sun]": masses[0]})
for i, body in enumerate(BODIES[1:], start=1):
    rows.append(
        {
            "body": body.name,
            "mass scale": 10.0 ** planet_mass_log10[i - 1],
            "distance scale": planet_distance_scale[i - 1],
            "model mass [M_sun]": masses[i],
        }
    )
st.dataframe(rows, hide_index=True, use_container_width=True)

st.caption(
    "Marker diameters are visually compressed and are not plotted on the same linear AU scale as the orbital distances. "
    "The compression preserves the ordering of body radii but is chosen so that both the Sun and the planets remain visible."
)
