#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast Streamlit web app: Solar-System Newton gravity vs pairwise 1PN approximation.

Form-based fixed animation version: sidebar sliders are staged and trajectories
are recomputed only when Apply and recompute is pressed. Plotly uses a fixed
uirevision to better preserve camera/zoom across reruns.

Run locally:
    streamlit run app.py

Main design choice in this fast version:
    - Streamlit is used only to choose parameters and build the figure.
    - The animation itself is handled inside Plotly in the browser.
    - The full orbit curves are static traces.
    - Only the moving body markers are animated.

This avoids the slow Streamlit autorefresh/rerun loop and is much smoother on
Streamlit Community Cloud and ordinary browsers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# =============================================================================
# Units and body data
# =============================================================================

# Units: length = AU, time = Julian year, mass = solar mass.
G_MODEL = 4.0 * math.pi * math.pi       # AU^3 / (M_sun yr^2)
C_REAL_AU_PER_YR = 63241.07708426628    # c in AU/year
SOFTENING_AU = 1.0e-7                   # tiny numerical softening
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


# Rounded NASA/JPL-style physical parameters.  The initial conditions are
# simplified circular orbits based on semi-major axes; this is not an ephemeris
# for a particular date.
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

PLANET_NAMES = tuple(b.name for b in BODIES[1:])
PLANET_MAX_RADIUS_KM = max(b.radius_km for b in BODIES[1:])


# =============================================================================
# Mechanics
# =============================================================================

def rotation_x(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)), dtype=float)


def barycentric_transform(pos: np.ndarray, vel: np.ndarray, masses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total_mass = float(np.sum(masses))
    r_cm = np.sum(pos * masses[:, None], axis=0) / total_mass
    v_cm = np.sum(vel * masses[:, None], axis=0) / total_mass
    return pos - r_cm[None, :], vel - v_cm[None, :]


def build_initial_conditions(
    sun_mass_log10: float,
    planet_mass_log10: Sequence[float],
    planet_distance_scale: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create simplified circular-orbit initial conditions."""
    n = len(BODIES)
    masses = np.zeros(n, dtype=float)
    pos = np.zeros((n, 3), dtype=float)
    vel = np.zeros((n, 3), dtype=float)

    masses[0] = max(10.0 ** float(sun_mass_log10), 1.0e-15) * BODIES[0].mass_msun

    for i, body in enumerate(BODIES[1:], start=1):
        mass_factor = 10.0 ** float(planet_mass_log10[i - 1])
        dist_factor = max(float(planet_distance_scale[i - 1]), 1.0e-5)
        masses[i] = max(mass_factor, 0.0) * body.mass_msun

        r = max(body.semi_major_au * dist_factor, 1.0e-6)
        phase = math.radians(body.phase_deg)
        inc = math.radians(body.inclination_deg)

        local_pos = np.array((r * math.cos(phase), r * math.sin(phase), 0.0), dtype=float)
        speed = math.sqrt(G_MODEL * (masses[0] + masses[i]) / r)
        local_vel = np.array((-speed * math.sin(phase), speed * math.cos(phase), 0.0), dtype=float)

        rot = rotation_x(inc)
        pos[i] = rot @ local_pos
        vel[i] = rot @ local_vel

    pos, vel = barycentric_transform(pos, vel, masses)
    return pos, vel, masses


def acceleration_newton(pos: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Newtonian N-body acceleration with small softening."""
    n = len(masses)
    acc = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            dr = pos[i] - pos[j]
            r2 = float(np.dot(dr, dr)) + SOFTENING_AU * SOFTENING_AU
            inv_r3 = 1.0 / (r2 * math.sqrt(r2))
            pair = -G_MODEL * dr * inv_r3
            acc[i] += masses[j] * pair
            acc[j] -= masses[i] * pair
    return acc


def acceleration_pairwise_1pn(
    pos: np.ndarray,
    vel: np.ndarray,
    masses: np.ndarray,
    c_au_per_year: float,
    pn_multiplier: float,
) -> np.ndarray:
    """Newtonian acceleration plus pairwise two-body 1PN correction.

    This is intentionally not a full Einstein-Infeld-Hoffmann N-body model.  It
    adds standard relative two-body 1PN corrections pair by pair, which is
    adequate for a didactic visualization of weak relativistic deviations.
    """
    acc = acceleration_newton(pos, masses)
    if pn_multiplier == 0.0:
        return acc

    c2 = float(c_au_per_year) ** 2
    if c2 <= 0.0:
        return acc

    n = len(masses)
    for i in range(n):
        for j in range(i + 1, n):
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
            a_rel_corr = pn_multiplier * (G_MODEL * mtot / (c2 * r2)) * bracket

            # Keep the pair center-of-mass acceleration zero.
            acc[i] += (mj / mtot) * a_rel_corr
            acc[j] -= (mi / mtot) * a_rel_corr

    return acc


def rhs(state: np.ndarray, masses: np.ndarray, model: str, c_value: float, pn_multiplier: float) -> np.ndarray:
    n = len(masses)
    pos = state[: 3 * n].reshape((n, 3))
    vel = state[3 * n :].reshape((n, 3))
    if model == "newton":
        acc = acceleration_newton(pos, masses)
    elif model == "1pn":
        acc = acceleration_pairwise_1pn(pos, vel, masses, c_value, pn_multiplier)
    else:
        raise ValueError(model)
    return np.concatenate((vel.reshape(-1), acc.reshape(-1)))


def rk4_step(state: np.ndarray, dt: float, masses: np.ndarray, model: str, c_value: float, pn_multiplier: float) -> np.ndarray:
    k1 = rhs(state, masses, model, c_value, pn_multiplier)
    k2 = rhs(state + 0.5 * dt * k1, masses, model, c_value, pn_multiplier)
    k3 = rhs(state + 0.5 * dt * k2, masses, model, c_value, pn_multiplier)
    k4 = rhs(state + dt * k3, masses, model, c_value, pn_multiplier)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@st.cache_data(show_spinner=False)
def simulate_cached(
    total_years: float,
    dt_days: float,
    stored_stride: int,
    sun_mass_log10: float,
    planet_mass_log10: tuple[float, ...],
    planet_distance_scale: tuple[float, ...],
    c_value: float,
    pn_log10: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Integrate both models and return stored trajectory frames."""
    pos0, vel0, masses = build_initial_conditions(sun_mass_log10, planet_mass_log10, planet_distance_scale)
    n = len(masses)
    state_n = np.concatenate((pos0.reshape(-1), vel0.reshape(-1)))
    state_p = state_n.copy()

    dt = float(dt_days) / DAYS_PER_YEAR
    n_steps = int(math.ceil(float(total_years) / dt))
    stored_stride = max(int(stored_stride), 1)
    pn_multiplier = 10.0 ** float(pn_log10)

    times: list[float] = []
    frames_n: list[np.ndarray] = []
    frames_p: list[np.ndarray] = []

    def store(step: int) -> None:
        times.append(step * dt)
        frames_n.append(state_n[: 3 * n].reshape((n, 3)).copy())
        frames_p.append(state_p[: 3 * n].reshape((n, 3)).copy())

    store(0)
    for step in range(1, n_steps + 1):
        state_n = rk4_step(state_n, dt, masses, "newton", c_value, pn_multiplier)
        state_p = rk4_step(state_p, dt, masses, "1pn", c_value, pn_multiplier)
        if step % stored_stride == 0 or step == n_steps:
            store(step)

    vel_n = state_n[3 * n :].reshape((n, 3))
    vel_p = state_p[3 * n :].reshape((n, 3))
    diag = diagnostics(np.asarray(frames_n), np.asarray(frames_p), vel_n, vel_p, masses, c_value)
    return np.asarray(times), np.asarray(frames_n), np.asarray(frames_p), masses, diag


def diagnostics(frames_n: np.ndarray, frames_p: np.ndarray, vel_n: np.ndarray, vel_p: np.ndarray, masses: np.ndarray, c_value: float) -> dict[str, float]:
    max_v_over_c = max(float(np.max(np.linalg.norm(vel_n, axis=1))) / c_value, float(np.max(np.linalg.norm(vel_p, axis=1))) / c_value)
    max_compactness = 0.0
    final_positions = [frames_n[-1], frames_p[-1]]
    for pos in final_positions:
        n = len(masses)
        for i in range(n):
            for j in range(i + 1, n):
                dr = pos[i] - pos[j]
                r = math.sqrt(float(np.dot(dr, dr)) + SOFTENING_AU * SOFTENING_AU)
                max_compactness = max(max_compactness, G_MODEL * masses[i] / (r * c_value * c_value), G_MODEL * masses[j] / (r * c_value * c_value))
    return {"max_v_over_c": max_v_over_c, "max_GM_over_rc2": max_compactness}


# =============================================================================
# Plotting
# =============================================================================

def visible_body_indices(view: str) -> list[int]:
    if view == "Inner planets":
        return list(range(0, 5))
    if view == "To Jupiter":
        return list(range(0, 6))
    return list(range(0, len(BODIES)))


def marker_sizes(indices: Iterable[int], gamma: float, sun_size: float, planet_min: float, planet_max: float) -> list[float]:
    sizes: list[float] = []
    for idx in indices:
        body = BODIES[idx]
        if idx == 0:
            sizes.append(float(sun_size))
        else:
            normalized = max(body.radius_km / PLANET_MAX_RADIUS_KM, 1.0e-12)
            sizes.append(float(planet_min + (planet_max - planet_min) * normalized ** gamma))
    return sizes


def axis_range_for(frames_n: np.ndarray, frames_p: np.ndarray, indices: Sequence[int]) -> tuple[float, float]:
    selected = np.concatenate((frames_n[:, indices, :].reshape((-1, 3)), frames_p[:, indices, :].reshape((-1, 3))), axis=0)
    max_abs = float(np.nanmax(np.abs(selected)))
    if not np.isfinite(max_abs) or max_abs < 0.5:
        max_abs = 1.0
    return -1.10 * max_abs, 1.10 * max_abs


def downsample_indices(n_points: int, max_points: int) -> np.ndarray:
    if n_points <= max_points:
        return np.arange(n_points, dtype=int)
    return np.unique(np.linspace(0, n_points - 1, max_points).astype(int))


def animation_indices(n_points: int, max_frames: int) -> np.ndarray:
    max_frames = max(int(max_frames), 2)
    if n_points <= max_frames:
        return np.arange(n_points, dtype=int)
    return np.unique(np.linspace(0, n_points - 1, max_frames).astype(int))


def make_fast_figure(
    times: np.ndarray,
    frames_n: np.ndarray,
    frames_p: np.ndarray,
    visible_indices: Sequence[int],
    sizes: Sequence[float],
    show_labels: bool,
    orbit_max_points: int,
    max_animation_frames: int,
    frame_duration_ms: int,
    line_width: float,
    marker_opacity: float,
    show_orbit_lines: bool,
) -> go.Figure:
    """Create a Plotly figure with browser-side animation.

    Static traces: full orbit lines.
    Animated traces: only the body marker positions.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Newton gravity", "Einstein GTR 1PN approximation"),
        horizontal_spacing=0.02,
    )

    names = [BODIES[i].name for i in visible_indices]
    colors = [BODIES[i].color for i in visible_indices]
    marker_mode = "markers+text" if show_labels else "markers"
    text_values = names if show_labels else None

    line_idx = downsample_indices(len(times), int(orbit_max_points))

    if show_orbit_lines:
        for col, frames, prefix in ((1, frames_n, "Newton"), (2, frames_p, "1PN")):
            for idx in visible_indices:
                body = BODIES[idx]
                xyz = frames[line_idx, idx, :]
                fig.add_trace(
                    go.Scatter3d(
                        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                        mode="lines",
                        line=dict(width=float(line_width), color=body.color),
                        opacity=0.78,
                        name=f"{prefix} {body.name} orbit",
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1, col=col,
                )

    # Two marker traces are the only traces updated by animation frames.
    pts_n0 = frames_n[0, visible_indices, :]
    pts_p0 = frames_p[0, visible_indices, :]
    marker_common = dict(size=list(sizes), color=colors, opacity=float(marker_opacity), sizemode="diameter")
    hover = "%{text}<br>x=%{x:.4f} AU<br>y=%{y:.4f} AU<br>z=%{z:.4f} AU<extra></extra>"

    fig.add_trace(
        go.Scatter3d(
            x=pts_n0[:, 0], y=pts_n0[:, 1], z=pts_n0[:, 2],
            mode=marker_mode, marker=marker_common, text=text_values, textposition="top center",
            name="Newton bodies", showlegend=False, hovertemplate=hover,
        ),
        row=1, col=1,
    )
    newton_marker_trace_index = len(fig.data) - 1

    fig.add_trace(
        go.Scatter3d(
            x=pts_p0[:, 0], y=pts_p0[:, 1], z=pts_p0[:, 2],
            mode=marker_mode, marker=marker_common, text=text_values, textposition="top center",
            name="1PN bodies", showlegend=False, hovertemplate=hover,
        ),
        row=1, col=2,
    )
    pn_marker_trace_index = len(fig.data) - 1

    anim_idx = animation_indices(len(times), int(max_animation_frames))
    frames: list[go.Frame] = []
    for k, idx in enumerate(anim_idx):
        pts_n = frames_n[idx, visible_indices, :]
        pts_p = frames_p[idx, visible_indices, :]
        # For Scatter3d/WebGL traces, Plotly playback is much more reliable
        # when redraw=True is used in the animation controls below.  We also
        # provide complete marker traces in every frame instead of only x/y/z.
        frames.append(
            go.Frame(
                name=str(k),
                data=[
                    go.Scatter3d(
                        x=pts_n[:, 0], y=pts_n[:, 1], z=pts_n[:, 2],
                        mode=marker_mode, marker=marker_common, text=text_values,
                        textposition="top center", hovertemplate=hover,
                    ),
                    go.Scatter3d(
                        x=pts_p[:, 0], y=pts_p[:, 1], z=pts_p[:, 2],
                        mode=marker_mode, marker=marker_common, text=text_values,
                        textposition="top center", hovertemplate=hover,
                    ),
                ],
                traces=[newton_marker_trace_index, pn_marker_trace_index],
            )
        )
    fig.frames = frames

    axis_min, axis_max = axis_range_for(frames_n, frames_p, visible_indices)
    axis_template = dict(
        xaxis=dict(title="x [AU]", range=[axis_min, axis_max], showspikes=False),
        yaxis=dict(title="y [AU]", range=[axis_min, axis_max], showspikes=False),
        zaxis=dict(title="z [AU]", range=[axis_min, axis_max], showspikes=False),
        aspectmode="cube",
        camera=dict(eye=dict(x=1.35, y=1.35, z=0.85)),
        uirevision="solar-system-camera",
    )

    # Slider steps for the animation.  Keep this client-side.
    slider_steps = []
    for k, idx in enumerate(anim_idx):
        slider_steps.append(
            dict(
                method="animate",
                args=[[str(k)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
                label=f"{times[idx]:.1f}",
            )
        )

    fig.update_layout(
        scene=axis_template,
        scene2=axis_template,
        height=760,
        margin=dict(l=5, r=5, t=105, b=5),
        title=f"Solar-System model: t = 0.00 yr",
        uirevision="solar-system-fast",
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.02,
                y=1.10,
                xanchor="left",
                yanchor="top",
                showactive=False,
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[None, {"frame": {"duration": int(frame_duration_ms), "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}, "mode": "immediate"}],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": "t [yr] = "},
                steps=slider_steps,
                x=0.18,
                y=0.02,
                len=0.76,
                pad={"t": 35, "b": 5},
            )
        ],
    )
    return fig


# =============================================================================
# Streamlit session-state defaults
# =============================================================================

DEFAULT_UI_VALUES: dict[str, object] = {
    "view": "To Jupiter",
    "total_years": 12.0,
    "dt_days": 2.0,
    "stored_stride": 1,
    "show_orbit_lines": True,
    "orbit_max_points": 1800,
    "max_animation_frames": 450,
    "frame_duration_ms": 35,
    "log10_c": math.log10(C_REAL_AU_PER_YR),
    "pn_log10": 0.0,
    "size_gamma": 0.25,
    "sun_marker": 5.0,
    "planet_min": 8.0,
    "planet_max": 15.0,
    "line_width": 2.0,
    "marker_opacity": 0.95,
    "show_labels": True,
    "sun_mass_log10": 0.0,
}
for _name in PLANET_NAMES:
    DEFAULT_UI_VALUES[f"mass_{_name}"] = 0.0
    DEFAULT_UI_VALUES[f"dist_{_name}"] = 1.0


def initialize_default_state() -> None:
    """Fill missing Streamlit widget keys with default values."""
    for key, value in DEFAULT_UI_VALUES.items():
        st.session_state.setdefault(key, value)


def reset_to_initial_values() -> None:
    """Reset all controls to their initial didactic values."""
    for key, value in DEFAULT_UI_VALUES.items():
        st.session_state[key] = value


# =============================================================================
# Streamlit UI
# =============================================================================

# =============================================================================
# Streamlit UI
# =============================================================================

st.set_page_config(page_title="Solar System: Newton vs 1PN", layout="wide")
initialize_default_state()

# In this version, widget values are staged in a form.  The expensive numerical
# integration uses only st.session_state["applied_params"].  Moving sliders in the
# sidebar therefore does not recompute trajectories until the user explicitly
# clicks Apply and recompute.

FORM_KEYS = {key: f"w_{key}" for key in DEFAULT_UI_VALUES}


def initialize_applied_and_widget_state() -> None:
    if "applied_params" not in st.session_state:
        st.session_state["applied_params"] = dict(DEFAULT_UI_VALUES)
    for key, value in DEFAULT_UI_VALUES.items():
        st.session_state.setdefault(FORM_KEYS[key], st.session_state["applied_params"].get(key, value))


def reset_to_initial_values() -> None:
    st.session_state["applied_params"] = dict(DEFAULT_UI_VALUES)
    for key, value in DEFAULT_UI_VALUES.items():
        st.session_state[FORM_KEYS[key]] = value


def collect_form_params() -> dict[str, object]:
    return {key: st.session_state[FORM_KEYS[key]] for key in DEFAULT_UI_VALUES}


initialize_applied_and_widget_state()

st.title("Solar-System motion: Newton gravity vs. Einstein GTR 1PN approximation")

with st.expander("Model description", expanded=False):
    st.markdown(
        """
This app integrates a simplified Solar-System model in astronomical units.  The
left panel solves Newtonian N-body gravity.  The right panel solves Newtonian
gravity plus a pairwise two-body 1PN correction.  The model is designed for
visualization, not as a date-specific JPL ephemeris and not as full numerical
relativity.

Performance architecture of this version:

- parameter widgets are inside a form;
- moving sliders does not immediately recompute the trajectories;
- the trajectories are recomputed only after **Apply and recompute**;
- Plotly receives a fixed `uirevision`, which helps preserve the 3D camera/zoom
  across Streamlit reruns;
- animation frames update only the planet markers, while orbit curves remain
  static.
        """
    )
    st.latex(r"\ddot{\mathbf r}_i=-\sum_{j\ne i}Gm_j\frac{\mathbf r_i-\mathbf r_j}{(|\mathbf r_i-\mathbf r_j|^2+\epsilon^2)^{3/2}}")
    st.latex(r"\mathbf a_i=\mathbf a_i^{\rm Newton}+\lambda_{\rm 1PN}\sum_{j\ne i}\mathbf a_{ij}^{\rm pairwise\;1PN}")

st.sidebar.header("Presets")
st.sidebar.button("Reset to initial values", on_click=reset_to_initial_values, use_container_width=True)
st.sidebar.caption("Reset immediately restores the default values and recomputes using them.")

st.sidebar.header("Controls")
st.sidebar.info("Change parameters, then click **Apply and recompute**. This avoids slow recomputation while dragging sliders.")

with st.sidebar.form("parameter_form", clear_on_submit=False):
    st.subheader("Main controls")
    st.selectbox("Displayed region", ("Inner planets", "To Jupiter", "All planets"), key=FORM_KEYS["view"])
    st.slider("Simulated time [yr]", 0.5, 250.0, key=FORM_KEYS["total_years"], step=0.5)
    st.slider("RK4 time step [days]", 0.25, 20.0, key=FORM_KEYS["dt_days"], step=0.25)
    st.slider("Stored trajectory stride [RK4 steps]", 1, 20, key=FORM_KEYS["stored_stride"], step=1)

    st.subheader("Animation performance")
    st.checkbox("Show smooth orbit curves", key=FORM_KEYS["show_orbit_lines"])
    st.slider("Max points per orbit curve", 200, 5000, key=FORM_KEYS["orbit_max_points"], step=100)
    st.slider("Max browser animation frames", 30, 1200, key=FORM_KEYS["max_animation_frames"], step=30)
    st.slider("Animation frame duration [ms]", 10, 200, key=FORM_KEYS["frame_duration_ms"], step=5)

    st.subheader("1PN parameters")
    st.slider("log10(c [AU/yr])", 1.0, 6.0, key=FORM_KEYS["log10_c"], step=0.05)
    st.slider("log10(1PN multiplier)", -3.0, 6.0, key=FORM_KEYS["pn_log10"], step=0.1)

    st.subheader("Display sizes")
    st.slider("Planet size compression gamma", 0.05, 0.80, key=FORM_KEYS["size_gamma"], step=0.05)
    st.slider("Sun marker diameter [px]", 1.0, 18.0, key=FORM_KEYS["sun_marker"], step=0.5)
    st.slider("Minimum planet diameter [px]", 2.0, 16.0, key=FORM_KEYS["planet_min"], step=0.5)
    st.slider("Largest planet diameter [px]", 4.0, 30.0, key=FORM_KEYS["planet_max"], step=0.5)
    st.slider("Orbit line width [px]", 1.0, 6.0, key=FORM_KEYS["line_width"], step=0.5)
    st.slider("Body marker opacity", 0.30, 1.00, key=FORM_KEYS["marker_opacity"], step=0.05)
    st.checkbox("Show body labels", key=FORM_KEYS["show_labels"])

    st.subheader("Mass scaling")
    st.slider("Sun: log10(M/M_real)", -3.0, 3.0, key=FORM_KEYS["sun_mass_log10"], step=0.1)
    with st.expander("Individual planet masses", expanded=False):
        for name in PLANET_NAMES:
            st.slider(f"{name}: log10(M/M_real)", -3.0, 6.0, key=FORM_KEYS[f"mass_{name}"], step=0.1)

    st.subheader("Distance scaling")
    with st.expander("Individual planet distances", expanded=False):
        for name in PLANET_NAMES:
            st.slider(f"{name}: a/a_real", 0.10, 5.00, key=FORM_KEYS[f"dist_{name}"], step=0.05)

    apply_clicked = st.form_submit_button("Apply and recompute", use_container_width=True, type="primary")

if apply_clicked:
    st.session_state["applied_params"] = collect_form_params()

p = st.session_state["applied_params"]
view = str(p["view"])
total_years = float(p["total_years"])
dt_days = float(p["dt_days"])
stored_stride = int(p["stored_stride"])
show_orbit_lines = bool(p["show_orbit_lines"])
orbit_max_points = int(p["orbit_max_points"])
max_animation_frames = int(p["max_animation_frames"])
frame_duration_ms = int(p["frame_duration_ms"])
log10_c = float(p["log10_c"])
c_value = 10.0 ** log10_c
pn_log10 = float(p["pn_log10"])
size_gamma = float(p["size_gamma"])
sun_marker = float(p["sun_marker"])
planet_min = float(p["planet_min"])
planet_max = float(p["planet_max"])
line_width = float(p["line_width"])
marker_opacity = float(p["marker_opacity"])
show_labels = bool(p["show_labels"])
sun_mass_log10 = float(p["sun_mass_log10"])
planet_mass_log10 = [float(p[f"mass_{name}"]) for name in PLANET_NAMES]
planet_distance_scale = [float(p[f"dist_{name}"]) for name in PLANET_NAMES]

st.sidebar.header("Applied values")
st.sidebar.caption(f"Region: {view}")
st.sidebar.caption(f"c = {c_value:,.1f} AU/yr; physical c ≈ {C_REAL_AU_PER_YR:,.1f} AU/yr")
st.sidebar.caption(f"1PN multiplier = {10.0 ** pn_log10:.3g}")

# Safety limits to avoid generating unreasonably large figures on Streamlit Cloud.
n_steps = int(math.ceil(total_years / (dt_days / DAYS_PER_YEAR)))
stored_frames = int(math.ceil(n_steps / max(stored_stride, 1))) + 1
st.sidebar.caption(f"RK4 steps after Apply: {n_steps:,}; stored frames: about {stored_frames:,}.")

if n_steps > 65_000:
    st.error("Too many RK4 steps. Increase RK4 time step or shorten simulated time.")
    st.stop()
if stored_frames > 25_000:
    st.error("Too many stored frames. Increase stored trajectory stride or shorten simulated time.")
    st.stop()

with st.spinner("Computing trajectories after the last applied parameter set..."):
    times, frames_n, frames_p, masses, diag = simulate_cached(
        total_years=float(total_years),
        dt_days=float(dt_days),
        stored_stride=int(stored_stride),
        sun_mass_log10=float(sun_mass_log10),
        planet_mass_log10=tuple(float(x) for x in planet_mass_log10),
        planet_distance_scale=tuple(float(x) for x in planet_distance_scale),
        c_value=float(c_value),
        pn_log10=float(pn_log10),
    )

visible_indices = visible_body_indices(view)
sizes = marker_sizes(visible_indices, size_gamma, sun_marker, planet_min, planet_max)

fig = make_fast_figure(
    times=times,
    frames_n=frames_n,
    frames_p=frames_p,
    visible_indices=visible_indices,
    sizes=sizes,
    show_labels=show_labels,
    orbit_max_points=int(orbit_max_points),
    max_animation_frames=int(max_animation_frames),
    frame_duration_ms=int(frame_duration_ms),
    line_width=float(line_width),
    marker_opacity=float(marker_opacity),
    show_orbit_lines=bool(show_orbit_lines),
)

st.plotly_chart(
    fig,
    use_container_width=True,
    key="solar_system_plotly_chart",
    config={"scrollZoom": True, "displaylogo": False, "responsive": True},
)

st.info(
    "Change sliders in the sidebar and then click **Apply and recompute**.  The 3D camera/zoom is preserved as much as Plotly allows by using a fixed `uirevision`; avoiding repeated Streamlit reruns while dragging sliders also prevents many unwanted view resets."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Stored frames", f"{len(times):,}")
col2.metric("Browser animation frames", f"{min(len(times), max_animation_frames):,}")
col3.metric("1PN multiplier", f"{10.0 ** pn_log10:.3g}×")
col4.metric("c", f"{c_value:,.0f} AU/yr")

st.subheader("Approximation diagnostics")
d1, d2 = st.columns(2)
d1.metric("max v/c", f"{diag['max_v_over_c']:.3e}")
d2.metric("max GM/(rc²)", f"{diag['max_GM_over_rc2']:.3e}")
if diag["max_v_over_c"] > 0.3 or diag["max_GM_over_rc2"] > 0.1:
    st.warning("The selected parameters are outside the comfortable weak-field / slow-motion 1PN regime.")

st.subheader("Current applied body parameters")
rows = [{"body": "Sun", "mass scale": 10.0 ** sun_mass_log10, "distance scale": 0.0, "model mass [M_sun]": masses[0]}]
for i, body in enumerate(BODIES[1:], start=1):
    rows.append({"body": body.name, "mass scale": 10.0 ** planet_mass_log10[i - 1], "distance scale": planet_distance_scale[i - 1], "model mass [M_sun]": masses[i]})
st.dataframe(rows, hide_index=True, use_container_width=True)

st.caption(
    "Performance tips: for public demos use 'Inner planets' or 'To Jupiter', keep Max browser animation frames around 200–450, and increase Max points per orbit curve only if the static orbit curves look polygonal after zooming. "
    "The animation moves only body markers; the orbit curves are static precomputed trajectories. Marker diameters are visually compressed and are not plotted on the same AU scale as orbital distances."
)
