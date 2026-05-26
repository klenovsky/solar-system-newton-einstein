# Solar-System motion: Newton gravity vs. Einstein GTR 1PN approximation

This Streamlit app visualizes a simplified Solar-System model in two parallel panels:

- **Newton gravity**
- **Einstein GTR 1PN approximation**

The 1PN panel uses a pairwise two-body first post-Newtonian correction. It is a didactic weak-field visualization, not a full Einstein-Infeld-Hoffmann ephemeris and not a JPL Horizons replacement.

## Run locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Put these files in the root of a GitHub repository:

```text
app.py
requirements.txt
README.md
```

Then create a new app in Streamlit Community Cloud and select `app.py` as the main file.

## Smoothness and playback speed

The app separates three controls:

1. **RK4 time step [days]**: numerical integration step.
2. **Stored trajectory stride [RK4 steps]**: how many RK4 steps are skipped between plotted points.
3. **Stored frames advanced per refresh**: apparent browser playback speed.

For smooth zoomed-in orbits use:

```text
RK4 time step [days] = 0.5 to 1.0
Stored trajectory stride = 1
High-quality trails = enabled
```

For faster apparent playback increase:

```text
Stored frames advanced per refresh
```

This changes display speed only; it does not change the numerical model.
