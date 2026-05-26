# Solar-System motion: Newton gravity vs. Einstein GTR 1PN approximation

Fast Streamlit + Plotly web application.

This version is optimized for Streamlit Cloud usability:

- all controls are inside a sidebar form;
- dragging sliders does **not** recompute trajectories;
- click **Apply and recompute** to update the simulation;
- Plotly uses a fixed `uirevision` to preserve the 3D camera/zoom as much as possible across reruns;
- browser-side animation moves only the body markers, while orbit curves are static.

Run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The model is a simplified circular-orbit Solar-System demonstration. It is not a date-specific JPL ephemeris and not full numerical relativity. The right panel uses a pairwise two-body 1PN correction.
