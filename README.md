# Solar-System motion: Newton gravity vs. Einstein GTR 1PN approximation

Fast Streamlit + Plotly web version.

This version is optimized for Streamlit Community Cloud:

- trajectories are precomputed once after parameter changes,
- orbit curves are static Plotly traces,
- only body markers are animated inside the browser,
- no `streamlit-autorefresh` loop is used.

This avoids expensive Streamlit reruns during playback.

## Local run

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Upload these files to the root of a GitHub repository:

```text
app.py
requirements.txt
README.md
```

Deploy the repository on Streamlit Community Cloud with `app.py` as the main file.

## Notes

The model is educational.  It is not a date-specific JPL ephemeris and it is not full numerical relativity.  The right panel adds a pairwise two-body 1PN correction to Newtonian gravity.
