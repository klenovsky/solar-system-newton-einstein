# Solar-System Newton vs. 1PN Streamlit app

This is a Streamlit web application that visualizes a simplified Solar-System model in 3D.

The left panel integrates Newtonian N-body gravity. The right panel integrates Newtonian gravity plus a pairwise two-body first post-Newtonian (1PN) correction inspired by general relativity.

The model is educational. It is not a full Einstein-Infeld-Hoffmann N-body ephemeris and not a JPL Horizons replacement.

## Files

```text
app.py
requirements.txt
README.md
```

## Local run

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, and `README.md` to the repository root.
3. Go to Streamlit Community Cloud.
4. Click `Create app`.
5. Select the repository, branch, and entrypoint file `app.py`.
6. Deploy.

## Notes

The app uses units AU, Julian year, and solar mass. In these units

```text
G = 4*pi^2 AU^3 / (M_sun yr^2)
```

The physical speed of light is approximately

```text
c = 63241 AU/yr
```

The plotted body diameters are visually compressed. They preserve the ordering of physical radii, but they are not drawn on the same linear scale as the orbital distances.


## Live playback

This version includes a visible `Start`, `Pause`, and `Reset` panel above the figure.
Live playback is implemented with `streamlit-autorefresh`, which periodically reruns the app and advances the displayed frame.

If the deployed app does not run automatically, check that `streamlit-autorefresh` is present in `requirements.txt` and redeploy/reboot the app.

## Finding the public URL

After deployment on Streamlit Community Cloud, open the app and use the `Share` button in the upper-right corner.
The public app address has the form:

```text
https://your-custom-subdomain.streamlit.app
```

## Changes in this version

- Added a sidebar button **Reset to initial values**. It restores all physical, visual and playback controls to their defaults and resets the live playback frame.
- Expanded the in-app **What this app computes** section with the model equations, unit definitions, 1PN validity conditions, numerical-integration method, model limitations, and literature/data references.

## References shown in the app

- NASA/JPL Solar System Dynamics, Planetary Physical Parameters: https://ssd.jpl.nasa.gov/planets/phys_par.html
- NASA/JPL Solar System Dynamics, Approximate Positions of the Planets: https://ssd.jpl.nasa.gov/planets/approx_pos.html
- A. Einstein, L. Infeld and B. Hoffmann, Annals of Mathematics 39, 65--100 (1938), DOI: 10.2307/1968714
- L. Blanchet, Living Reviews in Relativity 17, 2 (2014), DOI: 10.12942/lrr-2014-2
- J. C. Butcher, Journal of Computational and Applied Mathematics 125, 1--29 (2000), DOI: 10.1016/S0377-0427(00)00455-6
- W. Dehnen, Monthly Notices of the Royal Astronomical Society 324, 273--291 (2001), DOI: 10.1046/j.1365-8711.2001.04237.x
