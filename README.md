# Solar-System Newton vs. 1PN Streamlit app / Sluneční soustava Newton vs. 1PN

This Streamlit web application visualizes a simplified Solar-System model in 3D and lets the user switch the interface language between English and Czech. English is the default language.

Tato Streamlit webová aplikace vizualizuje zjednodušený 3D model Sluneční soustavy a umožňuje přepínat rozhraní mezi angličtinou a češtinou. Výchozím jazykem je angličtina.

## Files

```text
app.py
requirements.txt
README.md
```

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model summary

The left panel integrates Newtonian N-body gravity. The right panel integrates Newtonian gravity plus a pairwise two-body first post-Newtonian (1PN) correction inspired by general relativity.

Levý panel integruje Newtonovu gravitaci N těles. Pravý panel integruje Newtonovu gravitaci doplněnou o párovou dvoutělesovou první post-newtonovskou (1PN) korekci inspirovanou obecnou relativitou.

The model is educational. It is not a full Einstein-Infeld-Hoffmann N-body ephemeris and not a JPL Horizons replacement.

Model je výukový. Nejde o plnou Einstein-Infeld-Hoffmannovu efemeridu N těles a nejde o náhradu JPL Horizons.

## Language switch

The language selector is in the sidebar:

```text
Language / Jazyk
```

Choose `English` or `Čeština`. English is the default and `Reset to initial values` returns the app to English.

## GIF export

This version includes an `Export and downloads` section below the main Plotly figure.
It can render the currently computed simulation into a downloadable animated GIF.
The GIF is generated server-side with Matplotlib and Pillow, so rendering can take some time on Streamlit Community Cloud.
Start with a moderate number of frames, for example 60--80.
