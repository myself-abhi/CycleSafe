# Austin CycleSafe

A go / no-go decision tool for Austin cyclists, built on the City of Austin's
bicycle crash record (2010–2017, 2,463 incidents).

The user is an Austin cyclist about to leave the house. The decision the app
supports is binary — ride now, or wait, change the route, or skip it. The
indicator is the rate at which historical crashes turned serious (Killed or
Incapacitating Injury — "KSI"), filtered to match the rider's conditions and
compared against the 11.0% citywide baseline.

## Live demo

Deploy on [Streamlit Community Cloud](https://share.streamlit.io) — point it at
this repo, leave defaults, and it picks up `app.py` and `requirements.txt`
automatically.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open <http://localhost:8501>.

## How risk is determined

The hero metric is the KSI rate. Across the full record it sits at 11.0%. When
a rider applies filters — time of day, day of week, surface (a proxy for
weather), roadway type, intersection — the app recomputes that rate on the
matching subset and compares it. Above baseline → DANGER. Below → SAFE.

If a filter combination matches fewer than 30 historical crashes, the app
refuses to render a verdict and says so. Three crashes is not a signal.

## Design

Three colors only — Charcoal `#1E1E1E`, Safety Teal `#2ECC71`, Pure White.
Four panels in a 2×2 grid on laptop, single column on mobile. The verdict is
the largest element on the page; in DANGER mode the entire palette inverts and
the teal accent disappears, so the absence of the safety color becomes the
warning signal.

## Data

`bike_crash.csv` — City of Austin Open Data, cyclist-involved crashes
2010–2017. No GPS coordinates in this release, which is why the app indexes on
*when* and *what kind of road* rather than *where*.

## Built for

Northeastern University, ALY6040 Practical Data Mining, Module 5: Building
Products.
