# Free Geocoder + DB Mapper (Streamlit)

A simple, **free** geocoding + nearest-DB-location mapping tool.

- Uses **OpenStreetMap Nominatim** for geocoding (fair-use compliant: ~1 req/sec, custom User-Agent).
- Maps each geocoded point to the **nearest** location from your fixed DB list using the **Haversine** distance.
- Built with **Streamlit** for an easy web UI.

---

## Features

- Upload `raw_locations.csv` (with a `raw_location` column).
- Upload `db_locations.csv` (with `db_location, lat, lon`).
- Choose country bias (default: `in` for India), set rate limit, set cutoff distance.
- Get two outputs:
  - `geocoded_raw.csv` — raw + lat/lon
  - `mapped_output.csv` — raw + nearest_db_location + distance_km

---

## CSV Schemas

### raw_locations.csv
| raw_location | ... |
|--------------|-----|
| "Kothari Road, Nagpur" | (any extra columns allowed; preserved) |

**Required column:** `raw_location` (string)

### db_locations.csv
| db_location | lat    | lon    |
|-------------|--------|--------|
| Nagpur      | 21.1498| 79.0826|

**Required columns:** `db_location` (string), `lat` (float), `lon` (float)

> Tip: DB lat/lon aap ek baar Google Maps se nikaal kar fix rakho. Ye list rarely change hoti hai.

---

## Local Setup

1. Python 3.10+ install.
2. In project folder, create these files:
   - `main.py` (Streamlit app code)
   - `requirements.txt`
   - (optional) `.gitignore`
3. Install deps:
   ```bash
   pip install -r requirements.txt
