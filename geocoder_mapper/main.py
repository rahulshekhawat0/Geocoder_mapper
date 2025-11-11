import time
import math
import pandas as pd
import streamlit as st
from io import StringIO
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# --------------------------
# Streamlit UI Configuration
# --------------------------
st.set_page_config(
    page_title="Free Geocoder + DB Mapper",
    page_icon="🗺️",
    layout="wide",
)

st.title("🗺️ Free Location Geocoder + DB Mapper")
st.write(
    "This tool geocodes raw text locations (free via OpenStreetMap Nominatim) "
    "and maps them to the **nearest location in your DB** using Haversine distance."
)

# --------------------------
# Helpers
# --------------------------
def haversine(lat1, lon1, lat2, lon2):
    """Return distance in KM between two lat/lon points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))

@st.cache_data(show_spinner=False)
def _cached_geocode(text: str, country_code: str, ua: str, delay: float):
    """
    Cached single-shot geocode. We instantiate the geocoder inside so the
    cache key depends on (text, country_code, ua, delay).
    Returns (lat, lon) or (None, None).
    """
    if not isinstance(text, str) or not text.strip():
        return (None, None)

    geolocator = Nominatim(user_agent=ua)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=delay, swallow_exceptions=True)

    candidates = [
        text,
        f"{text}, India" if country_code.lower() == "in" else f"{text}",
    ]

    for cand in candidates:
        try:
            loc = geocode(
                cand,
                addressdetails=False,
                country_codes=country_code.lower() if country_code else None,
                exactly_one=True,
            )
            if loc:
                return (loc.latitude, loc.longitude)
        except Exception:
            # swallow per RateLimiter; try next candidate
            pass

    return (None, None)

def nearest_row_to_db(lat, lon, db_df: pd.DataFrame):
    if pd.isna(lat) or pd.isna(lon):
        return (None, None)
    # Compute distances to all DB rows
    dists = db_df.apply(lambda r: haversine(lat, lon, r["lat"], r["lon"]), axis=1)
    idx = dists.idxmin()
    return (db_df.loc[idx, "db_location"], float(dists.min()))

# --------------------------
# Sidebar settings
# --------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    country_bias = st.text_input("Country code bias (2 letters)", value="in")
    user_agent = st.text_input(
        "User-Agent (required by Nominatim)",
        value="rahul-geocoder-mapper/1.0 (educational use)",
        help="Put your app/site name or email so Nominatim can contact if needed.",
    )
    rate_delay = st.number_input(
        "Rate limit delay (seconds)",
        min_value=0.5,
        max_value=10.0,
        value=1.0,
        step=0.5,
        help="Be polite: free Nominatim suggests ~1 request/sec.",
    )
    cutoff_km = st.number_input(
        "Optional cutoff distance (km) to flag unreliable matches",
        min_value=0.0,
        max_value=500.0,
        value=0.0,
        step=1.0,
        help="Set > 0 to mark matches farther than this as 'No reliable match'.",
    )

# --------------------------
# File inputs
# --------------------------
col1, col2 = st.columns(2)
with col1:
    raw_file = st.file_uploader(
        "Upload raw_locations.csv (must have column: raw_location). You can include extra columns too.",
        type=["csv"],
        accept_multiple_files=False,
    )
with col2:
    db_file = st.file_uploader(
        "Upload db_locations.csv (columns: db_location, lat, lon).",
        type=["csv"],
        accept_multiple_files=False,
    )

run = st.button("🚀 Run Geocode + Map")

# --------------------------
# Main flow
# --------------------------
if run:
    if not raw_file or not db_file:
        st.error("Please upload both files first.")
        st.stop()

    try:
        raw_df = pd.read_csv(raw_file)
    except Exception as e:
        st.error(f"Couldn't read raw CSV: {e}")
        st.stop()

    try:
        db_df = pd.read_csv(db_file)
    except Exception as e:
        st.error(f"Couldn't read DB CSV: {e}")
        st.stop()

    # Validate columns
    if "raw_location" not in raw_df.columns:
        st.error("raw_locations.csv must contain a 'raw_location' column.")
        st.stop()

    req_cols = {"db_location", "lat", "lon"}
    if not req_cols.issubset(db_df.columns):
        st.error("db_locations.csv must contain columns: db_location, lat, lon")
        st.stop()

    # Ensure numeric
    try:
        db_df["lat"] = pd.to_numeric(db_df["lat"], errors="coerce")
        db_df["lon"] = pd.to_numeric(db_df["lon"], errors="coerce")
    except Exception:
        st.error("DB lat/lon must be numeric.")
        st.stop()

    st.info("Geocoding raw locations… this uses a free service, so it's intentionally rate-limited.")

    # Geocode with a progress bar
    lats, lons = [], []
    prog = st.progress(0)
    total = len(raw_df)

    for i, txt in enumerate(raw_df["raw_location"].astype(str).tolist(), start=1):
        lat, lon = _cached_geocode(txt, country_bias, user_agent, rate_delay)
        lats.append(lat)
        lons.append(lon)
        prog.progress(min(i / total, 1.0))

    raw_df["lat"] = lats
    raw_df["lon"] = lons

    # Show geocoded preview
    st.subheader("📍 Geocoded Raw (preview)")
    st.dataframe(raw_df.head(20), use_container_width=True)

    # Map to nearest DB location
    st.info("Mapping to nearest DB location…")
    mapped_rows = []
    for _, row in raw_df.iterrows():
        near_name, dist_km = nearest_row_to_db(row["lat"], row["lon"], db_df)
        mapped_rows.append((near_name, dist_km))

    mapped_df = raw_df.copy()
    mapped_df["nearest_db_location"] = [x[0] for x in mapped_rows]
    mapped_df["distance_km"] = [x[1] for x in mapped_rows]

    if cutoff_km and cutoff_km > 0:
        mapped_df["match_status"] = mapped_df["distance_km"].apply(
            lambda d: "No reliable match" if (pd.notna(d) and d > cutoff_km) else "OK"
        )

    st.subheader("🧭 Final Mapping (preview)")
    st.dataframe(mapped_df.head(20), use_container_width=True)

    # Downloads
    geocoded_csv = raw_df.to_csv(index=False).encode("utf-8")
    mapped_csv = mapped_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Geocoded (CSV)",
        data=geocoded_csv,
        file_name="geocoded_raw.csv",
        mime="text/csv",
    )
    st.download_button(
        "⬇️ Download Mapped Output (CSV)",
        data=mapped_csv,
        file_name="mapped_output.csv",
        mime="text/csv",
    )

    st.success("Done! You can review the tables above and download the CSVs.")
else:
    st.caption(
        "Tip: Prepare two CSVs — `raw_locations.csv` with a `raw_location` column, "
        "and `db_locations.csv` with `db_location,lat,lon`. Then click Run."
    )
