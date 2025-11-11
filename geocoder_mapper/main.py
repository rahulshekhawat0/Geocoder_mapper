# main.py

import math
import io
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
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
st.write("""
This tool geocodes raw text locations (free using OpenStreetMap Nominatim)  
and maps them to the **nearest location in your DB** using Haversine distance.

**Special handling when not in DB**:
- If it’s an **Indian State/UT**, we keep the state/UT name and tag it as *Indian State/UT*.
- If it’s a **foreign country**, we keep the country name and tag it as *Foreign Country*.
""")

# --------------------------
# Constants
# --------------------------
INDIA_COUNTRY_CODE = "in"

INDIAN_STATES_UTS = {
    # States
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
    "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka",
    "kerala", "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram",
    "nagaland", "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu",
    "telangana", "tripura", "uttar pradesh", "uttarakhand", "west bengal",
    # UTs
    "andaman and nicobar islands", "chandigarh", "dadra and nagar haveli and daman and diu",
    "daman and diu", "dadra and nagar haveli",  # old names for safety
    "delhi", "national capital territory of delhi", "jammu and kashmir",
    "ladakh", "lakshadweep", "puducherry", "pondicherry"  # alt name
}

# --------------------------
# Helpers
# --------------------------
def safe_read_csv(uploaded_file) -> pd.DataFrame:
    """
    Robust CSV loader that auto-detects encoding (fixes 'utf-8 decode' errors).
    Tries chardet if available; falls back to common encodings.
    """
    raw = uploaded_file.getvalue()
    # Try chardet if present
    encoding_candidates = []
    try:
        import chardet  # type: ignore
        guess = chardet.detect(raw).get("encoding")
        if guess:
            encoding_candidates.append(guess)
    except Exception:
        pass
    # Common encodings to try
    encoding_candidates += ["utf-8", "utf-8-sig", "cp1252", "latin-1"]

    last_err = None
    for enc in encoding_candidates:
        try:
            return pd.read_csv(io.StringIO(raw.decode(enc)))
        except Exception as e:
            last_err = e
            continue
    # If all failed, raise last error
    raise RuntimeError(f"Couldn't read CSV with tried encodings {encoding_candidates}. Last error: {last_err}")

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0  # km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat/2)**2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon/2)**2
    )
    return 2 * R * math.asin(math.sqrt(a))

def classify_special_case(geo_raw: dict, query_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (label, note) if it qualifies for a special rule, else (None, None).

    - Indian State/UT: label = state/UT name; note = "Indian State/UT"
    - Foreign Country: label = country name; note = "Foreign Country"
    """
    if not geo_raw:
        # Try pure text recognition for Indian State/UT if geocode failed completely
        ql = (query_text or "").strip().lower()
        if ql in INDIAN_STATES_UTS:
            return (query_text.strip(), "Indian State/UT")
        return (None, None)

    address = geo_raw.get("address", {})
    country_code = (address.get("country_code") or "").lower()
    state_name = (address.get("state") or "").strip()
    country_name = (address.get("country") or "").strip()
    nominatim_type = (geo_raw.get("type") or "").strip().lower()

    # Check Indian State/UT
    if country_code == INDIA_COUNTRY_CODE:
        st_lower = state_name.lower()
        q_lower = (query_text or "").strip().lower()
        if st_lower in INDIAN_STATES_UTS or q_lower in INDIAN_STATES_UTS:
            # Prefer canonical state name from geocoder if present, else original query
            label = state_name if state_name else query_text.strip()
            return (label, "Indian State/UT")

    # Check Foreign Country (if not India and looks like a country)
    if country_code and country_code != INDIA_COUNTRY_CODE:
        # Mark as foreign country if the feature itself is a country or
        # if the text directly matches the country name
        if nominatim_type == "country" or (country_name and country_name.lower() == (query_text or "").strip().lower()):
            return (country_name or query_text.strip(), "Foreign Country")

    return (None, None)

def nearest_db_location(lat: float, lon: float, db_df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
    if pd.isna(lat) or pd.isna(lon) or db_df is None or db_df.empty:
        return (None, None)
    dists = []
    for _, r in db_df.iterrows():
        try:
            d = haversine_km(lat, lon, float(r["lat"]), float(r["lon"]))
        except Exception:
            d = float("inf")
        dists.append(d)
    if not dists:
        return (None, None)
    i = int(pd.Series(dists).idxmin())
    nearest = db_df.iloc[i]["db_location"]
    return (nearest, float(min(dists)))

# --------------------------
# Sidebar settings
# --------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    country_bias = st.text_input("Country code bias (2 letters)", value="in")
    user_agent = st.text_input(
        "User-Agent (required by Nominatim)",
        value="rahul-geocoder-mapper/1.0 (educational use)"
    )
    rate_delay = st.number_input("Rate limit (seconds/request)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
    dist_cutoff = st.number_input("Optional cutoff (km) to accept nearest DB match", min_value=0.0, value=0.0, step=1.0,
                                  help="0 disables cutoff. If >0, only accept DB match if distance ≤ cutoff.")

    st.markdown("---")
    raw_csv = st.file_uploader("Upload raw_locations.csv (must have column: raw_location)", type=["csv"])
    db_csv = st.file_uploader("Upload db_locations.csv (columns: db_location, lat, lon)", type=["csv"])

# --------------------------
# Main Action
# --------------------------
if st.button("Run Geocode + Map", type="primary"):
    if not raw_csv:
        st.error("Please upload **raw_locations.csv** with a `raw_location` column.")
        st.stop()

    # Load raw safely
    try:
        raw_df = safe_read_csv(raw_csv)
    except Exception as e:
        st.error(f"Couldn't read raw CSV: {e}")
        st.stop()

    if "raw_location" not in raw_df.columns:
        st.error("`raw_locations.csv` must contain a column named **raw_location**.")
        st.stop()

    # Load DB (optional but recommended)
    db_df = pd.DataFrame()
    if db_csv:
        try:
            db_df = safe_read_csv(db_csv)
        except Exception as e:
            st.error(f"Couldn't read DB CSV: {e}")
            st.stop()

        missing_cols = {"db_location", "lat", "lon"} - set(db_df.columns)
        if missing_cols:
            st.error(f"`db_locations.csv` is missing columns: {missing_cols}")
            st.stop()

    # Geocoder setup
    try:
        geolocator = Nominatim(user_agent=user_agent)
    except Exception as e:
        st.error(f"Failed to init Nominatim geocoder. Check your User-Agent. Error: {e}")
        st.stop()

    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=float(rate_delay), swallow_exceptions=True)

    # Work on a copy to preserve other columns
    out = raw_df.copy()
    out["geo_lat"] = None
    out["geo_lon"] = None
    out["geo_display_name"] = None
    out["geo_raw_type"] = None
    out["geo_country_code"] = None
    out["nearest_db_location"] = None
    out["distance_km"] = None
    out["final_label"] = None
    out["final_note"] = None  # "Mapped to DB" | "Indian State/UT" | "Foreign Country" | "Unmapped"

    st.info("Geocoding in progress… please wait (free service: ~1 req/sec).")

    for idx, row in out.iterrows():
        query = str(row["raw_location"]).strip()
        if not query:
            continue

        loc = None
        # Try with bias to India first, then without
        try:
            loc = geocode(query, country_codes=country_bias or None, exactly_one=True, addressdetails=True)
            if not loc:
                loc = geocode(query, exactly_one=True, addressdetails=True)
        except Exception:
            loc = None

        if loc:
            out.at[idx, "geo_lat"] = loc.latitude
            out.at[idx, "geo_lon"] = loc.longitude
            out.at[idx, "geo_display_name"] = loc.address
            out.at[idx, "geo_raw_type"] = (loc.raw or {}).get("type")
            out.at[idx, "geo_country_code"] = ((loc.raw or {}).get("address", {}).get("country_code") or "").lower()

            # First, see if this qualifies for special classification (Indian State/UT or Foreign Country)
            label, note = classify_special_case(loc.raw or {}, query)

            if label and note:
                out.at[idx, "final_label"] = label
                out.at[idx, "final_note"] = note
            else:
                # Else try nearest DB mapping
                if not db_df.empty:
                    nearest, dist = nearest_db_location(float(loc.latitude), float(loc.longitude), db_df)
                    if nearest is not None and (dist_cutoff == 0.0 or (dist is not None and dist <= dist_cutoff)):
                        out.at[idx, "nearest_db_location"] = nearest
                        out.at[idx, "distance_km"] = dist
                        out.at[idx, "final_label"] = nearest
                        out.at[idx, "final_note"] = "Mapped to DB"
                    else:
                        out.at[idx, "final_label"] = query
                        out.at[idx, "final_note"] = "Unmapped"
                else:
                    # No DB provided → keep as-is
                    out.at[idx, "final_label"] = query
                    out.at[idx, "final_note"] = "Unmapped"
        else:
            # Geocode failed → try pure text check for Indian State/UT
            label, note = classify_special_case({}, query)
            if label and note:
                out.at[idx, "final_label"] = label
                out.at[idx, "final_note"] = note
            else:
                out.at[idx, "final_label"] = query
                out.at[idx, "final_note"] = "Unmapped"

    st.success("Done!")

    st.subheader("Preview")
    st.dataframe(out, use_container_width=True)

    # Downloads
    geocoded_cols = [
        c for c in out.columns
    ]
    geocoded_csv = out.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Download results (CSV)",
        geocoded_csv,
        file_name="mapped_output.csv",
        mime="text/csv"
    )

    st.caption("Tip: Use a non-empty cutoff (km) to avoid bad DB matches on far-away points.")
