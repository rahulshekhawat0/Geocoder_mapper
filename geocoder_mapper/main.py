import io
import math
import time
import csv
import unicodedata
from typing import Optional, Tuple, Dict, List

import pandas as pd
import streamlit as st
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# =========================
# Page setup
# =========================
st.set_page_config(page_title="Free Geocoder + DB Mapper", page_icon="🗺️", layout="wide")
st.title("🗺️ Free Location Geocoder + DB Mapper")
st.write(
    "Upload **raw locations** (text) and your **DB locations**. "
    "We geocode raw text (free via OpenStreetMap Nominatim) and map to the nearest DB location. "
    "Special handling: Indian State/UT labels and foreign countries."
)

# =========================
# Constants
# =========================
INDIA_CC = "in"
RATE_DELAY_SEC = 1.05  # be polite to Nominatim

INDIAN_STATES_UTS = {
    # States
    "andhra pradesh","arunachal pradesh","assam","bihar","chhattisgarh","goa",
    "gujarat","haryana","himachal pradesh","jharkhand","karnataka","kerala",
    "madhya pradesh","maharashtra","manipur","meghalaya","mizoram","nagaland",
    "odisha","punjab","rajasthan","sikkim","tamil nadu","telangana","tripura",
    "uttar pradesh","uttarakhand","west bengal",
    # UTs
    "andaman and nicobar islands","chandigarh","dadra and nagar haveli and daman and diu",
    "lakshadweep","delhi","puducherry","jammu and kashmir","ladakh"
}

# =========================
# Helpers
# =========================
def normalize_txt(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s).strip()
    return " ".join(s.split())

def is_indian_state_or_ut(name: str) -> bool:
    return normalize_txt(name).lower() in INDIAN_STATES_UTS

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    a = (math.sin(dphi/2)**2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2)
    return 2 * R * math.asin(math.sqrt(a))

def _pandas_read_csv_from_text(text: str, expected_min_cols=1) -> pd.DataFrame:
    """Very robust CSV parsing with multiple delimiter fallbacks."""
    # 1) Try Sniffer
    sep = None
    try:
        sample = text[:5000]
        dialect = csv.Sniffer().sniff(sample)
        if getattr(dialect, "delimiter", None) and len(dialect.delimiter) == 1:
            sep = dialect.delimiter
    except Exception:
        sep = None

    # 2) Try pandas inference first
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, engine="python", on_bad_lines="skip")
        if df.shape[1] >= expected_min_cols:
            return df
    except Exception:
        pass

    # 3) Try common delimiters manually
    for candidate in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=candidate, engine="python", on_bad_lines="skip")
            if df.shape[1] >= expected_min_cols:
                return df
        except Exception:
            continue

    # 4) Last resort: whitespace
    df = pd.read_csv(io.StringIO(text), delim_whitespace=True, engine="python", on_bad_lines="skip")
    if df.shape[1] < expected_min_cols:
        raise ValueError(f"Only {df.shape[1]} columns detected, need ≥ {expected_min_cols}.")
    return df

def robust_read_upload(upload, expected_min_cols=1) -> pd.DataFrame:
    if upload is None:
        raise ValueError("No file provided.")
    raw = upload.read()
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            text = raw.decode(enc, errors="replace")
            return _pandas_read_csv_from_text(text, expected_min_cols=expected_min_cols)
        except Exception as e:
            last_err = e
    raise ValueError(f"Couldn't read CSV with tried encodings {encodings}. Last error: {last_err}")

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("⚙️ Settings")

    country_bias = st.text_input("Country code bias (2 letters)", value=INDIA_CC)
    user_agent = st.text_input("User-Agent (required by Nominatim)", value="rahul-geocoder-mapper/1.0 (educational)")

    max_rows = st.number_input("Limit rows (for testing)", min_value=0, value=0, step=1, help="0 = no limit")
    cutoff_km = st.number_input("Cutoff distance (km) to accept nearest DB match", min_value=0.0, value=0.0, step=1.0,
                                help="0 = always accept nearest. If >0, farther rows become 'No reliable match'.")

    prefer_full_string = st.checkbox("Prefer full location string (do NOT trim after commas)", value=False)
    strict_country = st.checkbox("Strict country match (must match bias)", value=False)

    allow_db_geocode = st.checkbox("If DB lacks lat/lon, geocode DB names (slower)", value=True)
    show_debug = st.checkbox("Show debug panel", value=False)

    run_button = st.button("Run Mapping")

# Debug store
if "debug_logs" not in st.session_state:
    st.session_state.debug_logs = []  # list of strings

def log(msg: str):
    if show_debug:
        st.session_state.debug_logs.append(msg)

# =========================
# File uploads
# =========================
left, right = st.columns(2)
with left:
    raw_file = st.file_uploader("Upload RAW locations CSV", type=["csv"],
                                help="Needs one column that has the free-text locations.")
with right:
    db_file = st.file_uploader("Upload DB locations CSV", type=["csv"],
                               help="Preferred columns: db_location, lat, lon (if lat/lon missing we can geocode names).")

raw_col = None
db_name_col = None
db_lat_col = None
db_lon_col = None

if raw_file is not None:
    try:
        raw_df = robust_read_upload(raw_file, expected_min_cols=1)
        st.success(f"RAW loaded: {raw_df.shape[0]} rows, {raw_df.shape[1]} cols")
        raw_col = st.selectbox("Select RAW location text column", raw_df.columns.tolist())
    except Exception as e:
        st.error(f"Couldn't read RAW CSV: {e}")
        raw_df = None
else:
    raw_df = None

if db_file is not None:
    try:
        db_df = robust_read_upload(db_file, expected_min_cols=1)
        st.success(f"DB loaded: {db_df.shape[0]} rows, {db_df.shape[1]} cols")
        guess_name = "db_location" if "db_location" in db_df.columns else db_df.columns[0]
        db_name_col = st.selectbox("Select DB location name column", db_df.columns.tolist(),
                                   index=db_df.columns.get_loc(guess_name))
        db_lat_col = st.selectbox("Select DB latitude column (or 'None' if absent)",
                                  ["None"] + db_df.columns.tolist(),
                                  index=(db_df.columns.get_loc("lat")+1) if "lat" in db_df.columns else 0)
        db_lon_col = st.selectbox("Select DB longitude column (or 'None' if absent)",
                                  ["None"] + db_df.columns.tolist(),
                                  index=(db_df.columns.get_loc("lon")+1) if "lon" in db_df.columns else 0)
    except Exception as e:
        st.error(f"Couldn't read DB CSV: {e}")
        db_df = None
else:
    db_df = None

# =========================
# Geocoder
# =========================
@st.cache_resource(show_spinner=False)
def get_geocoder(ua: str):
    geolocator = Nominatim(user_agent=ua)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=RATE_DELAY_SEC, swallow_exceptions=True)
    return geolocator, geocode

geolocator = None
geocode = None
if user_agent.strip():
    geolocator, geocode = get_geocoder(user_agent.strip())

def build_candidates(q: str, cc_bias: Optional[str]) -> List[str]:
    """Generate smart candidates from a raw string."""
    q = normalize_txt(q)
    if not q:
        return []

    parts = [p.strip() for p in q.split(",") if p.strip()]
    cand = []

    # Prefer full string or trimmed first token
    if prefer_full_string or len(parts) == 1:
        cand.append(q)
    else:
        cand.append(parts[0])                  # e.g., 'Sohagpur'
        cand.append(q)                         # full 'Sohagpur, MP, India'

    # Add country-biased variants
    if (cc_bias or "").lower() == "in":
        for base in list(cand):
            if not base.lower().endswith("india"):
                cand.append(f"{base}, India")

    # Dedupe, keep order
    seen, uniq = set(), []
    for x in cand:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    log(f"Candidates for '{q}': {uniq}")
    return uniq

def geocode_text(q: str, cc_bias: Optional[str]) -> Optional[Tuple[float, float, Dict]]:
    if not geocode or not q or not q.strip():
        return None
    for cand in build_candidates(q, cc_bias):
        loc = geocode(cand, addressdetails=True, country_codes=cc_bias, exactly_one=True)
        if loc:
            lat, lon = float(loc.latitude), float(loc.longitude)
            raw = loc.raw or {}
            log(f"✅ Hit: '{cand}' -> {lat},{lon} ({raw.get('display_name','')[:80]}...)")
            return lat, lon, raw
        else:
            log(f"❌ Miss: '{cand}'")
    return None

def extract_country_info(raw: Dict) -> Tuple[Optional[str], Optional[str]]:
    addr = raw.get("address", {}) if isinstance(raw, dict) else {}
    return addr.get("country"), addr.get("country_code")

# =========================
# DB lat/lon prep
# =========================
def ensure_db_latlon(df: pd.DataFrame, name_col: str, lat_col_opt: Optional[str], lon_col_opt: Optional[str],
                     cc_bias: Optional[str], allow_geocode_db: bool) -> pd.DataFrame:
    work = df.copy()
    work[name_col] = work[name_col].astype(str).map(normalize_txt)

    has_lat = lat_col_opt and lat_col_opt != "None" and lat_col_opt in work.columns
    has_lon = lon_col_opt and lon_col_opt != "None" and lon_col_opt in work.columns

    if has_lat and has_lon:
        work["__lat"] = pd.to_numeric(work[lat_col_opt], errors="coerce")
        work["__lon"] = pd.to_numeric(work[lon_col_opt], errors="coerce")
        return work[[name_col, "__lat", "__lon"]].rename(columns={name_col: "db_location"})

    if not allow_geocode_db:
        raise ValueError("DB lacks lat/lon and geocoding DB is disabled. Enable it in sidebar.")
    if geocode is None:
        raise ValueError("Geocoder not ready. Provide a valid User-Agent.")

    uniq = work[name_col].dropna().unique().tolist()
    rows = []
    prog = st.progress(0.0, text="Geocoding DB locations…")
    for i, nm in enumerate(uniq, start=1):
        g = geocode_text(nm, cc_bias)
        if g:
            rows.append({"db_location": nm, "__lat": g[0], "__lon": g[1]})
        prog.progress(i/len(uniq), text=f"Geocoding DB locations… ({i}/{len(uniq)})")
    prog.empty()
    if not rows:
        raise ValueError("Could not geocode any DB locations.")
    return pd.DataFrame(rows)

# =========================
# Mapping logic
# =========================
def map_one(raw_txt: str, db_tbl: pd.DataFrame, cutoff: float, cc_bias: Optional[str]) -> Dict:
    result = {
        "raw_location": raw_txt,
        "mapped_to": None,
        "label": None,            # nearest_db | indian_state | foreign_country | no_reliable_match | not_geocoded
        "distance_km": None,
        "country": None,
        "country_code": None,
        "lat": None,
        "lon": None,
    }
    raw_txt_norm = normalize_txt(raw_txt)

    # Rule 1: If direct Indian State/UT string
    if is_indian_state_or_ut(raw_txt_norm):
        result["mapped_to"] = raw_txt_norm
        result["label"] = "indian_state"
        return result

    # Geocode raw
    g = geocode_text(raw_txt_norm, cc_bias)
    if not g:
        result["label"] = "not_geocoded"
        return result

    lat, lon, raw = g
    result["lat"], result["lon"] = lat, lon
    country_name, country_code = extract_country_info(raw)
    result["country"], result["country_code"] = country_name, country_code

    # Strict country check
    if strict_country and country_code and cc_bias and country_code.lower() != cc_bias.lower():
        result["mapped_to"] = country_name or raw_txt_norm
        result["label"] = "foreign_country" if country_code.lower() != INDIA_CC else "no_reliable_match"
        return result

    # If outside India
    if country_code and country_code.lower() != INDIA_CC:
        result["mapped_to"] = country_name or raw_txt_norm
        result["label"] = "foreign_country"
        return result

    # Nearest DB
    if db_tbl.empty:
        result["label"] = "no_reliable_match"
        return result

    dists = db_tbl.apply(lambda r: haversine(lat, lon, r["__lat"], r["__lon"]), axis=1)
    idx = dists.idxmin()
    nearest_name = db_tbl.loc[idx, "db_location"]
    nearest_dist = float(dists.min())

    if cutoff and cutoff > 0 and nearest_dist > cutoff:
        result["label"] = "no_reliable_match"
        result["distance_km"] = nearest_dist
        return result

    result["mapped_to"] = nearest_name
    result["label"] = "nearest_db"
    result["distance_km"] = nearest_dist
    return result

# =========================
# Run
# =========================
if run_button:
    st.session_state.debug_logs = []  # reset each run if debug is on
    if raw_df is None or db_df is None:
        st.error("Please upload both RAW and DB CSVs.")
        st.stop()
    if raw_col is None or db_name_col is None:
        st.error("Please select the required columns.")
        st.stop()

    try:
        with st.spinner("Preparing DB lat/lon…"):
            db_ready = ensure_db_latlon(db_df, db_name_col, db_lat_col, db_lon_col, country_bias, allow_db_geocode)
            if db_ready[["__lat", "__lon"]].isna().any().any():
                st.warning("Some DB rows missing lat/lon after processing; they will be ignored.")
                db_ready = db_ready.dropna(subset=["__lat", "__lon"]).reset_index(drop=True)

        work_raw = raw_df.copy()
        if max_rows and max_rows > 0:
            work_raw = work_raw.head(int(max_rows))

        results = []
        total = work_raw.shape[0]
        prog = st.progress(0.0, text="Mapping…")
        for i, val in enumerate(work_raw[raw_col].astype(str).tolist(), start=1):
            try:
                res = map_one(val, db_ready, cutoff_km, country_bias)
            except Exception as e:
                res = {
                    "raw_location": val,
                    "mapped_to": None,
                    "label": f"error: {e}",
                    "distance_km": None,
                    "country": None,
                    "country_code": None,
                    "lat": None,
                    "lon": None,
                }
            results.append(res)
            if total:
                prog.progress(i/total, text=f"Mapping… ({i}/{total})")
            time.sleep(0.01)
        prog.empty()

        out_df = pd.DataFrame(results)
        st.subheader("Preview")
        st.dataframe(out_df.head(50), use_container_width=True)

        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download results CSV", data=csv_bytes,
                           file_name="mapped_output.csv", mime="text/csv")

        st.success("Done!")
        st.info("Labels: nearest_db, indian_state, foreign_country, no_reliable_match, not_geocoded")
    except Exception as e:
        st.error(f"Processing failed: {e}")

# =========================
# Debug panel
# =========================
if show_debug and st.session_state.debug_logs:
    with st.expander("🔎 Debug panel (geocoding attempts)"):
        for line in st.session_state.debug_logs:
            st.write(line)
