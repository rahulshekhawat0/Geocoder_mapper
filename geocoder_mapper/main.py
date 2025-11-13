# geocoder_mapper/main.py
import os
import io
import math
import time
import csv
import unicodedata
from typing import Optional, Tuple, Dict, List
from datetime import datetime

import pandas as pd
import streamlit as st
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# =========================
# Config / constants
# =========================
APP_DIR = os.path.dirname(__file__) if "__file__" in globals() else "."
CACHE_PATH = os.path.join(APP_DIR, ".geocode_cache.csv")  # Option A: inside app folder
INDIA_CC = "in"
RATE_DELAY_SEC = 1.05  # be polite to Nominatim

INDIAN_STATES_UTS = {
    # states
    "andhra pradesh","arunachal pradesh","assam","bihar","chhattisgarh","goa",
    "gujarat","haryana","himachal pradesh","jharkhand","karnataka","kerala",
    "madhya pradesh","maharashtra","manipur","meghalaya","mizoram","nagaland",
    "odisha","punjab","rajasthan","sikkim","tamil nadu","telangana","tripura",
    "uttar pradesh","uttarakhand","west bengal",
    # uts
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

# Robust CSV reading
def _pandas_read_csv_from_text(text: str, expected_min_cols=1) -> pd.DataFrame:
    sep = None
    try:
        sample = text[:5000]
        dialect = csv.Sniffer().sniff(sample)
        if getattr(dialect, "delimiter", None) and len(dialect.delimiter) == 1:
            sep = dialect.delimiter
    except Exception:
        sep = None

    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, engine="python", on_bad_lines="skip")
        if df.shape[1] >= expected_min_cols:
            return df
    except Exception:
        pass

    for candidate in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=candidate, engine="python", on_bad_lines="skip")
            if df.shape[1] >= expected_min_cols:
                return df
        except Exception:
            continue

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
# Streamlit UI
# =========================
st.set_page_config(page_title="Free Geocoder + DB Mapper", page_icon="🗺️", layout="wide")
st.title("🗺️ Free Location Geocoder + DB Mapper")
st.write(
    "Upload **raw locations** (text) and your **DB locations** (names or with lat/lon). "
    "We geocode raw text via OpenStreetMap Nominatim, map to nearest DB location, and cache results."
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    country_bias = st.text_input("Country code bias (2 letters)", value=INDIA_CC)
    user_agent = st.text_input("User-Agent (required by Nominatim)", value="rahul-geocoder-mapper/1.0 (educational)")
    max_rows = st.number_input("Limit rows (for testing)", min_value=0, value=0, step=1,
                               help="0 = no limit")
    cutoff_km = st.number_input("Cutoff distance (km) to accept nearest DB match", min_value=0.0, value=0.0, step=1.0,
                                help="0 = always accept nearest. If >0, farther rows become 'No reliable match'.")
    prefer_full_string = st.checkbox("Prefer full location string (do NOT trim after commas)", value=False)
    strict_country = st.checkbox("Strict country match (must match bias)", value=False)
    allow_db_geocode = st.checkbox("If DB lacks lat/lon, geocode DB names (slower)", value=True)
    show_debug = st.checkbox("Show debug panel", value=False)
    use_cache = st.checkbox("Use local cache file (fast)", value=True)
    run_button = st.button("Run Mapping")

# Debug/log storage
if "debug_logs" not in st.session_state:
    st.session_state.debug_logs = []
def log(msg: str):
    if show_debug:
        st.session_state.debug_logs.append(msg)

# =========================
# Cache handling (Option A)
# =========================
def load_cache(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["q","lat","lon","display_name","country","country_code","when"])
    try:
        df = pd.read_csv(path, dtype=str)
        # ensure columns
        for c in ["q","lat","lon","display_name","country","country_code","when"]:
            if c not in df.columns:
                df[c] = None
        return df
    except Exception:
        return pd.DataFrame(columns=["q","lat","lon","display_name","country","country_code","when"])

def save_cache(df: pd.DataFrame, path: str):
    try:
        df.to_csv(path, index=False, encoding="utf-8")
    except Exception as e:
        st.warning(f"Failed to save cache: {e}")

cache_df = load_cache(CACHE_PATH) if os.path.exists(CACHE_PATH) else load_cache(CACHE_PATH)

# =========================
# File uploads
# =========================
left, right = st.columns(2)
with left:
    raw_file = st.file_uploader("Upload RAW locations CSV", type=["csv"], help="Needs one column containing raw location text.")
with right:
    db_file = st.file_uploader("Upload DB locations CSV", type=["csv"], help="Preferred columns: db_location, lat, lon (lat/lon optional).")

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
        db_name_col = st.selectbox("Select DB location name column", db_df.columns.tolist(), index=db_df.columns.get_loc(guess_name))
        db_lat_col = st.selectbox("Select DB latitude column (or 'None' if absent)", ["None"] + db_df.columns.tolist(),
                                  index=(db_df.columns.get_loc("lat")+1) if "lat" in db_df.columns else 0)
        db_lon_col = st.selectbox("Select DB longitude column (or 'None' if absent)", ["None"] + db_df.columns.tolist(),
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
    try:
        geolocator, geocode = get_geocoder(user_agent.strip())
    except Exception as e:
        st.error(f"Failed to initialize geocoder: {e}")

# Candidate generation
def build_candidates(q: str, cc_bias: Optional[str]) -> List[str]:
    q = normalize_txt(q)
    if not q:
        return []
    parts = [p.strip() for p in q.split(",") if p.strip()]
    cand = []
    if prefer_full_string or len(parts) == 1:
        cand.append(q)
    else:
        cand.append(parts[0])
        cand.append(q)
    if (cc_bias or "").lower() == "in":
        for base in list(cand):
            if not base.lower().endswith("india"):
                cand.append(f"{base}, India")
    # keep unique order
    seen, uniq = set(), []
    for x in cand:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    log(f"Candidates for '{q}': {uniq}")
    return uniq

# Check cache first
def lookup_cache(q: str) -> Optional[Dict]:
    if not use_cache:
        return None
    qn = normalize_txt(q)
    if cache_df.empty:
        return None
    found = cache_df[cache_df["q"] == qn]
    if not found.empty:
        r = found.iloc[-1]  # most recent
        try:
            return {
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
                "display_name": r.get("display_name", None),
                "country": r.get("country", None),
                "country_code": r.get("country_code", None),
            }
        except Exception:
            return None
    return None

# Save a single cache entry (append)
def append_cache_entry(q: str, lat: float, lon: float, display_name: str, country: Optional[str], country_code: Optional[str]):
    global cache_df
    qn = normalize_txt(q)
    row = {"q": qn, "lat": lat, "lon": lon, "display_name": display_name, "country": country, "country_code": country_code, "when": datetime.utcnow().isoformat()}
    try:
        if cache_df is None or cache_df.empty:
            cache_df = pd.DataFrame([row])
        else:
            cache_df = pd.concat([cache_df, pd.DataFrame([row])], ignore_index=True, axis=0)
        save_cache(cache_df, CACHE_PATH)
    except Exception as e:
        log(f"Failed to append cache: {e}")

# Geocode with candidate tries; use cache if available
def geocode_text(q: str, cc_bias: Optional[str]) -> Optional[Tuple[float, float, Dict]]:
    if not geocode or not q or not q.strip():
        return None

    # cache lookup
    cached = lookup_cache(q)
    if cached:
        log(f"♻️ Cache hit: '{normalize_txt(q)}' -> {cached['lat']},{cached['lon']}")
        return float(cached["lat"]), float(cached["lon"]), {"display_name": cached.get("display_name"), "address": {"country": cached.get("country")}, "country_code": cached.get("country_code")}

    for cand in build_candidates(q, cc_bias):
        try:
            loc = geocode(cand, addressdetails=True, country_codes=cc_bias, exactly_one=True)
        except Exception:
            # swallow & continue (RateLimiter may handle delays)
            loc = None

        # If exactly_one=True returns None, try exactly_one=False and take first (sometimes works)
        if not loc:
            try:
                locs = geolocator.geocode(cand, addressdetails=True, country_codes=cc_bias, exactly_one=False)
                if locs and isinstance(locs, list) and len(locs) > 0:
                    loc = locs[0]
            except Exception:
                loc = None

        if loc:
            lat, lon = float(loc.latitude), float(loc.longitude)
            raw = loc.raw or {}
            addr = raw.get("address", {}) if isinstance(raw, dict) else {}
            display_name = raw.get("display_name", "") if isinstance(raw, dict) else ""
            country_name = addr.get("country")
            country_code = addr.get("country_code")
            log(f"✅ Hit: '{cand}' -> {lat},{lon} ({display_name[:80]}...)")
            # append to cache
            try:
                append_cache_entry(cand, lat, lon, display_name, country_name, country_code)
            except Exception as e:
                log(f"Cache append failed: {e}")
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
        time.sleep(0.01)
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

    if is_indian_state_or_ut(raw_txt_norm):
        result["mapped_to"] = raw_txt_norm
        result["label"] = "indian_state"
        return result

    g = geocode_text(raw_txt_norm, cc_bias)
    if not g:
        result["label"] = "not_geocoded"
        return result

    lat, lon, raw = g
    result["lat"], result["lon"] = lat, lon
    country_name, country_code = extract_country_info(raw)
    result["country"], result["country_code"] = country_name, country_code

    if strict_country and country_code and cc_bias and country_code.lower() != cc_bias.lower():
        result["mapped_to"] = country_name or raw_txt_norm
        result["label"] = "foreign_country" if country_code.lower() != INDIA_CC else "no_reliable_match"
        return result

    if country_code and country_code.lower() != INDIA_CC:
        result["mapped_to"] = country_name or raw_txt_norm
        result["label"] = "foreign_country"
        return result

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
    # reset debug logs
    st.session_state.debug_logs = []
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
        st.download_button("⬇️ Download results CSV", data=csv_bytes, file_name="mapped_output.csv", mime="text/csv")

        st.success("Done!")
        st.info("Labels: nearest_db, indian_state, foreign_country, no_reliable_match, not_geocoded")
    except Exception as e:
        st.error(f"Processing failed: {e}")

# Debug panel
if show_debug and st.session_state.debug_logs:
    with st.expander("🔎 Debug panel (geocoding attempts)"):
        for line in st.session_state.debug_logs:
            st.write(line)
