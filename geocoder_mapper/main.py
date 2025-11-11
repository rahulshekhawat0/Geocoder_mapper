import time
import math
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
""")

# --------------------------
# Helper: Haversine distance
# --------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat/2)**2 +
        math.cos(math.radians(lat1)) *
        math.cos(math.radians(lat2)) *
        math.sin(dlon/2)**2
    )
    return 2 * R * math.asin(math.sqrt(a))

# --------------------------
# Sidebar settings
# --------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    country_bias = st.text_input(
        "Country code bias (2 letters)",
        value="in"
    )

    user_agent = st.text_input(
        "User-Agent (required by Nominatim)",
