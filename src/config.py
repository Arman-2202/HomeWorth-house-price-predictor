"""
Central configuration for the HomeWorth house-price project.

Everything that describes the *Indian real-estate market* lives here:
cities, localities and their benchmark rates (Rs per sq ft), stamp-duty
estimates, amenity list, etc.  The data generator, model, and app all read
from this one file, so you can edit a locality rate here and re-train.

NOTE ON THE RATES
-----------------
`localities` maps  locality name -> approximate benchmark rate in Rs / sq ft
for a ready-to-move 2-3 BHK apartment (2025-26 levels).  They are calibrated to
publicly reported market ranges (e.g. Anarock / PropTiger quarterly reports,
where the top-7-city average in Q3 2026 was about Rs 9,700 / sq ft and
Delhi-NCR about Rs 9,980 / sq ft).  They are indicative, not live listings.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Project meta  (EDIT THESE before your demo)
# --------------------------------------------------------------------------
APP_NAME = "HomeWorth"
APP_TAGLINE = "AI-powered property valuation for Indian cities"
STUDENT_NAME = "Your Name"
COLLEGE_NAME = "Your College / University"
PROJECT_TITLE = "House Price Prediction using Machine Learning"
ACADEMIC_YEAR = "B.Tech (3rd Year) | 2026"

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
ASSET_DIR = ROOT / "assets"
DATA_FILE = DATA_DIR / "housing_data.csv"
MODEL_FILE = MODEL_DIR / "price_model.joblib"
METRICS_FILE = MODEL_DIR / "metrics.json"
HOLDOUT_FILE = MODEL_DIR / "holdout_sample.csv"

# --------------------------------------------------------------------------
# Cities.  rate = Rs / sq ft benchmark per locality
#   stamp   = approx. stamp duty + registration, % of property value
#   parking = approx. value of ONE covered car park (Rs lakh) when sold separately
#   bf/house/villa = share of listings that are builder floors / independent
#                    houses / villas (rest are apartments, 4% penthouses)
#   weight  = relative number of training records
# --------------------------------------------------------------------------
CITIES = {
    "Mumbai": dict(
        state="Maharashtra", tagline="The City of Dreams", stamp=7.0, parking=18.0,
        bf=0.00, house=0.03, villa=0.01, weight=1.4,
        localities={
            "Worli": 50000, "Bandra West": 46000, "Juhu": 42000, "Lower Parel": 38000,
            "Andheri West": 25500, "Powai": 24000, "Chembur": 22000, "Goregaon": 20000,
            "Thane West": 13000, "Mira Road": 10500, "Navi Mumbai (Kharghar)": 9800,
        }),
    "Delhi": dict(
        state="Delhi", tagline="The Capital Region", stamp=7.0, parking=5.0,
        bf=0.35, house=0.10, villa=0.02, weight=1.1,
        localities={
            "Vasant Vihar": 48000, "Defence Colony": 36000, "Greater Kailash": 28000,
            "Saket": 17500, "Janakpuri": 12500, "Pitampura": 11500, "Mayur Vihar": 10500,
            "Dwarka": 9500, "Rohini": 9000, "Laxmi Nagar": 8000,
        }),
    "Gurugram": dict(
        state="Haryana", tagline="The Millennium City", stamp=7.0, parking=4.0,
        bf=0.25, house=0.04, villa=0.06, weight=1.0,
        localities={
            "Golf Course Road": 27000, "DLF Phases 1-5": 22000, "MG Road": 17000,
            "Golf Course Extension": 15500, "Dwarka Expressway": 11500, "Sohna Road": 9500,
            "New Gurugram (Sec 80-95)": 8200, "Manesar": 6000,
        }),
    "Noida": dict(
        state="Uttar Pradesh", tagline="The Planned City", stamp=7.0, parking=2.5,
        bf=0.10, house=0.06, villa=0.02, weight=0.9,
        localities={
            "Sector 128 (Expressway)": 12000, "Sector 44-50": 11000, "Sector 150": 10000,
            "Sector 62": 8800, "Sector 75-78": 8200, "Sector 137": 7800,
            "Greater Noida West": 5800, "Yamuna Expressway": 4200,
        }),
    "Bengaluru": dict(
        state="Karnataka", tagline="The Silicon Valley of India", stamp=6.6, parking=3.5,
        bf=0.03, house=0.07, villa=0.07, weight=1.5,
        localities={
            "Indiranagar": 21000, "Koramangala": 19500, "Jayanagar": 17500, "HSR Layout": 14500,
            "Hebbal": 11500, "Marathahalli": 10500, "Whitefield": 9800, "Sarjapur Road": 8800,
            "Bannerghatta Road": 8500, "Yelahanka": 7600, "Electronic City": 6600,
            "Devanahalli": 6200,
        }),
    "Hyderabad": dict(
        state="Telangana", tagline="The City of Pearls", stamp=6.0, parking=3.0,
        bf=0.02, house=0.08, villa=0.09, weight=1.3,
        localities={
            "Jubilee Hills": 20000, "Banjara Hills": 19000, "HITEC City": 11500, "Madhapur": 11000,
            "Kokapet": 10500, "Gachibowli": 10000, "Kondapur": 9500, "Financial District": 9200,
            "Kukatpally": 7500, "Miyapur": 6700, "Tellapur": 6200, "Uppal": 5800,
        }),
    "Pune": dict(
        state="Maharashtra", tagline="The Oxford of the East", stamp=7.0, parking=3.5,
        bf=0.04, house=0.06, villa=0.06, weight=1.2,
        localities={
            "Koregaon Park": 18000, "Kalyani Nagar": 14500, "Kothrud": 12800, "Aundh": 12200,
            "Baner": 11800, "Viman Nagar": 11500, "Kharadi": 9800, "Hadapsar": 8300,
            "Wakad": 8000, "Hinjewadi": 7600, "Undri": 6800, "Wagholi": 5800,
        }),
    "Chennai": dict(
        state="Tamil Nadu", tagline="The Gateway of South India", stamp=9.0, parking=3.5,
        bf=0.04, house=0.10, villa=0.05, weight=1.0,
        localities={
            "Nungambakkam": 22000, "T. Nagar": 18500, "Adyar": 16500, "Besant Nagar": 16500,
            "Anna Nagar": 14500, "Guindy": 11500, "ECR": 10500, "Velachery": 9500,
            "Porur": 8800, "OMR (Sholinganallur)": 7800, "Tambaram": 6300, "Perumbakkam": 5800,
        }),
    "Kolkata": dict(
        state="West Bengal", tagline="The City of Joy", stamp=7.0, parking=4.0,
        bf=0.12, house=0.12, villa=0.02, weight=0.9,
        localities={
            "Alipore": 17000, "Park Street": 14500, "Ballygunge": 13500, "Salt Lake": 8800,
            "EM Bypass": 7800, "Tollygunge": 7000, "New Town": 6600, "Rajarhat": 6200,
            "Behala": 5800, "Dum Dum": 5600, "Garia": 5300, "Howrah": 5000,
        }),
    "Ahmedabad": dict(
        state="Gujarat", tagline="The Manchester of India", stamp=5.9, parking=2.0,
        bf=0.04, house=0.12, villa=0.08, weight=0.8,
        localities={
            "Bodakdev": 9000, "Thaltej": 8200, "Vastrapur": 8000, "Prahlad Nagar": 8000,
            "SG Highway": 7800, "Satellite": 7600, "Naranpura": 6500, "Shela": 6000,
            "Bopal": 5800, "Gota": 5000, "Maninagar": 4800, "Chandkheda": 4300,
        }),
    "Chandigarh": dict(
        state="Chandigarh (UT)", tagline="The City Beautiful", stamp=6.0, parking=3.0,
        bf=0.30, house=0.22, villa=0.03, weight=0.55,
        localities={
            "Sector 8-11": 14500, "Sector 19-27": 12000, "Sector 33-38": 11000,
            "Sector 40-47": 9800, "Sector 48-56": 8800, "Manimajra": 6500,
        }),
    "Mohali": dict(
        state="Punjab", tagline="The Tricity's Growth Hub", stamp=7.0, parking=2.0,
        bf=0.25, house=0.12, villa=0.04, weight=0.6,
        localities={
            "Sector 66-69": 7800, "Aerotropolis / IT City": 7200, "Sector 70": 6900,
            "Phase 7-11": 6300, "Sector 82 (JLPL)": 5900, "Zirakpur": 5400, "Kharar": 4500,
        }),
    "Jaipur": dict(
        state="Rajasthan", tagline="The Pink City", stamp=6.0, parking=1.5,
        bf=0.08, house=0.14, villa=0.07, weight=0.6,
        localities={
            "C-Scheme": 12500, "Malviya Nagar": 7600, "Vaishali Nagar": 6300, "Tonk Road": 5700,
            "Jagatpura": 5400, "Mansarovar": 5200, "Ajmer Road": 4500, "Sirsi Road": 4100,
        }),
    "Lucknow": dict(
        state="Uttar Pradesh", tagline="The City of Nawabs", stamp=7.0, parking=1.5,
        bf=0.06, house=0.18, villa=0.05, weight=0.55,
        localities={
            "Hazratganj": 9500, "Gomti Nagar": 7800, "Indira Nagar": 6200, "Aliganj": 6000,
            "Sushant Golf City": 5600, "Alambagh": 5000, "Faizabad Road": 4800,
            "Jankipuram": 4500,
        }),
    "Kochi": dict(
        state="Kerala", tagline="The Queen of the Arabian Sea", stamp=10.0, parking=2.0,
        bf=0.02, house=0.20, villa=0.06, weight=0.5,
        localities={
            "Marine Drive": 10500, "Panampilly Nagar": 9200, "Kadavanthra": 8300,
            "Edappally": 7200, "Vyttila": 6400, "Kakkanad": 6100, "Tripunithura": 5000,
            "Aluva": 4600,
        }),
    "Indore": dict(
        state="Madhya Pradesh", tagline="The Cleanest City", stamp=8.0, parking=1.5,
        bf=0.06, house=0.16, villa=0.08, weight=0.5,
        localities={
            "Palasia": 7600, "Vijay Nagar": 6600, "Scheme 78": 5800, "AB Road": 5500,
            "Bhawarkua": 5000, "Super Corridor": 4500, "Nipania": 4300, "Rau": 4200,
        }),
}

# --------------------------------------------------------------------------
# Property options
# --------------------------------------------------------------------------
PROPERTY_TYPES = ["Apartment", "Builder Floor", "Independent House", "Villa", "Penthouse"]
# price multiplier relative to an apartment (land value, exclusivity, etc.)
TYPE_FACTOR = {
    "Apartment": 1.00, "Builder Floor": 1.04, "Independent House": 1.10,
    "Villa": 1.30, "Penthouse": 1.40,
}
FLOOR_TYPES = ["Apartment", "Builder Floor", "Penthouse"]       # types that have a floor number
LAND_TYPES = ["Independent House", "Villa"]                      # types described by storeys

FURNISHING = ["Unfurnished", "Semi-Furnished", "Fully Furnished"]
FURNISH_FACTOR = {"Unfurnished": 1.00, "Semi-Furnished": 1.03, "Fully Furnished": 1.07}

CONSTRUCTION = ["Ready to Move", "Under Construction"]

# Vastu-driven preference in India: East / North facing homes sell at a small premium
FACINGS = ["East", "North", "West", "South"]
FACING_FACTOR = {"East": 1.02, "North": 1.015, "West": 0.99, "South": 0.985}

# label -> (column name, price uplift used by the market simulator)
AMENITIES = {
    "Lift": ("am_lift", 0.0),
    "Power Backup": ("am_power_backup", 0.010),
    "24x7 Security / CCTV": ("am_security", 0.010),
    "Gym": ("am_gym", 0.012),
    "Swimming Pool": ("am_pool", 0.020),
    "Clubhouse": ("am_clubhouse", 0.018),
    "Children's Play Area": ("am_play_area", 0.006),
    "Garden / Park": ("am_garden", 0.010),
    "Gated Community": ("am_gated", 0.015),
}
AMENITY_COLS = [c for c, _ in AMENITIES.values()]

# --------------------------------------------------------------------------
# Model feature lists
# --------------------------------------------------------------------------
CAT_FEATURES = ["city", "property_type", "furnishing", "construction_status", "facing"]
NUM_FEATURES = [
    "locality_rate", "area_sqft", "bhk", "bathrooms", "balconies", "floor",
    "total_floors", "age_years", "parking", "metro_km", "rera", "near_social", "modular_kitchen",
] + AMENITY_COLS
FEATURES = CAT_FEATURES + NUM_FEATURES
TARGET = "price_inr"

# human friendly names for charts
FEATURE_LABELS = {
    "locality_rate": "Locality benchmark rate", "area_sqft": "Built-up area",
    "city": "City", "property_type": "Property type", "bhk": "BHK",
    "bathrooms": "Bathrooms", "balconies": "Balconies", "floor": "Floor number",
    "total_floors": "Building height", "age_years": "Age of property",
    "parking": "Car parking", "metro_km": "Distance to metro / transit",
    "rera": "RERA registered", "furnishing": "Furnishing",
    "construction_status": "Construction status", "amenities": "Amenities (all)",
    "facing": "Facing direction", "near_social": "Schools / hospitals nearby", "modular_kitchen": "Modular kitchen",
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def locality_rate(city: str, locality: str) -> float:
    return float(CITIES[city]["localities"][locality])


def city_median_rate(city: str) -> float:
    vals = sorted(CITIES[city]["localities"].values())
    n = len(vals)
    return float(vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2)


def locality_tier(city: str, locality: str) -> str:
    ratio = locality_rate(city, locality) / city_median_rate(city)
    if ratio >= 1.5:
        return "Premium"
    if ratio >= 1.1:
        return "Upper-Mid"
    if ratio >= 0.85:
        return "Mid-Range"
    return "Emerging"
