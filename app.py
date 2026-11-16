"""
HomeWorth  |  Tell us your requirements -> get the price of your house.
Run with:  streamlit run app.py
"""
import base64
from datetime import datetime
from functools import lru_cache

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import charts as ch
from src import config as C
from src import finance as fin
from src.predictor import Valuer, ensure_model, inr, load_metrics, words_in_indian

st.set_page_config(page_title=f"{C.APP_NAME} | Find the price of your house", page_icon="\U0001F3E0",
                   layout="wide", initial_sidebar_state="collapsed")

TYPE_IMG = {"Apartment": "type_apartment", "Builder Floor": "type_builder_floor",
            "Independent House": "type_independent_house", "Villa": "type_villa",
            "Penthouse": "type_penthouse"}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
@lru_cache(maxsize=64)
def b64(filename):
    path = C.ASSET_DIR / filename
    mime = "image/png" if filename.endswith(".png") else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


@st.cache_resource(show_spinner="Loading the AI model (first run can take ~20 seconds)...")
def get_valuer():
    meta = load_metrics()
    valuer = Valuer(ensure_model(), meta)
    return valuer, load_metrics()


@st.cache_data(show_spinner=False)
def load_data():
    if not C.DATA_FILE.exists():
        from src.data_generator import generate
        C.DATA_DIR.mkdir(exist_ok=True)
        generate().to_csv(C.DATA_FILE, index=False)
    return pd.read_csv(C.DATA_FILE)


@st.cache_data(show_spinner=False)
def load_holdout():
    return pd.read_csv(C.HOLDOUT_FILE) if C.HOLDOUT_FILE.exists() else None


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


CSS = """
<style>
/* Make form labels dark and clearly visible */
.stNumberInput label,
.stSelectbox label,
.stTextInput label,
.stSlider label,
.stRadio label,
.stMultiSelect label {
    color: #0B1F3A !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    opacity: 1 !important;
}
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1280px; }
h1, h2, h3, h4 { font-family: 'Poppins', 'Segoe UI', sans-serif !important; color: #0B1F3A; letter-spacing: -.01em; }
#MainMenu, footer { visibility: hidden; }

.hero { border-radius: 22px; padding: 2.3rem 2.6rem 2.1rem; color: #fff; background-size: cover;
        background-position: center bottom; box-shadow: 0 18px 40px rgba(11,31,58,.28); margin-bottom: 1.3rem; }
.hero-badge { display:inline-block; font-size:.72rem; letter-spacing:.14em; font-weight:600; padding:.35rem .8rem;
        border-radius:999px; background:rgba(245,158,11,.18); color:#FBBF24; border:1px solid rgba(245,158,11,.5); }
.hero-title { font-family:'Poppins',sans-serif; font-size:2.35rem; font-weight:700; line-height:1.18; margin:.8rem 0 .5rem; max-width:680px; }
.hero-title span { color:#FBBF24; }
.hero-sub { font-size:1.02rem; color:#D5E1EE; max-width:580px; margin-bottom:1.2rem; }
.hero-stats { display:flex; gap:2.2rem; flex-wrap:wrap; }
.hero-stats div { display:flex; flex-direction:column; }
.hero-stats b { font-family:'Poppins',sans-serif; font-size:1.55rem; color:#fff; }
.hero-stats span { font-size:.78rem; color:#AFC3D8; text-transform:uppercase; letter-spacing:.08em; }

.stepbar { display:flex; align-items:center; gap:.7rem; margin:.9rem 0 .35rem; }
.stepbar .n { width:30px; height:30px; border-radius:50%; background:linear-gradient(135deg,#0E7C86,#0B1F3A); color:#fff;
        display:flex; align-items:center; justify-content:center; font-weight:700; font-size:.9rem; flex:none; }
.stepbar .t { font-family:'Poppins',sans-serif; font-weight:600; color:#0B1F3A; font-size:1.05rem; }
.stepbar .d { font-size:.8rem; color:#627D98; margin-left:.2rem; }

[data-testid="stForm"] { border:1px solid #E3EAF1; border-radius:18px; padding:1.1rem 1.3rem 1.3rem; background:#FBFDFE;
        box-shadow:0 6px 20px rgba(11,31,58,.06); }
[data-testid="stForm"] button[kind="secondaryFormSubmit"], [data-testid="stForm"] button[kind="primaryFormSubmit"] {
        background:linear-gradient(135deg,#0E7C86,#0B5E73); color:#fff; border:none; border-radius:12px; font-weight:600;
        padding:.65rem 1rem; font-size:1.05rem; box-shadow:0 8px 18px rgba(14,124,134,.35); }

.result-card { border-radius:20px; padding:1.5rem 1.7rem; color:#fff; box-shadow:0 14px 30px rgba(14,124,134,.30);
        background: linear-gradient(135deg, #0B1F3A 0%, #0E5C6B 55%, #0E7C86 100%); }
.rc-label { font-size:.74rem; letter-spacing:.16em; color:#9FD8DD; font-weight:600; }
.rc-price { font-family:'Poppins',sans-serif; font-size:3.1rem; font-weight:700; line-height:1.1; margin:.2rem 0; }
.rc-words { font-size:.9rem; color:#CFE7EA; }
.rc-range { margin-top:.8rem; padding:.7rem 1rem; border-radius:12px; background:rgba(255,255,255,.10); font-size:.95rem; }
.rc-range b { color:#FBBF24; }
.chip { display:inline-block; margin:.7rem .45rem 0 0; padding:.32rem .75rem; border-radius:999px; font-size:.8rem;
        background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.22); }

.waiting { border:2px dashed #BFD3E3; border-radius:20px; padding:1.6rem 1.7rem; background:#F6FAFD; text-align:center; color:#4A6076; }
.waiting .big { font-family:'Poppins',sans-serif; font-size:1.25rem; color:#0B1F3A; font-weight:600; margin-bottom:.3rem; }

.kpi { background:#fff; border:1px solid #E3EAF1; border-radius:16px; padding:1rem 1.1rem; box-shadow:0 4px 14px rgba(11,31,58,.05); height:100%; }
.kpi .l { font-size:.74rem; color:#627D98; text-transform:uppercase; letter-spacing:.08em; font-weight:600; }
.kpi .v { font-family:'Poppins',sans-serif; font-size:1.4rem; color:#0B1F3A; font-weight:600; margin:.15rem 0; }
.kpi .s { font-size:.78rem; color:#627D98; }
.up { color:#0E9F6E !important; font-weight:600; } .down { color:#DC2626 !important; font-weight:600; }

.imgcard { position:relative; border-radius:18px; overflow:hidden; box-shadow:0 8px 22px rgba(11,31,58,.16); }
.imgcard img { width:100%; display:block; }
.imgcard .cap { position:absolute; left:0; right:0; bottom:0; padding:1.6rem 1rem .8rem;
        background:linear-gradient(0deg, rgba(8,20,40,.88), rgba(8,20,40,0)); color:#fff; }
.imgcard .cap b { font-family:'Poppins',sans-serif; font-size:1.1rem; display:block; }
.imgcard .cap span { font-size:.8rem; color:#D5E1EE; }

.step { background:#fff; border:1px solid #E3EAF1; border-radius:16px; padding:1.1rem 1.2rem; height:100%; box-shadow:0 4px 14px rgba(11,31,58,.05); }
.step .n { width:34px; height:34px; border-radius:50%; background:linear-gradient(135deg,#0E7C86,#0B1F3A); color:#fff;
        display:flex; align-items:center; justify-content:center; font-weight:700; margin-bottom:.6rem; }
.step h4 { margin:0 0 .3rem; font-size:1rem; } .step p { margin:0; font-size:.86rem; color:#4A6076; }
.note { background:#FFF8E6; border:1px solid #F8E0A4; border-radius:12px; padding:.75rem 1rem; font-size:.86rem; color:#6B4E0A; }
.sect { font-family:'Poppins',sans-serif; font-size:1.15rem; font-weight:600; color:#0B1F3A; margin:1.2rem 0 .3rem; }
.sect-sub { font-size:.86rem; color:#627D98; margin-bottom:.6rem; }
.footer { text-align:center; color:#7B8FA3; font-size:.82rem; margin-top:2.2rem; padding-top:1rem; border-top:1px solid #E3EAF1; }
button[data-baseweb="tab"] { font-weight:600; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

valuer, meta = get_valuer()
final = meta.get("final", {})


def kpi(label, value, sub="", cls=""):
    return f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div><div class="s {cls}">{sub}</div></div>'


def sect(title, sub=""):
    st.markdown(f'<div class="sect">{title}</div>' + (f'<div class="sect-sub">{sub}</div>' if sub else ""),
                unsafe_allow_html=True)


def stepbar(n, title, desc=""):
    st.markdown(f'<div class="stepbar"><div class="n">{n}</div><div class="t">{title}</div>'
                f'<div class="d">{desc}</div></div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# HERO
# ----------------------------------------------------------------------------
st.markdown(f"""
<div class="hero" style="background-image: linear-gradient(90deg, rgba(8,20,40,.96) 0%, rgba(8,20,40,.78) 40%, rgba(8,20,40,.12) 100%), url('{b64("hero.jpg")}');">
  <span class="hero-badge">MACHINE LEARNING &nbsp;·&nbsp; INDIAN REAL ESTATE</span>
  <div class="hero-title">Tell us the house you want. <span>We'll tell you its price.</span></div>
  <div class="hero-sub">Enter the number of bedrooms, bathrooms, area and a few other details - {C.APP_NAME}'s AI model
  trained on {meta.get('n_train', 0) + meta.get('n_test', 0):,} property records predicts what it should cost.</div>
  <div class="hero-stats">
    <div><b>{meta.get('n_cities', len(C.CITIES))}</b><span>Cities</span></div>
    <div><b>{meta.get('n_localities', 0)}+</b><span>Localities</span></div>
    <div><b>{final.get('r2', 0):.2f}</b><span>R² accuracy</span></div>
    <div><b>{final.get('median_ape_pct', 0):.1f}%</b><span>Median error</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# REQUIREMENTS  (left)   |   PREDICTED PRICE  (right)
# ----------------------------------------------------------------------------
left, right = st.columns([1.25, 1], gap="large")

with left:
    stepbar(1, "Where do you want to buy?", "city and locality")
    lc1, lc2 = st.columns(2)
    city = lc1.selectbox("City", list(C.CITIES), index=list(C.CITIES).index("Bengaluru"))
    locs = sorted(C.CITIES[city]["localities"], key=C.CITIES[city]["localities"].get, reverse=True)
    locality = lc2.selectbox("Locality / micro-market", locs, index=min(6, len(locs) - 1))
    rate = C.locality_rate(city, locality)
    tier = C.locality_tier(city, locality)
    st.caption(f"{tier} locality · market benchmark ≈ ₹{rate:,.0f} per sq ft")

    with st.form("requirements"):
        stepbar(2, "Rooms & size", "what you need inside")
        f1, f2 = st.columns(2)
        ptype = f1.selectbox("Property type", C.PROPERTY_TYPES)
        area = f2.number_input("Built-up area (sq ft)", min_value=300, max_value=9000, value=1200, step=50)
        f3, f4, f5, f6 = st.columns(4)
        bhk = f3.selectbox("Bedrooms (BHK)", [1, 2, 3, 4, 5, 6, 7], index=2)
        baths = f4.selectbox("Bathrooms", [1, 2, 3, 4, 5, 6, 7, 8], index=2)
        balc = f5.selectbox("Balconies", [0, 1, 2, 3, 4], index=1)
        parking = f6.selectbox("Car parks", [0, 1, 2, 3, 4], index=1)

        stepbar(3, "Building & condition", "floor, age and furnishing")
        g1, g2, g3 = st.columns(3)
        total_floors = g1.number_input("Floors in building (storeys for a house)", min_value=1, max_value=60, value=14)
        floor = g2.number_input("Your floor (0 = ground)", min_value=0, max_value=60, value=6)
        furnishing = g3.selectbox("Furnishing", C.FURNISHING)
        g4, g5, g6 = st.columns(3)
        status = g4.selectbox("Construction status", C.CONSTRUCTION)
        age = g5.number_input("Age of property (years)", min_value=0, max_value=40, value=4)
        rera = g6.selectbox("RERA registered?", ["Yes", "No"])
        h1, h2, h3 = st.columns(3)
        facing = h1.selectbox("Facing direction", C.FACINGS)
        near_social = h2.selectbox("Schools / hospitals within 2 km?", ["Yes", "No"])
        kitchen = h3.selectbox("Modular kitchen?", ["No", "Yes"])
        metro_km = st.slider("Distance to nearest metro / railway / transit hub (km)", 0.1, 20.0, 2.0, 0.1)

        stepbar(4, "Amenities", "pick everything the society offers")
        amenities = st.multiselect("Amenities", list(C.AMENITIES),
                                   default=["Lift", "Power Backup", "24x7 Security / CCTV"],
                                   label_visibility="collapsed")
        submitted = st.form_submit_button("🔍  Predict Price", type="primary")

if submitted:
    tf, fl = int(total_floors), int(floor)
    if ptype == "Builder Floor":
        tf = min(tf, 5)
    if ptype in C.LAND_TYPES:
        tf, fl = min(tf, 4), 0
    elif ptype == "Penthouse":
        tf = max(tf, 8)
        fl = tf
    else:
        fl = min(fl, tf)
    if status == "Under Construction":
        age = 0
    st.session_state["spec"] = dict(
        city=city, locality=locality, property_type=ptype, area_sqft=int(area), bhk=int(bhk),
        bathrooms=int(baths), balconies=int(balc), floor=fl, total_floors=tf, age_years=int(age),
        furnishing=furnishing, construction_status=status, parking=int(parking), metro_km=float(metro_km),
        rera=(rera == "Yes"), amenities=list(amenities), facing=facing,
        near_social=(near_social == "Yes"), modular_kitchen=(kitchen == "Yes"))
    # keep a log of every prediction made in this session
    st.session_state.setdefault("history", [])
    _r = valuer.predict(st.session_state["spec"])
    st.session_state["history"].append({
        "Time": datetime.now().strftime("%H:%M:%S"), "City": city, "Locality": locality, "Type": ptype,
        "BHK": int(bhk), "Baths": int(baths), "Area (sq ft)": int(area),
        "Predicted price": inr(_r["price"]), "Price (Rs)": int(_r["price"])})

spec = st.session_state.get("spec")

with right:
    if spec is None:
        st.markdown(f"""
        <div class="waiting"><div class="big">Your price will appear here</div>
        Fill in your requirements on the left and press <b>Predict Price</b>.</div>""", unsafe_allow_html=True)
        st.write("")
        st.markdown(f'<div class="imgcard"><img src="{b64("type_apartment.jpg")}"><div class="cap"><b>Find your number</b>'
                    f'<span>{len(C.CITIES)} cities · {meta.get("n_localities", 0)}+ localities</span></div></div>',
                    unsafe_allow_html=True)
    else:
        result = valuer.predict(spec)
        price = result["price"]
        s_city, s_loc, s_type = spec["city"], spec["locality"], spec["property_type"]
        s_cfg = C.CITIES[s_city]
        s_tier = C.locality_tier(s_city, s_loc)
        s_rate = C.locality_rate(s_city, s_loc)
        st.markdown(f"""
        <div class="result-card">
          <div class="rc-label">PREDICTED PRICE OF YOUR HOUSE</div>
          <div class="rc-price">{inr(price)}</div>
          <div class="rc-words">≈ {words_in_indian(price)} rupees &nbsp;·&nbsp; ₹{price:,.0f}</div>
          <div class="rc-range">Likely range (80% confidence): <b>{inr(result['low'])}</b> &nbsp;to&nbsp; <b>{inr(result['high'])}</b></div>
          <span class="chip">{spec['bhk']} BHK {s_type}</span><span class="chip">{spec['bathrooms']} bath</span>
          <span class="chip">{spec['area_sqft']:,} sq ft</span><span class="chip">{s_loc}, {s_city}</span>
        </div>""", unsafe_allow_html=True)

        warnings = []
        per_bhk = spec["area_sqft"] / max(spec["bhk"], 1)
        if per_bhk < 250:
            warnings.append(f"{spec['area_sqft']:,} sq ft is very small for {spec['bhk']} BHK - please double-check.")
        elif per_bhk > 1500:
            warnings.append(f"{spec['area_sqft']:,} sq ft is very large for {spec['bhk']} BHK - please double-check.")
        share = {"Independent House": s_cfg["house"], "Villa": s_cfg["villa"], "Builder Floor": s_cfg["bf"]}.get(s_type, 1)
        if share < 0.04:
            warnings.append(f"{s_type}s are rare in {s_city}, so this estimate is less reliable than usual.")
        if spec["bathrooms"] > spec["bhk"] + 2:
            warnings.append("That is a lot of bathrooms for the number of bedrooms.")
        for w in warnings:
            st.warning(w)

        st.write("")
        i1, i2 = st.columns(2)
        i1.markdown(f'<div class="imgcard"><img src="{b64(TYPE_IMG[s_type] + ".jpg")}"><div class="cap"><b>{s_type}</b>'
                    f'<span>{spec["bhk"]} BHK · {spec["area_sqft"]:,} sq ft</span></div></div>', unsafe_allow_html=True)
        i2.markdown(f'<div class="imgcard"><img src="{b64("city_" + s_city.lower() + ".jpg")}"><div class="cap"><b>{s_city}</b>'
                    f'<span>{s_cfg["tagline"]}</span></div></div>', unsafe_allow_html=True)

        emi_est = fin.emi(price * 0.8, 8.5, 20)
        costs = fin.buying_costs(price, s_city, spec["construction_status"] == "Under Construction")
        cls_dir = "up" if result["vs_benchmark_pct"] >= 0 else "down"
        st.write("")
        k1, k2 = st.columns(2)
        k1.markdown(kpi("Price per sq ft", f"₹{result['psf']:,.0f}", f"{result['vs_benchmark_pct']:+.1f}% vs locality benchmark", cls_dir), unsafe_allow_html=True)
        k2.markdown(kpi("Indicative EMI", f"₹{emi_est:,.0f}", "80% loan · 8.5% · 20 yrs"), unsafe_allow_html=True)
        st.write("")
        k3, k4 = st.columns(2)
        k3.markdown(kpi("Locality benchmark", f"₹{s_rate:,.0f}", f"{s_loc} · {s_tier}"), unsafe_allow_html=True)
        k4.markdown(kpi("All-in buying cost", inr(costs["grand_total"]), f"incl. ~{costs['stamp_pct']:.1f}% stamp duty & fees"), unsafe_allow_html=True)

        report = f"""{C.APP_NAME} - PROPERTY PRICE REPORT
{'=' * 48}
Location   : {s_loc}, {s_city} ({s_cfg['state']})
Requirement: {spec['bhk']} BHK {s_type}, {spec['area_sqft']:,} sq ft, {spec['bathrooms']} bathrooms, {spec['balconies']} balconies
Status     : {spec['construction_status']}, age {spec['age_years']} yrs, {spec['furnishing']}
Floor      : {spec['floor']} of {spec['total_floors']} | Parking: {spec['parking']} | Metro/transit: {spec['metro_km']} km | RERA: {'Yes' if spec['rera'] else 'No'}
Extras     : Facing {spec['facing']} | Schools/hospitals nearby: {'Yes' if spec['near_social'] else 'No'} | Modular kitchen: {'Yes' if spec['modular_kitchen'] else 'No'}
Amenities  : {', '.join(spec['amenities']) if spec['amenities'] else 'None'}

PREDICTED PRICE : Rs {price:,.0f}  ({inr(price)})
Likely range    : {inr(result['low'])} to {inr(result['high'])}
Price per sq ft : Rs {result['psf']:,.0f}  (locality benchmark Rs {s_rate:,.0f})
Indicative EMI  : Rs {emi_est:,.0f} / month (80% loan, 8.5%, 20 yrs)
All-in cost     : {inr(costs['grand_total'])}

Model: Gradient Boosting, R2 = {final.get('r2', 0):.3f}, median error = {final.get('median_ape_pct', 0):.1f}%
Disclaimer: AI estimate for educational use. Not a certified valuation.
"""
        st.download_button("⬇️  Download price report (.txt)", report, file_name="house_price_report.txt")

# ----------------------------------------------------------------------------
# DEEPER ANALYSIS TABS
# ----------------------------------------------------------------------------
st.write("")
tab_why, tab_budget, tab_cmp, tab_ins, tab_emi, tab_hist, tab_about = st.tabs(
    ["💡  Why this price?", "💼  Budget Finder", "🏙️  Compare Cities", "📊  Market & Data",
     "🧮  EMI & Buying Cost", "🕘  My Predictions", "🧠  About the Model"])


def need_prediction():
    st.info("Enter your requirements above and press **Predict Price** to unlock this section.")


# =============================== WHY =========================================
with tab_why:
    if spec is None:
        need_prediction()
    else:
        sect("What moves the price?", "Each bar shows how much a feature adds to (or removes from) your price, compared with a neutral reference.")
        w1, w2 = st.columns(2)
        with w1:
            show(ch.drivers_chart(valuer.value_drivers(spec)))
        with w2:
            st.markdown("**Price vs. size** - the same house if it were smaller or larger")
            areas, prices = valuer.area_curve(spec)
            show(ch.area_curve_chart(areas, prices, spec["area_sqft"], price))
        st.markdown('<div class="note">This is a machine-learning estimate for learning and planning. Actual prices depend on '
                    'negotiation, the exact project and market timing - use a registered valuer for loans or legal use.</div>',
                    unsafe_allow_html=True)

# ============================== BUDGET FINDER ================================
with tab_budget:
    if spec is None:
        need_prediction()
    else:
        sect("Work backwards from your budget", "Tell us how much you can spend - we show what size of home and which localities fit.")
        budget = st.number_input("Your budget (₹)", min_value=1000000, max_value=2000000000,
                                 value=int(round(price / 100000) * 100000), step=100000)
        st.caption(f"Budget: {inr(budget)}  ·  your requirement is priced at {inr(price)}")
        diff = budget - price
        b1, b2, b3 = st.columns(3)
        b1.markdown(kpi("Your requirement costs", inr(price), "as entered above"), unsafe_allow_html=True)
        b2.markdown(kpi("Budget vs. price", inr(abs(diff)), "within budget ✔" if diff >= 0 else "over budget ✖",
                        "up" if diff >= 0 else "down"), unsafe_allow_html=True)
        max_a = valuer.max_area_for_budget(spec, budget)
        b3.markdown(kpi("Largest area you can afford", f"{max_a:,.0f} sq ft" if max_a else "Not possible",
                        f"same {spec['property_type'].lower()} in {spec['locality']}"), unsafe_allow_html=True)

        bb1, bb2 = st.columns([1, 1.25])
        with bb1:
            st.markdown(f"**What can ₹{budget / 1e7:.2f} Cr buy in {spec['locality']}?**")
            tbl = valuer.bhk_budget_table(spec, budget)
            tbl["Max area (sq ft)"] = tbl["Max area (sq ft)"].map(lambda v: "-" if pd.isna(v) else f"{v:,.0f}")
            tbl["Sq ft per bedroom"] = tbl["Sq ft per bedroom"].map(lambda v: "-" if pd.isna(v) else f"{v:,.0f}")
            tbl["Realistic"] = tbl["Realistic"].map(lambda v: "✔" if v else "✖ too cramped")
            st.dataframe(tbl, hide_index=True)
            st.caption("'Realistic' means at least ~330 sq ft per bedroom. Other requirements stay as entered.")
        with bb2:
            st.markdown(f"**Where in {spec['city']} does your requirement fit the budget?**")
            ldf = valuer.localities_in_city(spec)
            show(ch.budget_locality_chart(ldf, budget))
            fit = ldf[ldf["Estimated price"] <= budget]
            st.caption(f"{len(fit)} of {len(ldf)} localities fit your budget (green bars).")

# ============================== COMPARE CITIES ===============================
with tab_cmp:
    if spec is None:
        need_prediction()
    else:
        sect("Where does your budget go further?", "Your exact requirement priced in a mid-range locality of every city.")
        cdf = valuer.compare_cities(spec)
        cheapest, dearest = cdf.iloc[-1], cdf.iloc[0]
        a1, a2, a3 = st.columns(3)
        a1.markdown(kpi("Most expensive", dearest["City"], inr(dearest["Estimated price"])), unsafe_allow_html=True)
        a2.markdown(kpi("Most affordable", cheapest["City"], inr(cheapest["Estimated price"])), unsafe_allow_html=True)
        a3.markdown(kpi("Price gap", f"{dearest['Estimated price'] / cheapest['Estimated price']:.1f}×",
                        f"{dearest['City']} vs {cheapest['City']}"), unsafe_allow_html=True)
        st.write("")
        cc1, cc2 = st.columns([1.3, 1])
        with cc1:
            show(ch.city_compare_chart(cdf, spec["city"]))
        with cc2:
            table = pd.DataFrame({"City": cdf["City"], "Reference locality": cdf["Reference locality"],
                                  "Estimated price": cdf["Estimated price"].map(inr),
                                  "₹ / sq ft": cdf["Price per sq ft"].map(lambda v: f"{v:,.0f}")})
            st.dataframe(table, hide_index=True, height=520)

# ============================== INSIGHTS =====================================
with tab_ins:
    data = load_data()
    sel_city = spec["city"] if spec else city
    sel_loc = spec["locality"] if spec else locality
    sect("Market rates across India", "Median benchmark rate of each city's localities (₹ per sq ft).")
    m1, m2 = st.columns(2)
    with m1:
        show(ch.city_rates_chart(sel_city))
    with m2:
        st.markdown(f"**Localities in {sel_city}**")
        show(ch.locality_chart(sel_city, sel_loc))
    if spec:
        sect(f"What {sel_city} homes cost", "From the training data - your predicted price is marked in gold.")
        d_city = data[data["city"] == sel_city]
        n1, n2 = st.columns(2)
        with n1:
            st.markdown("**Price distribution**")
            show(ch.distribution_chart(d_city[C.TARGET].values, price))
        with n2:
            st.markdown("**Price vs. area**")
            samp = d_city.sample(min(1800, len(d_city)), random_state=1)
            show(ch.scatter_chart(samp["area_sqft"].values, samp[C.TARGET].values, spec["area_sqft"], price))
    sect("Explore the dataset", "How the training data behaves - the kind of analysis done before building a model.")
    ex_city = st.selectbox("Filter by city", ["All cities"] + list(C.CITIES), key="ex_city")
    ex = data if ex_city == "All cities" else data[data["city"] == ex_city]
    x1, x2, x3 = st.columns(3)
    x1.markdown(kpi("Records", f"{len(ex):,}", ex_city), unsafe_allow_html=True)
    x2.markdown(kpi("Median price", inr(ex[C.TARGET].median()), "in this selection"), unsafe_allow_html=True)
    x3.markdown(kpi("Median ₹ / sq ft", f"₹{(ex[C.TARGET] / ex['area_sqft']).median():,.0f}", "price ÷ area"), unsafe_allow_html=True)
    st.write("")
    y1, y2 = st.columns(2)
    with y1:
        st.markdown("**Correlation heatmap** - how numeric features move with price")
        show(ch.correlation_chart(ex))
    with y2:
        st.markdown("**Median ₹ per sq ft by property type**")
        show(ch.psf_by_type_chart(ex))
        st.markdown("**Price by number of bedrooms**")
        show(ch.price_by_bhk_chart(ex))
    st.markdown("**Sample of the dataset**")
    st.dataframe(ex.drop(columns=["locality_rate"]).head(200), height=300)

    sect("Explore the cities", "Illustrated city cards with each market's median benchmark rate.")
    names = list(C.CITIES)
    for row in range(0, len(names), 4):
        cols = st.columns(4)
        for col, nm in zip(cols, names[row:row + 4]):
            col.image(str(C.ASSET_DIR / f"city_{nm.lower()}.jpg"), caption=f"{nm} · ₹{C.city_median_rate(nm):,.0f}/sq ft")

# ============================== EMI ==========================================
with tab_emi:
    if spec is None:
        need_prediction()
    else:
        sect("Home-loan EMI calculator", "Starts from your predicted price - change anything to explore scenarios.")
        e1, e2 = st.columns([1, 1.25])
        with e1:
            prop_price = st.number_input("Property price (₹)", min_value=500000, max_value=2000000000,
                                         value=int(round(price / 100000) * 100000), step=100000)
            down_pct = st.slider("Down payment (%)", 10, 60, 20)
            rate_pct = st.slider("Interest rate (% p.a.)", 6.5, 12.0, 8.5, 0.05)
            years = st.slider("Loan tenure (years)", 5, 30, 20)
            loan = prop_price * (1 - down_pct / 100)
            emi_v = fin.emi(loan, rate_pct, years)
            interest = emi_v * years * 12 - loan
            need = fin.required_income(emi_v)
        with e2:
            q1, q2 = st.columns(2)
            q1.markdown(kpi("Monthly EMI", f"₹{emi_v:,.0f}", f"on a loan of {inr(loan)}"), unsafe_allow_html=True)
            q2.markdown(kpi("Total interest", inr(interest), f"over {years} years"), unsafe_allow_html=True)
            st.write("")
            q3, q4 = st.columns(2)
            q3.markdown(kpi("Down payment", inr(prop_price * down_pct / 100), f"{down_pct}% of price"), unsafe_allow_html=True)
            q4.markdown(kpi("Income needed", f"₹{need:,.0f}/mo", "if EMI ≤ 45% of income"), unsafe_allow_html=True)
        d1, d2 = st.columns([1, 1.6])
        with d1:
            st.markdown("**Principal vs. interest**")
            show(ch.emi_donut(loan, interest))
        with d2:
            st.markdown("**Year-by-year repayment**")
            show(ch.amortization_chart(fin.amortization_by_year(loan, rate_pct, years)))
        sect("One-time buying costs", f"Indicative costs for buying in {spec['city']} ({C.CITIES[spec['city']]['state']}).")
        bc = fin.buying_costs(prop_price, spec["city"], spec["construction_status"] == "Under Construction")
        rows = [("Property price", prop_price),
                (f"Stamp duty & registration (~{bc['stamp_pct']:.1f}%)", bc["stamp_duty_registration"]),
                ("GST on under-construction (5%)", bc["gst"]), ("Brokerage (1%)", bc["brokerage"]),
                ("Legal, society & misc.", bc["legal_misc"]), ("Total cash needed overall", bc["grand_total"])]
        st.dataframe(pd.DataFrame({"Item": [r[0] for r in rows], "Amount": [inr(r[1], short=False) for r in rows]}),
                     hide_index=True)
        st.markdown('<div class="note">Stamp-duty rates differ by state, buyer and property value and change often. '
                    'Treat these as rough planning numbers.</div>', unsafe_allow_html=True)

# ============================== HISTORY ======================================
with tab_hist:
    sect("Your predictions this session", "Every time you press Predict Price, the result is saved here so you can compare options.")
    hist = st.session_state.get("history", [])
    if not hist:
        st.info("No predictions yet. Enter your requirements above and press **Predict Price**.")
    else:
        hdf = pd.DataFrame(hist)
        st.dataframe(hdf.drop(columns=["Price (Rs)"]).iloc[::-1], hide_index=True)
        if len(hdf) > 1:
            st.markdown("**Price of each prediction**")
            figh, axh = ch._fig(6.8, 3.0)
            axh.bar(range(1, len(hdf) + 1), hdf["Price (Rs)"], color=ch.TEAL, width=.6)
            axh.set_xticks(range(1, len(hdf) + 1))
            axh.set_xlabel("Prediction number")
            axh.yaxis.set_major_formatter(ch._money_fmt(hdf["Price (Rs)"].max()))
            axh.grid(axis="y", color=ch.GRID)
            axh.set_axisbelow(True)
            figh.tight_layout()
            show(figh)
        st.download_button("⬇️  Download history (.csv)", hdf.drop(columns=["Price (Rs)"]).to_csv(index=False),
                           file_name="my_predictions.csv")

# ============================== ABOUT ========================================
with tab_about:
    sect("How it works")
    s1, s2, s3, s4 = st.columns(4)
    for col, (n, t, d) in zip([s1, s2, s3, s4], [
        ("1", "Market data", f"{meta.get('n_train', 0) + meta.get('n_test', 0):,} records across {meta.get('n_cities', 0)} cities and {meta.get('n_localities', 0)} localities, built from Indian price-per-sq-ft benchmarks."),
        ("2", "Your requirements", "Bedrooms, bathrooms, area, floor, age, furnishing, facing, amenities, metro distance and more - 27 inputs."),
        ("3", "Model training", "Gradient Boosting learns price on a log scale, with rules such as 'more area never means a lower price'."),
        ("4", "Prediction", "Your requirements go through the trained model; the 80% range comes from its errors on unseen data."),
    ]):
        col.markdown(f'<div class="step"><div class="n">{n}</div><h4>{t}</h4><p>{d}</p></div>', unsafe_allow_html=True)

    sect("Accuracy on unseen data", f"Tested on {meta.get('n_test', 0):,} properties the model never saw during training.")
    t1, t2, t3, t4 = st.columns(4)
    t1.markdown(kpi("R² score", f"{final.get('r2', 0):.3f}", "1.0 = perfect"), unsafe_allow_html=True)
    t2.markdown(kpi("Median error", f"{final.get('median_ape_pct', 0):.1f}%", "typical miss"), unsafe_allow_html=True)
    t3.markdown(kpi("Within ±10%", f"{final.get('within_10_pct', 0):.0f}%", "of predictions"), unsafe_allow_html=True)
    t4.markdown(kpi("Within ±15%", f"{final.get('within_15_pct', 0):.0f}%", "of predictions"), unsafe_allow_html=True)
    st.write("")
    u1, u2 = st.columns(2)
    with u1:
        st.markdown("**Model comparison**")
        if meta.get("comparison"):
            show(ch.model_compare_chart(meta["comparison"]))
        st.markdown("**What the model pays attention to**")
        if meta.get("importance"):
            show(ch.importance_chart(meta["importance"]))
    with u2:
        st.markdown("**Actual vs. predicted price**")
        hold = load_holdout()
        if hold is not None:
            show(ch.actual_vs_pred_chart(hold))
        st.caption("Each dot is an unseen property. Dots on the gold line are perfect predictions.")
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**How far off are the predictions?**")
        if hold is not None:
            show(ch.residual_chart(hold))
        st.caption("Errors are centred on zero - the model is not systematically too high or too low.")
    with r2:
        st.markdown("**Average error by city**")
        if meta.get("mape_by_city"):
            show(ch.mape_by_city_chart(meta["mape_by_city"]))

    sect("Technology & honest limitations")
    l1, l2 = st.columns(2)
    l1.markdown("""
- **Python**, **pandas / NumPy** for data
- **scikit-learn** for the model pipeline
- **Streamlit** for the web interface
- **Matplotlib** for charts, **Pillow** for artwork
""")
    l2.markdown("""
- Training data is **simulated** from real market benchmarks (e.g. Anarock's Q3 2026 report: top-7-city average ≈ ₹9,700/sq ft) - not scraped listings.
- Locality rates are approximate and change quarterly; update them in `src/config.py` and re-train.
- Estimates ignore view, builder brand, exact floor plan and negotiation.
""")

st.markdown(f"""
<div class="footer">
  <b>{C.PROJECT_TITLE}</b><br>
  Developed by <b>{C.STUDENT_NAME}</b> · {C.COLLEGE_NAME} · {C.ACADEMIC_YEAR}
</div>""", unsafe_allow_html=True)
