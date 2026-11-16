"""
Loading the trained model and turning a user's property description into a price.
"""
import json

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import config as C


# --------------------------------------------------------------------------
# Model loading (auto re-trains if the file is missing / made by another sklearn)
# --------------------------------------------------------------------------
def load_metrics():
    if C.METRICS_FILE.exists():
        return json.loads(C.METRICS_FILE.read_text())
    return {}


def ensure_model():
    """Return a fitted pipeline.  Re-trains quickly if needed."""
    meta = load_metrics()
    ok = C.MODEL_FILE.exists() and meta.get("sklearn_version") == sklearn.__version__
    if ok:
        try:
            return joblib.load(C.MODEL_FILE)
        except Exception:
            ok = False
    from .train_model import train          # imported lazily (slow import)
    model, _ = train(quick=True, verbose=False)
    return model


# --------------------------------------------------------------------------
# Spec -> DataFrame
# --------------------------------------------------------------------------
def spec_to_frame(spec_list):
    """spec_list: list of dicts with the UI-level keys (see app.py)."""
    rows = []
    for s in spec_list:
        row = {
            "city": s["city"],
            "property_type": s["property_type"],
            "furnishing": s["furnishing"],
            "construction_status": s["construction_status"],
            "locality_rate": float(C.locality_rate(s["city"], s["locality"])),
            "area_sqft": float(s["area_sqft"]),
            "bhk": int(s["bhk"]),
            "bathrooms": int(s["bathrooms"]),
            "balconies": int(s["balconies"]),
            "floor": int(s["floor"]),
            "total_floors": int(s["total_floors"]),
            "age_years": int(s["age_years"]),
            "parking": int(s["parking"]),
            "metro_km": float(s["metro_km"]),
            "rera": int(bool(s["rera"])),
            "facing": s.get("facing", "East"),
            "near_social": int(bool(s.get("near_social", True))),
            "modular_kitchen": int(bool(s.get("modular_kitchen", False))),
        }
        chosen = set(s.get("amenities", []))
        for label, (col, _) in C.AMENITIES.items():
            row[col] = int(label in chosen)
        rows.append(row)
    return pd.DataFrame(rows)[C.FEATURES]


class Valuer:
    """Small wrapper: price + 80% confidence band + helpful derived numbers."""

    def __init__(self, pipeline, meta):
        self.pipe = pipeline
        lo, hi = meta.get("interval_log", [-0.14, 0.14])
        self.lo_mult, self.hi_mult = float(np.exp(lo)), float(np.exp(hi))

    def predict_many(self, specs):
        return self.pipe.predict(spec_to_frame(specs))

    def predict(self, spec):
        price = float(self.predict_many([spec])[0])
        rate = C.locality_rate(spec["city"], spec["locality"])
        return {
            "price": price,
            "low": price * self.lo_mult,
            "high": price * self.hi_mult,
            "psf": price / spec["area_sqft"],
            "benchmark": rate,
            "vs_benchmark_pct": (price / spec["area_sqft"] / rate - 1) * 100,
        }

    # ----- what-if analysis ---------------------------------------------
    def value_drivers(self, spec):
        """
        How much does each choice add (or remove) compared with a neutral
        reference?  Each driver is switched to a neutral value one at a time
        and the price change is measured with the model.
        Returns list of (label, delta_rupees) sorted by size.
        """
        base = float(self.predict_many([spec])[0])
        tests = {}

        def variant(**kw):
            s = dict(spec)
            s.update(kw)
            return s

        tests["Furnishing"] = variant(furnishing="Unfurnished")
        tests["Amenities"] = variant(amenities=[a for a in spec.get("amenities", []) if a == "Lift"])
        tests["Metro / transit access"] = variant(metro_km=4.0)
        tests["RERA registration"] = variant(rera=False)
        tests["Car parking"] = variant(parking=0)
        if spec["construction_status"] == "Ready to Move":
            tests["Property age (vs 10-yr-old)"] = variant(age_years=10)
        if spec["property_type"] in C.FLOOR_TYPES:
            tests["Floor position (vs 2nd floor)"] = variant(floor=min(2, spec["total_floors"]))
        tests["Balconies (vs none)"] = variant(balconies=0)
        tests["Modular kitchen"] = variant(modular_kitchen=False)
        tests["Schools / hospitals nearby"] = variant(near_social=False)
        tests["Facing (vs South)"] = variant(facing="South")

        out = []
        for label, s in tests.items():
            d = base - float(self.predict_many([s])[0])
            if abs(d) >= base * 0.0005:          # hide effects below 0.05%
                out.append((label, d))
        out.sort(key=lambda t: abs(t[1]), reverse=True)
        return out

    def area_curve(self, spec, lo=0.5, hi=2.0, n=24):
        areas = np.linspace(spec["area_sqft"] * lo, spec["area_sqft"] * hi, n)
        areas = np.maximum(areas, 300)
        specs = []
        for a in areas:
            s = dict(spec)
            s["area_sqft"] = float(a)
            specs.append(s)
        return areas, self.predict_many(specs)

    # ----- budget finder (works backwards from a budget) ----------------
    def max_area_for_budget(self, spec, budget, lo=300.0, hi=9000.0):
        """Largest area (sq ft) whose predicted price still fits the budget.
        The model is monotonic in area, so a simple binary search works."""
        def price_at(a):
            s = dict(spec)
            s["area_sqft"] = float(a)
            return float(self.predict_many([s])[0])

        if price_at(lo) > budget:
            return None
        if price_at(hi) <= budget:
            return hi
        for _ in range(22):
            mid = (lo + hi) / 2
            if price_at(mid) <= budget:
                lo = mid
            else:
                hi = mid
        return lo

    def bhk_budget_table(self, spec, budget, bhks=(1, 2, 3, 4, 5)):
        rows = []
        for b in bhks:
            s = dict(spec)
            s["bhk"], s["bathrooms"] = b, b
            area = self.max_area_for_budget(s, budget)
            if area is None:
                rows.append({"Bedrooms": f"{b} BHK", "Max area (sq ft)": None, "Sq ft per bedroom": None, "Realistic": False})
            else:
                rows.append({"Bedrooms": f"{b} BHK", "Max area (sq ft)": round(area / 10) * 10,
                             "Sq ft per bedroom": round(area / b), "Realistic": area / b >= 330})
        return pd.DataFrame(rows)

    def localities_in_city(self, spec):
        """Predicted price of the same requirement in every locality of the chosen city."""
        locs = list(C.CITIES[spec["city"]]["localities"])
        specs = []
        for loc in locs:
            s = dict(spec)
            s["locality"] = loc
            specs.append(s)
        prices = self.predict_many(specs)
        df = pd.DataFrame({"Locality": locs, "Estimated price": prices})
        df["Price per sq ft"] = df["Estimated price"] / spec["area_sqft"]
        return df.sort_values("Estimated price").reset_index(drop=True)

    def compare_cities(self, spec):
        """Same property in the mid-priced locality of every city."""
        rows = []
        for city, cfg in C.CITIES.items():
            locs = sorted(cfg["localities"], key=cfg["localities"].get)
            loc = locs[len(locs) // 2]
            s = dict(spec)
            s["city"], s["locality"] = city, loc
            rows.append((city, loc, s))
        prices = self.predict_many([r[2] for r in rows])
        df = pd.DataFrame({
            "City": [r[0] for r in rows],
            "Reference locality": [r[1] for r in rows],
            "Estimated price": prices,
        })
        df["Price per sq ft"] = df["Estimated price"] / spec["area_sqft"]
        return df.sort_values("Estimated price", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Formatting helpers (Indian numbering)
# --------------------------------------------------------------------------
def inr(x, short=True):
    """Rs 1.25 Cr / Rs 85.5 L / Rs 45,000"""
    x = float(x)
    sign = "-" if x < 0 else ""
    x = abs(x)
    if short:
        if x >= 1e7:
            return f"{sign}₹{x / 1e7:,.2f} Cr"
        if x >= 1e5:
            return f"{sign}₹{x / 1e5:,.1f} L"
    return f"{sign}₹{indian_grouping(round(x))}"


def indian_grouping(n):
    s = str(int(n))
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts) + "," + tail


def words_in_indian(x):
    """Rs 1.25 Cr -> 'One crore twenty-five lakh' style helper for the report."""
    x = float(x)
    cr, rem = divmod(x, 1e7)
    lk = rem / 1e5
    bits = []
    if cr >= 1:
        bits.append(f"{int(cr)} crore")
    if lk >= 1:
        bits.append(f"{int(round(lk))} lakh")
    return " ".join(bits) if bits else f"{int(x):,} rupees"
