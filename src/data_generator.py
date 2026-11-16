"""
Market simulator that creates the training dataset.

Why synthetic?  A student project can't legally scrape portals, and public
Kaggle sets are small, old and cover few cities.  Instead we encode how the
Indian market actually prices a home (locality benchmark x size x type x
age x floor x amenities x ...) plus random market noise, then let the ML
model *learn* those relationships from 28,000 records.

Run:  python -m src.data_generator
"""
import numpy as np
import pandas as pd

from . import config as C


def _pick_type(rng, city_cfg):
    p_bf, p_house, p_villa, p_pent = city_cfg["bf"], city_cfg["house"], city_cfg["villa"], 0.04
    p_apt = max(0.05, 1 - p_bf - p_house - p_villa - p_pent)
    probs = np.array([p_apt, p_bf, p_house, p_villa, p_pent])
    probs = probs / probs.sum()
    return rng.choice(C.PROPERTY_TYPES, p=probs)


def _area_for_type(rng, ptype):
    if ptype == "Apartment":
        a = rng.lognormal(np.log(1150), 0.38)
        lo, hi = 380, 4200
    elif ptype == "Builder Floor":
        a = rng.lognormal(np.log(1350), 0.30)
        lo, hi = 600, 3000
    elif ptype == "Independent House":
        a = rng.lognormal(np.log(1900), 0.42)
        lo, hi = 600, 6000
    elif ptype == "Villa":
        a = rng.lognormal(np.log(3200), 0.35)
        lo, hi = 1500, 9000
    else:  # Penthouse
        a = rng.lognormal(np.log(3000), 0.30)
        lo, hi = 1500, 7000
    return float(np.clip(a, lo, hi))


def _parking_value(city, rate):
    """Absolute value (Rs) of one covered car park, scaled by locality premium."""
    base = C.CITIES[city]["parking"] * 1e5
    scale = np.clip(rate / C.city_median_rate(city), 0.7, 2.0) ** 0.8
    return base * scale


def market_price(spec, rng=None, noise=0.09):
    """
    Price (Rs) for one property following Indian market logic.
    `spec` is a dict with the same keys as the feature columns.
    """
    rate = spec["locality_rate"]
    ptype = spec["property_type"]

    # --- price per sq ft multipliers --------------------------------------
    f_type = C.TYPE_FACTOR[ptype]
    f_furn = C.FURNISH_FACTOR[spec["furnishing"]]
    f_status = 0.90 if spec["construction_status"] == "Under Construction" else 1.0

    age = spec["age_years"]
    f_age = max(0.72, 1.02 - 0.0125 * age)

    if ptype in C.FLOOR_TYPES:
        fl = spec["floor"]
        f_floor = 0.96 if fl == 0 else 1.0 + 0.004 * min(fl, 20)
    else:
        f_floor = 1.0

    # amenities: sum of uplifts, capped at +12%
    uplift = 0.0
    for label, (col, up) in C.AMENITIES.items():
        if spec.get(col, 0):
            uplift += up
    f_amen = 1.0 + min(uplift, 0.12)
    # a tall building without a lift is heavily discounted
    if ptype in C.FLOOR_TYPES and spec["total_floors"] > 4 and not spec.get("am_lift", 0):
        f_amen *= 0.93

    d = spec["metro_km"]
    if d <= 0.5:
        f_metro = 1.06
    elif d <= 1:
        f_metro = 1.04
    elif d <= 2:
        f_metro = 1.02
    elif d <= 5:
        f_metro = 1.00
    elif d <= 10:
        f_metro = 0.97
    else:
        f_metro = 0.94

    f_rera = 1.02 if spec["rera"] else (0.94 if spec["construction_status"] == "Under Construction" else 0.99)

    # smaller homes sell at a higher rate per sq ft, big ones at a discount
    f_size = float(np.clip((spec["area_sqft"] / 1100.0) ** -0.06, 0.90, 1.12))

    f_bath = float(np.clip(1 + 0.01 * (spec["bathrooms"] - spec["bhk"]), 0.97, 1.03))
    f_balc = 1 + 0.005 * min(spec["balconies"], 3)
    f_facing = C.FACING_FACTOR[spec["facing"]]
    f_social = 1.02 if spec["near_social"] else 1.0
    f_kitchen = 1.01 if spec["modular_kitchen"] else 1.0

    psf = rate * f_type * f_furn * f_status * f_age * f_floor * f_amen * f_metro * f_rera * f_size * f_bath * f_balc * f_facing * f_social * f_kitchen
    value = psf * spec["area_sqft"]

    # separately-priced car parks (flats & floors only; houses/villas include parking)
    if ptype in C.FLOOR_TYPES:
        value += spec["parking"] * _parking_value(spec["city"], rate)

    if rng is not None and noise > 0:
        value *= rng.lognormal(0.0, noise)
    return float(value)


def generate(n=28000, seed=42):
    rng = np.random.default_rng(seed)
    cities = list(C.CITIES)
    w = np.array([C.CITIES[c]["weight"] for c in cities], dtype=float)
    w /= w.sum()

    rows = []
    for _ in range(n):
        city = cities[rng.choice(len(cities), p=w)]
        cfg = C.CITIES[city]
        locs = list(cfg["localities"])
        locality = locs[rng.integers(len(locs))]
        rate = float(cfg["localities"][locality])
        # percentile of this locality inside its city (0 = cheapest, 1 = priciest)
        sorted_rates = sorted(cfg["localities"].values())
        pr = sorted_rates.index(rate) / max(1, len(sorted_rates) - 1)

        # project-level variation around the locality benchmark
        rate_proj = rate * rng.lognormal(0, 0.04)

        ptype = _pick_type(rng, cfg)
        area = round(_area_for_type(rng, ptype) / 10) * 10

        bhk = int(np.clip(round((area - 150) / rng.normal(500, 55)), 1, 7))
        if ptype in ("Villa", "Penthouse"):
            bhk = max(bhk, 3)
        bathrooms = int(np.clip(bhk + rng.choice([-1, 0, 0, 1]), 1, 8))
        balconies = int(np.clip(rng.choice([0, 1, 1, 2, 2, 3]) if ptype in C.FLOOR_TYPES
                                else rng.choice([0, 1, 2, 3]), 0, 4))

        # building geometry ---------------------------------------------------
        if ptype == "Apartment":
            total_floors = int(np.clip(rng.lognormal(2.4 + 0.5 * pr, 0.6), 3, 60))
            floor = int(rng.integers(0, total_floors + 1))
        elif ptype == "Builder Floor":
            total_floors = int(rng.integers(2, 6))
            floor = int(rng.integers(0, total_floors))
        elif ptype == "Penthouse":
            total_floors = int(np.clip(rng.lognormal(2.9, 0.4), 8, 60))
            floor = total_floors - int(rng.integers(0, 2))
        else:  # independent house / villa: total_floors = number of storeys
            total_floors = int(rng.choice([1, 2, 2, 3, 3, 4]))
            floor = 0

        status = "Under Construction" if rng.random() < 0.15 and ptype != "Independent House" else "Ready to Move"
        age = 0 if status == "Under Construction" else int(np.clip(rng.exponential(7.0), 0, 38))
        furnishing = rng.choice(C.FURNISHING, p=[0.45, 0.38, 0.17])

        if ptype in ("Independent House", "Villa"):
            parking = int(rng.integers(1, 5))
        else:
            parking = int(rng.choice([0, 1, 1, 2, 2, 3], p=[0.12, 0.35, 0.25, 0.2, 0.05, 0.03]))

        metro_km = float(np.clip(rng.lognormal(np.log(5.0 - 3.0 * pr), 0.85), 0.1, 20))
        metro_km = round(metro_km, 1)
        rera_p = 0.0 if ptype == "Independent House" else (0.8 if age <= 7 else 0.25)
        rera = int(rng.random() < rera_p)

        # amenities ------------------------------------------------------------
        society = ptype in ("Apartment", "Penthouse") or (ptype == "Villa" and rng.random() < 0.6)
        newer = 0.15 if age < 8 else -0.10
        base = np.clip(0.20 + 0.45 * pr + newer, 0.03, 0.95) if society else 0.04
        am = {}
        am["am_lift"] = int(rng.random() < (0.98 if (total_floors > 3 and society) else (0.35 + 0.4 * pr if ptype == "Builder Floor" else 0.05)))
        am["am_power_backup"] = int(rng.random() < min(0.97, base + 0.35))
        am["am_security"] = int(rng.random() < min(0.97, base + 0.35))
        am["am_gym"] = int(rng.random() < base)
        am["am_pool"] = int(rng.random() < base * 0.55)
        am["am_clubhouse"] = int(rng.random() < base * 0.8)
        am["am_play_area"] = int(rng.random() < base + 0.1 if society else rng.random() < 0.05)
        am["am_garden"] = int(rng.random() < base + 0.1 if society else rng.random() < 0.15)
        am["am_gated"] = int(rng.random() < base + 0.2 if society else rng.random() < 0.05)

        facing = str(rng.choice(C.FACINGS, p=[0.30, 0.30, 0.20, 0.20]))
        near_social = int(rng.random() < 0.55)
        modular_kitchen = int(rng.random() < np.clip(0.20 + 0.45 * pr + (0.15 if age < 8 else -0.05), 0.05, 0.9))

        spec = dict(
            city=city, locality=locality, locality_rate=round(rate_proj, 0),
            property_type=ptype, area_sqft=float(area), bhk=bhk, bathrooms=bathrooms,
            balconies=balconies, floor=floor, total_floors=total_floors, age_years=age,
            furnishing=str(furnishing), construction_status=status, parking=parking,
            metro_km=metro_km, rera=rera, facing=facing, near_social=near_social,
            modular_kitchen=modular_kitchen, **am,
        )
        price = market_price(spec, rng=rng, noise=0.09)
        spec[C.TARGET] = int(round(price / 10000.0) * 10000)
        rows.append(spec)

    df = pd.DataFrame(rows)
    # the model sees the *published* benchmark rate of the locality, not the hidden project variation
    df["locality_rate"] = [C.CITIES[c]["localities"][l] for c, l in zip(df["city"], df["locality"])]
    return df


def main():
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(C.DATA_FILE, index=False)
    df["psf"] = df[C.TARGET] / df["area_sqft"]
    print(f"Saved {len(df):,} rows -> {C.DATA_FILE}")
    print(df.groupby("city")["psf"].median().round(0).sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
