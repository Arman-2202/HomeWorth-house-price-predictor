"""
Home-loan and buying-cost maths used by the EMI tab.
"""
from . import config as C


def emi(principal, annual_rate_pct, years):
    """Standard reducing-balance EMI."""
    r = annual_rate_pct / 12 / 100
    n = int(years * 12)
    if principal <= 0 or n <= 0:
        return 0.0
    if r == 0:
        return principal / n
    f = (1 + r) ** n
    return principal * r * f / (f - 1)


def amortization_by_year(principal, annual_rate_pct, years):
    """List of dicts: year, principal_paid, interest_paid, balance."""
    r = annual_rate_pct / 12 / 100
    n = int(years * 12)
    e = emi(principal, annual_rate_pct, years)
    bal = principal
    rows = []
    for y in range(1, int(years) + 1):
        p_paid = i_paid = 0.0
        for _ in range(12):
            if bal <= 0:
                break
            interest = bal * r
            princ = min(e - interest, bal)
            bal -= princ
            p_paid += princ
            i_paid += interest
        rows.append({"year": y, "principal": p_paid, "interest": i_paid, "balance": max(bal, 0.0)})
    return rows


def buying_costs(price, city, under_construction=False, brokerage_pct=1.0):
    """
    Indicative one-time costs on top of the property price.
    Stamp duty + registration is state specific and changes often -
    treat as an estimate and confirm with the sub-registrar.
    """
    stamp_pct = C.CITIES[city]["stamp"]
    stamp = price * stamp_pct / 100
    gst = price * 0.05 if under_construction else 0.0      # 5% GST on under-construction (non-affordable)
    brokerage = price * brokerage_pct / 100
    legal_misc = max(25000.0, price * 0.003)
    total_extra = stamp + gst + brokerage + legal_misc
    return {
        "stamp_pct": stamp_pct,
        "stamp_duty_registration": stamp,
        "gst": gst,
        "brokerage": brokerage,
        "legal_misc": legal_misc,
        "total_extra": total_extra,
        "grand_total": price + total_extra,
    }


def required_income(emi_amount, foir=0.45):
    """Monthly take-home income a lender usually wants (EMI <= ~45% of income)."""
    return emi_amount / foir if foir > 0 else 0.0
