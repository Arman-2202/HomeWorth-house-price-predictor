"""
Matplotlib charts with one consistent, clean visual style.
Every function returns a Figure; the app displays it with st.pyplot().
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402
from matplotlib.ticker import FuncFormatter          # noqa: E402

from . import config as C                            # noqa: E402
from .predictor import inr                           # noqa: E402

TEAL, NAVY, GOLD = "#0E7C86", "#0B1F3A", "#F59E0B"
GREEN, RED = "#0E9F6E", "#DC2626"
SOFT, GRID, MUTED = "#C9D8E6", "#E6EDF4", "#627D98"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titleweight": "bold", "axes.titlecolor": NAVY,
})


def _fig(w=6.4, h=3.6):
    fig, ax = plt.subplots(figsize=(w, h), dpi=130)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    return fig, ax


def _money_fmt(vmax):
    if vmax >= 1e7:
        div, unit = 1e7, " Cr"
    elif vmax >= 1e5:
        div, unit = 1e5, " L"
    else:
        div, unit = 1.0, ""
    return FuncFormatter(lambda v, _: f"₹{v / div:,.1f}{unit}".replace(".0 ", " ").replace(".0", "") if unit else f"₹{v:,.0f}")


def drivers_chart(drivers):
    fig, ax = _fig(6.4, 3.5)
    if not drivers:
        ax.text(.5, .5, "No notable drivers", ha="center", va="center", color=MUTED)
        ax.axis("off")
        return fig
    labels = [d[0] for d in drivers][::-1]
    vals = [d[1] for d in drivers][::-1]
    cols = [GREEN if v >= 0 else RED for v in vals]
    ax.barh(labels, vals, color=cols, height=.58)
    ax.axvline(0, color=SOFT, lw=1)
    span = max(abs(v) for v in vals) or 1
    for i, v in enumerate(vals):
        ax.text(v + (span * .02 if v >= 0 else -span * .02), i, ("+" if v >= 0 else "-") + inr(abs(v)),
                va="center", ha="left" if v >= 0 else "right", fontsize=9, color=NAVY, fontweight="bold")
    ax.set_xlim(min(0, min(vals)) - span * .35, max(0, max(vals)) + span * .35)
    ax.xaxis.set_major_formatter(_money_fmt(span))
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def area_curve_chart(areas, prices, cur_area, cur_price):
    fig, ax = _fig(6.4, 3.5)
    ax.plot(areas, prices, color=TEAL, lw=2.4)
    ax.fill_between(areas, prices, prices.min() * .92, color=TEAL, alpha=.10)
    ax.scatter([cur_area], [cur_price], s=90, color=GOLD, edgecolor="white", zorder=5, lw=1.5)
    ax.annotate(f"You: {cur_area:,.0f} sq ft\n{inr(cur_price)}", (cur_area, cur_price),
                textcoords="offset points", xytext=(-8, 18), ha="right", fontsize=9,
                color=NAVY, fontweight="bold")
    ax.yaxis.set_major_formatter(_money_fmt(prices.max()))
    ax.set_xlabel("Built-up area (sq ft)")
    ax.grid(color=GRID)
    ax.set_axisbelow(True)
    ax.set_ylim(prices.min() * .92, prices.max() * 1.05)
    fig.tight_layout()
    return fig


def city_compare_chart(df, highlight):
    d = df.sort_values("Estimated price")
    fig, ax = _fig(7.0, 5.2)
    cols = [GOLD if c == highlight else TEAL for c in d["City"]]
    ax.barh(d["City"], d["Estimated price"], color=cols, height=.66)
    vmax = d["Estimated price"].max()
    for i, v in enumerate(d["Estimated price"]):
        ax.text(v + vmax * .012, i, inr(v), va="center", fontsize=9, color=NAVY, fontweight="bold")
    ax.set_xlim(0, vmax * 1.18)
    ax.xaxis.set_major_formatter(_money_fmt(vmax))
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def city_rates_chart(highlight):
    data = sorted(((c, C.city_median_rate(c)) for c in C.CITIES), key=lambda t: t[1])
    fig, ax = _fig(6.4, 5.0)
    ax.barh([d[0] for d in data], [d[1] for d in data],
            color=[GOLD if d[0] == highlight else TEAL for d in data], height=.66)
    vmax = max(d[1] for d in data)
    for i, (_, v) in enumerate(data):
        ax.text(v + vmax * .012, i, f"₹{v:,.0f}", va="center", fontsize=8.5, color=NAVY)
    ax.set_xlim(0, vmax * 1.15)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.set_xlabel("Median locality benchmark (₹ / sq ft)")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def locality_chart(city, highlight):
    items = sorted(C.CITIES[city]["localities"].items(), key=lambda t: t[1])
    fig, ax = _fig(6.4, max(3.2, .33 * len(items) + .8))
    ax.barh([i[0] for i in items], [i[1] for i in items],
            color=[GOLD if i[0] == highlight else TEAL for i in items], height=.66)
    vmax = max(i[1] for i in items)
    for k, (_, v) in enumerate(items):
        ax.text(v + vmax * .012, k, f"₹{v:,.0f}", va="center", fontsize=8.5, color=NAVY)
    ax.set_xlim(0, vmax * 1.15)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.set_xlabel("Benchmark rate (₹ / sq ft)")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def distribution_chart(prices, estimate):
    fig, ax = _fig(6.4, 3.4)
    hi = np.quantile(prices, .97)
    ax.hist(prices[prices <= hi], bins=36, color=TEAL, alpha=.85, edgecolor="white")
    ax.axvline(estimate, color=GOLD, lw=2.6)
    ax.text(estimate, ax.get_ylim()[1] * .96, "  Your estimate", color=NAVY, fontsize=9, fontweight="bold", va="top")
    ax.xaxis.set_major_formatter(_money_fmt(hi))
    ax.set_ylabel("Listings")
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def scatter_chart(areas, prices, cur_area, cur_price):
    fig, ax = _fig(6.4, 3.4)
    ax.scatter(areas, prices, s=9, color=TEAL, alpha=.28, linewidths=0)
    ax.scatter([cur_area], [cur_price], s=150, marker="*", color=GOLD, edgecolor=NAVY, lw=1, zorder=5)
    ax.yaxis.set_major_formatter(_money_fmt(np.quantile(prices, .98)))
    ax.set_ylim(0, np.quantile(prices, .98) * 1.1)
    ax.set_xlim(0, np.quantile(areas, .98) * 1.1)
    ax.set_xlabel("Built-up area (sq ft)")
    ax.grid(color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def importance_chart(importance, top=9):
    items = list(importance.items())[:top][::-1]
    fig, ax = _fig(6.4, 3.6)
    ax.barh([i[0] for i in items], [i[1] for i in items], color=TEAL, height=.62)
    vmax = max(i[1] for i in items)
    for k, (_, v) in enumerate(items):
        ax.text(v + vmax * .015, k, f"{v:.1f}%", va="center", fontsize=9, color=NAVY, fontweight="bold")
    ax.set_xlim(0, vmax * 1.15)
    ax.set_xlabel("Share of model's predictive power (%)")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def actual_vs_pred_chart(hold):
    fig, ax = _fig(5.4, 4.6)
    a, p = hold["actual"].values, hold["predicted"].values
    ax.scatter(a, p, s=7, color=TEAL, alpha=.30, linewidths=0)
    lo, hi = min(a.min(), p.min()), max(a.max(), p.max())
    ax.plot([lo, hi], [lo, hi], color=GOLD, lw=2, label="Perfect prediction")
    ax.set_xscale("log")
    ax.set_yscale("log")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda v, _: inr(v)))
    ax.set_xlabel("Actual price")
    ax.set_ylabel("Predicted price")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(color=GRID)
    fig.tight_layout()
    return fig


def model_compare_chart(comparison):
    items = sorted(comparison, key=lambda m: m["r2"])
    fig, ax = _fig(6.4, 2.9)
    cols = [GOLD if "final" in m["model"] else SOFT for m in items]
    ax.barh([m["model"] for m in items], [m["r2"] for m in items], color=cols, height=.55)
    for k, m in enumerate(items):
        ax.text(m["r2"] + .008, k, f"R² {m['r2']:.3f}", va="center", fontsize=9, color=NAVY, fontweight="bold")
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("R² score on unseen data (higher is better)")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def emi_donut(principal, interest):
    fig, ax = plt.subplots(figsize=(3.8, 3.8), dpi=130)
    fig.patch.set_alpha(0)
    ax.pie([principal, interest], colors=[TEAL, GOLD], startangle=90, counterclock=False,
           wedgeprops=dict(width=.36, edgecolor="white", linewidth=2))
    ax.text(0, .08, "Total payable", ha="center", fontsize=9, color=MUTED)
    ax.text(0, -.16, inr(principal + interest), ha="center", fontsize=14, color=NAVY, fontweight="bold")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def amortization_chart(rows):
    fig, ax = _fig(6.8, 3.4)
    yrs = [r["year"] for r in rows]
    ax.bar(yrs, [r["principal"] for r in rows], color=TEAL, label="Principal paid", width=.7)
    ax.bar(yrs, [r["interest"] for r in rows], bottom=[r["principal"] for r in rows], color=GOLD,
           label="Interest paid", width=.7)
    ax.yaxis.set_major_formatter(_money_fmt(max(r["principal"] + r["interest"] for r in rows)))
    ax.set_xlabel("Loan year")
    ax2 = ax.twinx()
    ax2.plot(yrs, [r["balance"] for r in rows], color=NAVY, lw=2, label="Outstanding balance")
    ax2.yaxis.set_major_formatter(_money_fmt(max(r["balance"] for r in rows) or 1))
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(GRID)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper center", ncol=3, fontsize=8.5,
              bbox_to_anchor=(.5, 1.16))
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def budget_locality_chart(df, budget, highlight=None):
    """df from Valuer.localities_in_city: green = fits the budget, grey = too expensive."""
    d = df.sort_values("Estimated price")
    fig, ax = _fig(6.6, max(3.2, .34 * len(d) + .9))
    cols = [GREEN if v <= budget else SOFT for v in d["Estimated price"]]
    ax.barh(d["Locality"], d["Estimated price"], color=cols, height=.66)
    vmax = max(d["Estimated price"].max(), budget)
    for i, v in enumerate(d["Estimated price"]):
        ax.text(v + vmax * .012, i, inr(v), va="center", fontsize=8.5, color=NAVY)
    ax.axvline(budget, color=GOLD, lw=2.4)
    ax.text(budget, len(d) - .4, " Your budget", color=NAVY, fontsize=9, fontweight="bold", va="bottom")
    ax.set_xlim(0, vmax * 1.18)
    ax.xaxis.set_major_formatter(_money_fmt(vmax))
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def correlation_chart(df):
    cols = ["price_inr", "area_sqft", "bhk", "bathrooms", "balconies", "floor", "total_floors",
            "age_years", "parking", "metro_km", "locality_rate"]
    names = ["Price", "Area", "BHK", "Baths", "Balconies", "Floor", "Bldg floors", "Age", "Parking", "Metro km", "Locality rate"]
    corr = df[cols].corr().values
    fig, ax = plt.subplots(figsize=(6.6, 5.4), dpi=130)
    fig.patch.set_alpha(0)
    im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticklabels(names, fontsize=8.5)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr[i, j]) > .6 else NAVY)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
    fig.tight_layout()
    return fig


def psf_by_type_chart(df):
    g = (df.assign(psf=df["price_inr"] / df["area_sqft"]).groupby("property_type")["psf"].median()
         .reindex(C.PROPERTY_TYPES))
    fig, ax = _fig(6.4, 3.3)
    ax.bar(g.index, g.values, color=[TEAL, "#2A9D8F", "#5DB4A8", GOLD, NAVY], width=.62)
    for i, v in enumerate(g.values):
        ax.text(i, v + g.max() * .015, f"\u20b9{v:,.0f}", ha="center", fontsize=9, color=NAVY, fontweight="bold")
    ax.set_ylabel("Median \u20b9 per sq ft")
    ax.tick_params(axis="x", labelsize=8.5)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def price_by_bhk_chart(df):
    groups = [df.loc[df["bhk"] == b, "price_inr"].values for b in range(1, 7)]
    fig, ax = _fig(6.4, 3.3)
    bp = ax.boxplot(groups, patch_artist=True, showfliers=False, medianprops=dict(color=GOLD, lw=2))
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels([f"{b} BHK" for b in range(1, 7)])
    for box in bp["boxes"]:
        box.set(facecolor="#BFE3E6", edgecolor=TEAL)
    ax.yaxis.set_major_formatter(_money_fmt(max(np.quantile(g, .95) for g in groups if len(g))))
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def residual_chart(hold):
    err = (hold["predicted"].values / hold["actual"].values - 1) * 100
    fig, ax = _fig(6.0, 3.4)
    ax.hist(np.clip(err, -40, 40), bins=40, color=TEAL, alpha=.88, edgecolor="white")
    ax.axvline(0, color=GOLD, lw=2.2)
    ax.set_xlabel("Prediction error (%)  -  negative = under-estimate")
    ax.set_ylabel("Properties")
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def mape_by_city_chart(by_city):
    items = sorted(by_city.items(), key=lambda t: t[1])
    fig, ax = _fig(6.0, 4.6)
    ax.barh([i[0] for i in items], [i[1] for i in items], color=TEAL, height=.64)
    for k, (_, v) in enumerate(items):
        ax.text(v + .1, k, f"{v:.1f}%", va="center", fontsize=8.5, color=NAVY)
    ax.set_xlim(0, max(v for _, v in items) * 1.15)
    ax.set_xlabel("Average prediction error (%)")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig
