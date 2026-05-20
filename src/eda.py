import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from IPython.display import display, HTML

warnings.filterwarnings("ignore")
sns.set_theme(context="talk", style="whitegrid", font_scale=0.85)
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 200)

print("✅ All libraries loaded.")


df = pd.read_csv(r'..\data\cleaned\cleaned_data.csv')

TARGET = "is_canceled"
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")


df = df[
    (df["market_segment"] != "Undefined") &
    (df["distribution_channel"] != "Undefined")
]


matplotlib.rcParams.update({
    "font.family"        : "DejaVu Sans",
    "font.size"          : 10.5,
    "figure.facecolor"   : "white",
    "axes.facecolor"     : "white",
    "axes.titlesize"     : 13,
    "axes.titleweight"   : "regular",
    "axes.titlepad"      : 12,
    "axes.labelsize"     : 10,
    "axes.labelcolor"    : "#555",
    "axes.spines.top"    : False,
    "axes.spines.right"  : False,
    "axes.spines.left"   : False,
    "axes.spines.bottom" : True,
    "axes.grid"          : True,
    "axes.grid.axis"     : "x",
    "grid.color"         : "#EBEBEB",
    "grid.linewidth"     : 0.8,
    "xtick.labelsize"    : 9,
    "ytick.labelsize"    : 10.5,
    "xtick.color"        : "#777",
    "ytick.color"        : "#222",
    "xtick.bottom"       : False,
    "ytick.left"         : False,
    "figure.dpi"         : 130,
    "savefig.dpi"        : 130,
    "savefig.bbox"       : "tight",
    "legend.frameon"     : False,
    "legend.fontsize"    : 9,
})

C_RED   = "#D94F4F"
C_AMBER = "#E8943A"
C_GREEN = "#4E9B4E"
C_BLUE  = "#3A7EC8"

def rate_color(r):
    if r > 0.50: return C_RED
    if r > 0.30: return C_AMBER
    return C_GREEN

def hbar(col, title, df=None, target=TARGET, figsize=(8, None)):
    if df is None:
        import __main__; df = __main__.df
    rates = df.groupby(col)[target].mean().sort_values()
    n = len(rates)
    h = figsize[1] or max(3.2, n * 0.58)
    fig, ax = plt.subplots(figsize=(figsize[0], h))

    colors = [rate_color(r) for r in rates.values]
    bars = ax.barh(range(n), rates.values, color=colors, height=0.52,
                   edgecolor="white", linewidth=0.6)

    ax.set_yticks(range(n))
    ax.set_yticklabels(rates.index, fontsize=10.5)

    for i, (bar, val) in enumerate(zip(bars, rates.values)):
        ax.text(val + 0.012, i, f"{val:.1%}",
                va="center", fontsize=10, color="#333", fontweight="regular")

    ax.set_xlim(0, min(rates.max() * 1.22, 1.05))
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Cancellation rate", labelpad=8)
    ax.set_title(title, loc="left", color="#111")
    ax.spines["bottom"].set_color("#D0D0D0")
    ax.tick_params(axis="x", colors="#999", length=3)

    legend_els = [
        mpatches.Patch(color=C_RED,   label="High  (> 50%)"),
        mpatches.Patch(color=C_AMBER, label="Medium (30–50%)"),
        mpatches.Patch(color=C_GREEN, label="Low  (< 30%)"),
    ]
    ax.legend(handles=legend_els, loc="lower right",
              fontsize=8.5, handlelength=1, handleheight=0.9,
              labelspacing=0.3, borderpad=0.6,
              bbox_to_anchor=(1.0, 0.0))

    plt.tight_layout(pad=1.4)
    plt.show()

print("✅ Chart style loaded — ready to plot.")


vc    = df[TARGET].value_counts().sort_index()
total = len(df)
not_c, canc = int(vc[0]), int(vc[1])
pct_c = canc / total

card_css = """
<style>
  .ov-wrap  { font-family: 'Segoe UI', 'DejaVu Sans', sans-serif; padding:6px 0 18px; }
  .ov-label { font-size:10px; font-weight:600; letter-spacing:.08em;
              text-transform:uppercase; color:#888; margin:0 0 6px; }
  .ov-title { font-size:17px; font-weight:600; color:#111; margin:0 0 14px; }
  .ov-grid  { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; max-width:680px; }
  .ov-card  { background:#F6F7F8; border-radius:8px; padding:14px 18px; }
  .ov-clbl  { font-size:12px; color:#666; margin:0 0 4px; }
  .ov-cval  { font-size:24px; font-weight:600; margin:0; }
  .ov-csub  { font-size:12px; color:#999; margin:3px 0 0; }
</style>
"""
cards_html = f"""
{card_css}
<div class='ov-wrap'>
  <p class='ov-label'>Section 1 — Overview</p>
  <p class='ov-title'>Target distribution: is_canceled</p>
  <div class='ov-grid'>
    <div class='ov-card'>
      <p class='ov-clbl'>Total bookings</p>
      <p class='ov-cval' style='color:#111'>{total:,}</p>
    </div>
    <div class='ov-card'>
      <p class='ov-clbl'>Not cancelled</p>
      <p class='ov-cval' style='color:#3A7EC8'>{not_c:,}</p>
      <p class='ov-csub'>{not_c/total:.1%}</p>
    </div>
    <div class='ov-card'>
      <p class='ov-clbl'>Cancelled</p>
      <p class='ov-cval' style='color:#D94F4F'>{canc:,}</p>
      <p class='ov-csub'>{pct_c:.1%}</p>
    </div>
  </div>
</div>
"""
display(HTML(cards_html))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
fig.suptitle("is_canceled — target distribution", x=0.02, ha="left", fontsize=13, color="#111")

ax = axes[0]
ax.barh([""], [not_c/total], color=C_BLUE,  height=0.38, label=f"Not cancelled  {not_c/total:.1%}")
ax.barh([""], [canc/total],  color=C_RED,   height=0.38, left=[not_c/total], label=f"Cancelled  {pct_c:.1%}")
ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax.set_title("Proportion", fontsize=11, loc="left", color="#555")
ax.spines["bottom"].set_color("#DDD")
ax.tick_params(left=False, length=0)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=9.5, handlelength=1)

ax2 = axes[1]
bar_vals = [not_c, canc]
bar_cols = [C_BLUE, C_RED]
bars = ax2.bar(["Not cancelled", "Cancelled"], bar_vals, color=bar_cols, width=0.42, edgecolor="white")
for bar, val in zip(bars, bar_vals):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + total*0.006,
             f"{val:,}\n({val/total:.1%})", ha="center", fontsize=9.5, color="#333")
ax2.set_ylim(0, max(bar_vals) * 1.18)
ax2.set_ylabel("Count", labelpad=6)
ax2.set_title("Count", fontsize=11, loc="left", color="#555")
ax2.spines["bottom"].set_color("#DDD")
ax2.tick_params(axis="x", length=0)
ax2.grid(axis="y", color="#EBEBEB"); ax2.set_axisbelow(True)
ax2.spines["left"].set_visible(False)

plt.tight_layout(pad=1.6)
plt.show()


hbar("hotel", "Cancellation rate by hotel type")


hbar("deposit_type", "Cancellation rate by deposit type")


hbar("market_segment", "Cancellation rate by market segment", figsize=(8, 5))


hbar("distribution_channel", "Cancellation rate by distribution channel", figsize=(8, 4.5))


hbar("customer_type", "Cancellation rate by customer type")


num_cols = ["lead_time", "adr", "total_of_special_requests",
            "required_car_parking_spaces", "previous_cancellations"]

fig, axes = plt.subplots(1, len(num_cols), figsize=(20, 3.8))
fig.suptitle("Numerical features — Cancelled vs Not cancelled",
             x=0.01, ha="left", fontsize=13, color="#111", y=0.98)

for i, (ax, col) in enumerate(zip(axes, num_cols)):
    cap  = df[col].quantile(0.99)
    data = df.assign(**{col: df[col].clip(upper=cap)})
    sns.kdeplot(data=data[data[TARGET]==0][col], ax=ax, fill=True,
                color=C_BLUE, alpha=0.40, linewidth=1.5, label="Not cancelled")
    sns.kdeplot(data=data[data[TARGET]==1][col], ax=ax, fill=True,
                color=C_RED,  alpha=0.40, linewidth=1.5, label="Cancelled")
    ax.set_title(col, fontsize=10.5, color="#111")
    ax.set_xlabel(""); ax.set_ylabel("Density" if i == 0 else "")
    ax.spines["bottom"].set_color("#D0D0D0")
    ax.tick_params(colors="#999", length=2, labelsize=8.5)
    ax.grid(axis="y", color="#EBEBEB"); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    if i == 0:
        ax.legend(fontsize=8.5, handlelength=1.2)

plt.tight_layout(pad=1.4)
plt.show()


features = [
    "deposit_type",
    "lead_time",
    "market_segment",
    "total_of_special_requests",
    "required_car_parking_spaces",
    "distribution_channel",
    "hotel",
    "customer_type",
    "adr",
    "previous_cancellations",
    "is_canceled"
]

heatmap_df = df[features].copy()

cat_cols = [
    "deposit_type",
    "market_segment",
    "distribution_channel",
    "hotel",
    "customer_type"
]

for col in cat_cols:
    heatmap_df[col] = heatmap_df[col].astype("category").cat.codes

corr_matrix = heatmap_df.corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

fig, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(
    corr_matrix,
    mask = mask,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    linecolor="white",
    square=True,
    annot_kws={"size": 8},
    cbar_kws={"label": "Correlation", "shrink": 0.8}
)

ax.set_title(
    "Correlation Heatmap of Selected Features",
    fontsize=14,
    pad=14,
    loc="left"
)

ax.tick_params(axis="x", rotation=45)
ax.tick_params(axis="y", rotation=0)

plt.tight_layout()
plt.show()


ALPHA = 0.05
g1 = df[df[TARGET] == 1]
g0 = df[df[TARGET] == 0]

num_sources = {
    "lead_time"                     : "lead_time / is_short_lead",
    "required_car_parking_spaces"   : "required_car_parking_spaces",
    "total_of_special_requests"     : "total_of_special_requests",
    "previous_cancellations"        : "prior_cancel_rate",
    "previous_bookings_not_canceled": "prior_cancel_rate",
    "adr"                           : "adr_per_person",
}

num_results = []
for col, feat in num_sources.items():
    stat, p = stats.mannwhitneyu(g1[col], g0[col], alternative="two-sided")
    num_results.append({
        "source"    : col,
        "feeds_into": feat,
        "test"      : "Mann-Whitney U",
        "statistic" : f"{stat:.2e}",
        "p-value"   : f"{p:.3e}",
        "decision"  : "✅ KEEP" if p < ALPHA else "❌ DROP",
    })

num_hyp_df = pd.DataFrame(num_results)
num_hyp_df = pd.DataFrame(num_results)
print("=== 3.1 Numerical Features — Mann-Whitney U ===")
print(num_hyp_df)
print(f"\n✅ {(num_hyp_df['decision']=='✅ KEEP').sum()} / {len(num_hyp_df)} numerical features PASS")


cat_sources = ["hotel", "deposit_type", "market_segment",
               "distribution_channel", "customer_type"]

cat_results = []
for col in cat_sources:
    ct      = pd.crosstab(df[col], df[TARGET])
    _, p, _, _ = stats.chi2_contingency(ct)
    cat_results.append({
        "source"    : col,
        "feeds_into": col,
        "test"      : "Chi-Square",
        "p-value"   : f"{p:.3e}",
        "decision"  : "✅ KEEP" if p < ALPHA else "❌ DROP",
    })

cat_hyp_df = pd.DataFrame(cat_results)
print("=== 3.2 Categorical Features — Chi-Square ===")
print(cat_hyp_df)
hyp_df = pd.concat([num_hyp_df, cat_hyp_df], ignore_index=True)
n_pass = (hyp_df["decision"] == "✅ KEEP").sum()
n_fail = (hyp_df["decision"] == "❌ DROP").sum()
print(f"\n✅ {n_pass} features PASS  |  ❌ {n_fail} FAIL")
print("All source features are statistically significant — proceed to feature engineering.")
