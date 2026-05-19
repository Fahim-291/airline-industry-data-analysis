import warnings
warnings.filterwarnings("ignore")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.stats import norm
from statsmodels.formula.api import logit, ols

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & SAMPLING
# ─────────────────────────────────────────────────────────────────────────────

df_full = pd.read_csv("flight_data.csv")
df = df_full.sample(n=10_000, random_state=12310048).reset_index(drop=True)

print("=" * 70)
print("AIRLINE INDUSTRY DATA ANALYSIS  |  STAT-2205  |  ID: 12310048")
print("=" * 70)
print(f"  Full dataset rows : {len(df_full):,}")
print(f"  Sample rows       : {len(df):,}")
print(f"  Columns           : {list(df.columns)}")


# ─────────────────────────────────────────────────────────────────────────────
# Q1 – COACH TICKET PRICE DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Q1 – COACH TICKET PRICE DISTRIBUTION")
print("=" * 70)

cp = df["coach_price"]

minimum_price    = cp.min()
maximum_price    = cp.max()
average_price    = cp.mean()
median_price     = cp.median()
standard_deviation = cp.std()
first_quartile   = cp.quantile(0.25)
third_quartile   = cp.quantile(0.75)
skewness_value   = cp.skew()
kurtosis_value   = cp.kurtosis()

price_statistics = {
    "Minimum Price"      : minimum_price,
    "Maximum Price"      : maximum_price,
    "Average Price"      : average_price,
    "Median Price"       : median_price,
    "Standard Deviation" : standard_deviation,
    "1st Quartile (Q1)"  : first_quartile,
    "3rd Quartile (Q3)"  : third_quartile,
}

for statistic_name, statistic_value in price_statistics.items():
    print(f"  {statistic_name:<22}: ${statistic_value:,.2f}")

percentage_above_500 = (cp > 500).mean() * 100
print(f"\n  Flights above $500 : {percentage_above_500:.1f}%")
print(f"  Skewness           : {skewness_value:.4f}")
print(f"  Kurtosis           : {kurtosis_value:.4f}")

print("\n  Interpretation:")
print(f"  The average coach price is ${average_price:.2f} with a median of ${median_price:.2f}.")
print(f"  Only {percentage_above_500:.1f}% of tickets cost more than $500.")
print("  $500 is above average — it would be considered an expensive coach ticket.")

# Q1 PLOTS: histogram + boxplot + KDE + ECDF
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram with mean/median/$500 lines
axes[0, 0].hist(cp, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
axes[0, 0].axvline(average_price, color="red",    linestyle="--", linewidth=1.8,
                   label=f"Mean  ${average_price:.0f}")
axes[0, 0].axvline(median_price,  color="orange", linestyle="--", linewidth=1.8,
                   label=f"Median ${median_price:.0f}")
axes[0, 0].axvline(500,           color="green",  linestyle="--", linewidth=1.8,
                   label="$500 mark")
axes[0, 0].set_title("Q1 – Histogram of Coach Prices")
axes[0, 0].set_xlabel("Coach Price ($)")
axes[0, 0].set_ylabel("Count")
axes[0, 0].legend()

# Boxplot
bp = axes[0, 1].boxplot(cp, vert=True, patch_artist=True,
                        boxprops=dict(facecolor="steelblue", color="navy"),
                        medianprops=dict(color="orange", linewidth=2),
                        whiskerprops=dict(color="navy"),
                        capprops=dict(color="navy"),
                        flierprops=dict(marker="o", color="steelblue",
                                        alpha=0.3, markersize=3))
axes[0, 1].axhline(500, color="green", linestyle="--", linewidth=1.8, label="$500 mark")
axes[0, 1].set_title("Q1 – Boxplot of Coach Prices")
axes[0, 1].set_ylabel("Coach Price ($)")
axes[0, 1].legend()

# KDE density plot
cp.plot.kde(ax=axes[1, 0], color="steelblue", linewidth=2)
axes[1, 0].axvline(average_price, color="red",   linestyle="--", linewidth=1.5,
                   label=f"Mean ${average_price:.0f}")
axes[1, 0].axvline(median_price,  color="orange", linestyle="--", linewidth=1.5,
                   label=f"Median ${median_price:.0f}")
axes[1, 0].axvline(500,           color="green",  linestyle="--", linewidth=1.5,
                   label="$500 mark")
axes[1, 0].fill_between(
    np.linspace(500, cp.max(), 200),
    0,
    pd.Series(np.linspace(500, cp.max(), 200)).apply(
        lambda x: stats.gaussian_kde(cp)(x)[0]
    ),
    alpha=0.2, color="red", label=f"Above $500 ({percentage_above_500:.1f}%)"
)
axes[1, 0].set_title("Q1 – KDE Density of Coach Prices")
axes[1, 0].set_xlabel("Coach Price ($)")
axes[1, 0].set_ylabel("Density")
axes[1, 0].legend(fontsize=8)

# ECDF
sorted_cp = np.sort(cp)
ecdf_y    = np.arange(1, len(sorted_cp) + 1) / len(sorted_cp)
axes[1, 1].plot(sorted_cp, ecdf_y, color="steelblue", linewidth=2)
axes[1, 1].axvline(500, color="green", linestyle="--", linewidth=1.5,
                   label=f"$500 → {(cp <= 500).mean()*100:.1f}% of flights")
axes[1, 1].axhline(0.5, color="orange", linestyle=":", linewidth=1.5,
                   label=f"50th pct → ${median_price:.0f}")
axes[1, 1].set_title("Q1 – ECDF of Coach Prices")
axes[1, 1].set_xlabel("Coach Price ($)")
axes[1, 1].set_ylabel("Cumulative Proportion")
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.3)

plt.suptitle("Q1 – Coach Ticket Price Distribution", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("q1_coach_price.png", dpi=150, bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Q2 – COACH PRICES FOR 8-HOUR FLIGHTS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Q2 – COACH PRICES FOR 8-HOUR FLIGHTS")
print("=" * 70)

eight_hour_coach_prices = df[df["hours"] == 8]["coach_price"]
number_of_8hr_flights   = len(eight_hour_coach_prices)

print(f"  Number of 8-hour flights in sample : {number_of_8hr_flights}")

q2_stats = {
    "Min"    : eight_hour_coach_prices.min(),
    "Max"    : eight_hour_coach_prices.max(),
    "Mean"   : eight_hour_coach_prices.mean(),
    "Median" : eight_hour_coach_prices.median(),
    "Std"    : eight_hour_coach_prices.std(),
}
for stat_name, stat_value in q2_stats.items():
    print(f"  {stat_name:<10}: ${stat_value:,.2f}")

pct_8hr_above_500 = (eight_hour_coach_prices > 500).mean() * 100
print(f"\n  8-hr flights above $500 : {pct_8hr_above_500:.1f}%")
print("\n  Interpretation:")
print(f"  For 8-hour flights, the average price is ${q2_stats['Mean']:.2f}.")
print(f"  {pct_8hr_above_500:.1f}% cost over $500, making $500 more reasonable for long-haul flights.")

# Q2 PLOTS: histogram + side-by-side boxplot + violin + bar by hour
hours_order = sorted(df["hours"].unique())
prices_by_hour = [df[df["hours"] == h]["coach_price"].values for h in hours_order]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram: all vs 8-hr
axes[0, 0].hist(cp, bins=40, color="steelblue", alpha=0.6, edgecolor="white",
                label="All flights")
axes[0, 0].hist(eight_hour_coach_prices, bins=20, color="darkorange", alpha=0.8,
                edgecolor="white", label="8-hour flights")
axes[0, 0].axvline(500, color="green", linestyle="--", linewidth=1.8, label="$500 mark")
axes[0, 0].set_title("Q2 – Coach Price: All vs 8-Hour Flights")
axes[0, 0].set_xlabel("Coach Price ($)")
axes[0, 0].set_ylabel("Count")
axes[0, 0].legend()

# Side-by-side boxplot: all vs 8-hr
bp2 = axes[0, 1].boxplot([cp, eight_hour_coach_prices],
                         labels=["All Flights", "8-Hour Flights"],
                         patch_artist=True,
                         medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp2["boxes"], ["steelblue", "darkorange"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
axes[0, 1].axhline(500, color="green", linestyle="--", linewidth=1.8, label="$500 mark")
axes[0, 1].set_title("Q2 – Side-by-Side Boxplot")
axes[0, 1].set_ylabel("Coach Price ($)")
axes[0, 1].legend()

# Violin plot by flight hour
vp = axes[1, 0].violinplot(prices_by_hour, positions=hours_order,
                           showmedians=True, showmeans=False)
for body in vp["bodies"]:
    body.set_facecolor("steelblue")
    body.set_alpha(0.6)
vp["cmedians"].set_color("orange")
axes[1, 0].axhline(500, color="green", linestyle="--", linewidth=1.5, label="$500 mark")
axes[1, 0].set_title("Q2 – Coach Price Distribution by Flight Duration (Violin)")
axes[1, 0].set_xlabel("Flight Hours")
axes[1, 0].set_ylabel("Coach Price ($)")
axes[1, 0].legend()

# Mean price by hour (bar)
mean_price_by_hour = df.groupby("hours")["coach_price"].mean()
axes[1, 1].bar(mean_price_by_hour.index, mean_price_by_hour.values,
               color="steelblue", edgecolor="white", alpha=0.85)
axes[1, 1].axhline(cp.mean(), color="red",   linestyle="--", linewidth=1.5,
                   label=f"Overall Mean ${cp.mean():.0f}")
axes[1, 1].axhline(500,       color="green", linestyle="--", linewidth=1.5,
                   label="$500 mark")
axes[1, 1].set_title("Q2 – Mean Coach Price by Flight Duration")
axes[1, 1].set_xlabel("Flight Hours")
axes[1, 1].set_ylabel("Avg Coach Price ($)")
axes[1, 1].legend()

plt.suptitle("Q2 – Coach Prices for 8-Hour Flights", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("q2_8hour_prices.png", dpi=150, bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Q3 – FLIGHT DELAY DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Q3 – FLIGHT DELAY DISTRIBUTION")
print("=" * 70)

flight_delays = df["delay"]

average_delay        = flight_delays.mean()
median_delay         = flight_delays.median()
std_delay            = flight_delays.std()
maximum_delay        = flight_delays.max()
no_delay_pct         = (flight_delays == 0).mean() * 100
delay_over_15_pct    = (flight_delays > 15).mean() * 100
delay_over_60_pct    = (flight_delays > 60).mean() * 100

print(f"  Mean delay   : {average_delay:.2f} min")
print(f"  Median delay : {median_delay:.2f} min")
print(f"  Std Dev      : {std_delay:.2f} min")
print(f"  Max delay    : {maximum_delay:.0f} min")
print(f"  No delay (0) : {no_delay_pct:.1f}%")
print(f"  > 15 min     : {delay_over_15_pct:.1f}%  (connection risk)")
print(f"  > 60 min     : {delay_over_60_pct:.1f}%  (severe delay)")

print("\n  Interpretation:")
print("  Most flights have zero or very small delays.")
print(f"  About {delay_over_15_pct:.1f}% of flights have delays longer than 15 minutes,")
print("  which may affect connecting flights.")
print("  Severe delays longer than 60 minutes are relatively uncommon.")

# Q3 PLOTS: histogram + boxplot + CDF + delay-category bar
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram (capped at 120 min for readability)
delay_capped = flight_delays[flight_delays <= 120]
axes[0, 0].hist(delay_capped, bins=50, color="tomato", edgecolor="white", alpha=0.85)
axes[0, 0].axvline(15, color="orange", linestyle="--", linewidth=1.8,
                   label=f"15 min — {delay_over_15_pct:.1f}% at risk")
axes[0, 0].axvline(60, color="darkred", linestyle="--", linewidth=1.8,
                   label=f"60 min — {delay_over_60_pct:.2f}% severe")
axes[0, 0].set_title("Q3 – Histogram of Delay Times (≤ 120 min)")
axes[0, 0].set_xlabel("Delay (minutes)")
axes[0, 0].set_ylabel("Count")
axes[0, 0].legend()

# Boxplot
axes[0, 1].boxplot(flight_delays, patch_artist=True,
                   boxprops=dict(facecolor="tomato"),
                   medianprops=dict(color="darkred", linewidth=2),
                   flierprops=dict(marker="o", color="tomato", alpha=0.2, markersize=3))
axes[0, 1].set_title("Q3 – Boxplot of Delay Times")
axes[0, 1].set_ylabel("Delay (minutes)")

# CDF of delay
sorted_delay = np.sort(flight_delays)
ecdf_delay   = np.arange(1, len(sorted_delay) + 1) / len(sorted_delay)
axes[1, 0].plot(sorted_delay, ecdf_delay, color="tomato", linewidth=2)
axes[1, 0].axvline(15, color="orange",  linestyle="--", linewidth=1.5,
                   label=f"15 min ({(flight_delays <= 15).mean()*100:.1f}%)")
axes[1, 0].axvline(60, color="darkred", linestyle="--", linewidth=1.5,
                   label=f"60 min ({(flight_delays <= 60).mean()*100:.1f}%)")
axes[1, 0].set_xlim(0, 200)
axes[1, 0].set_title("Q3 – ECDF of Delay Times")
axes[1, 0].set_xlabel("Delay (minutes)")
axes[1, 0].set_ylabel("Cumulative Proportion")
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Delay-category bar chart
delay_categories = {
    "No delay (0)"     : (flight_delays == 0).sum(),
    "1–15 min"         : ((flight_delays > 0)  & (flight_delays <= 15)).sum(),
    "16–60 min"        : ((flight_delays > 15) & (flight_delays <= 60)).sum(),
    "> 60 min (severe)": (flight_delays > 60).sum(),
}
cat_colors = ["seagreen", "steelblue", "orange", "darkred"]
axes[1, 1].bar(list(delay_categories.keys()), list(delay_categories.values()),
               color=cat_colors, edgecolor="white")
axes[1, 1].set_title("Q3 – Flights by Delay Category")
axes[1, 1].set_xlabel("Delay Category")
axes[1, 1].set_ylabel("Number of Flights")
axes[1, 1].tick_params(axis="x", rotation=15)
for i, (k, v) in enumerate(delay_categories.items()):
    axes[1, 1].text(i, v + 20, f"{v:,}", ha="center", fontsize=9)

plt.suptitle("Q3 – Flight Delay Distribution", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("q3_delays.png", dpi=150, bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Q4 – CORRELATION: COACH PRICE VS. PREDICTORS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Q4 – CORRELATION: COACH PRICE VS. PREDICTORS")
print("=" * 70)

numeric_variables = ["miles", "passengers", "delay", "hours"]

corr_results = {}
for variable_name in numeric_variables:
    r, p = stats.pearsonr(df[variable_name], df["coach_price"])
    corr_results[variable_name] = (r, p)
    significance = "✓ significant" if p < 0.05 else "✗ not significant"
    print(f"  coach_price ~ {variable_name:<12}: r = {r:+.4f}  p = {p:.4f}  {significance}")

print("\n  Interpretation:")
print("  Miles and hours are the strongest positive predictors of coach price.")
print("  Passengers has a weaker but significant correlation.")
print("  Delay is not a significant pricing factor.")

# Q4 PLOTS: 4 scatterplots + correlation bar chart
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
plot_colors = ["steelblue", "darkorange", "tomato", "seagreen"]

for ax, variable_name, color_name in zip(axes.flat[:4], numeric_variables, plot_colors):
    ax.scatter(df[variable_name], df["coach_price"], alpha=0.1, s=5, color=color_name)
    slope, intercept = np.polyfit(df[variable_name], df["coach_price"], 1)
    x_line = np.linspace(df[variable_name].min(), df[variable_name].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=2)
    r, _ = corr_results[variable_name]
    ax.set_title(f"Coach Price vs {variable_name.capitalize()}\n(r = {r:+.3f})")
    ax.set_xlabel(variable_name)
    ax.set_ylabel("Coach Price ($)")

# Correlation bar chart
r_values = [corr_results[v][0] for v in numeric_variables]
bar_colors = ["steelblue" if r >= 0 else "tomato" for r in r_values]
axes[1, 1].barh(numeric_variables, r_values, color=bar_colors, edgecolor="white")
axes[1, 1].axvline(0, color="black", linewidth=1)
axes[1, 1].set_title("Q4 – Pearson r with Coach Price")
axes[1, 1].set_xlabel("Pearson r")
for i, r in enumerate(r_values):
    axes[1, 1].text(r + 0.005 if r >= 0 else r - 0.005, i,
                    f"{r:+.3f}", va="center", fontsize=9)

# Pairwise scatterplot matrix (miles & hours as strongest predictors)
axes[1, 2].scatter(df["miles"], df["hours"], c=df["coach_price"],
                   cmap="coolwarm", alpha=0.3, s=5)
sm_plot = axes[1, 2].scatter(df["miles"], df["hours"], c=df["coach_price"],
                              cmap="coolwarm", alpha=0.3, s=5)
plt.colorbar(sm_plot, ax=axes[1, 2], label="Coach Price ($)")
axes[1, 2].set_title("Q4 – Miles vs Hours\n(colored by Coach Price)")
axes[1, 2].set_xlabel("Miles")
axes[1, 2].set_ylabel("Hours")

plt.suptitle("Q4 – Coach Price Correlations with Numeric Predictors", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("q4_correlations.png", dpi=150, bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Q5 – COACH VS. FIRST-CLASS PRICE RELATIONSHIP
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Q5 – COACH VS. FIRST-CLASS PRICE RELATIONSHIP")
print("=" * 70)

coach_prices     = df["coach_price"]
firstclass_prices = df["firstclass_price"]

r_q5, p_q5   = stats.pearsonr(coach_prices, firstclass_prices)
average_ratio = (firstclass_prices / coach_prices).mean()

print(f"  Pearson r              : {r_q5:.4f}")
print(f"  p-value                : {p_q5:.4e}")
print(f"  Avg first-class/coach  : {average_ratio:.2f}x")
print(f"  Avg coach price        : ${coach_prices.mean():.2f}")
print(f"  Avg first-class price  : ${firstclass_prices.mean():.2f}")

print("\n  Interpretation:")
print(f"  Strong positive correlation (r = {r_q5:.3f}, p < 0.001).")
print(f"  First-class tickets cost ~{average_ratio:.1f}× more than coach tickets.")
print("  Higher coach prices generally predict higher first-class prices.")

# Q5 PLOTS: scatter + residual + ratio distribution + hexbin
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Scatterplot with regression line
slope5, intercept5 = np.polyfit(coach_prices, firstclass_prices, 1)
x_line5 = np.linspace(coach_prices.min(), coach_prices.max(), 200)
axes[0, 0].scatter(coach_prices, firstclass_prices, alpha=0.12, s=5, color="purple")
axes[0, 0].plot(x_line5, slope5 * x_line5 + intercept5, color="black", linewidth=2)
axes[0, 0].set_title(f"Q5 – Coach vs First-Class Price\n(r = {r_q5:.3f})")
axes[0, 0].set_xlabel("Coach Price ($)")
axes[0, 0].set_ylabel("First-Class Price ($)")

# Hexbin density
hb = axes[0, 1].hexbin(coach_prices, firstclass_prices, gridsize=40, cmap="Purples",
                        mincnt=1)
plt.colorbar(hb, ax=axes[0, 1], label="Count")
axes[0, 1].set_title("Q5 – Hexbin Density: Coach vs First-Class")
axes[0, 1].set_xlabel("Coach Price ($)")
axes[0, 1].set_ylabel("First-Class Price ($)")

# First-class / Coach ratio distribution
price_ratio = firstclass_prices / coach_prices
axes[1, 0].hist(price_ratio, bins=40, color="mediumpurple", edgecolor="white", alpha=0.85)
axes[1, 0].axvline(average_ratio, color="red", linestyle="--", linewidth=1.8,
                   label=f"Mean ratio {average_ratio:.2f}x")
axes[1, 0].set_title("Q5 – Distribution of First-Class / Coach Price Ratio")
axes[1, 0].set_xlabel("Price Ratio (First-Class / Coach)")
axes[1, 0].set_ylabel("Count")
axes[1, 0].legend()

# Average prices side-by-side bar
avg_labels  = ["Coach", "First-Class"]
avg_values  = [coach_prices.mean(), firstclass_prices.mean()]
bar_colors5 = ["steelblue", "mediumpurple"]
bars5 = axes[1, 1].bar(avg_labels, avg_values, color=bar_colors5, edgecolor="white", width=0.5)
for bar, val in zip(bars5, avg_values):
    axes[1, 1].text(bar.get_x() + bar.get_width() / 2, val + 20,
                    f"${val:,.0f}", ha="center", fontsize=11, fontweight="bold")
axes[1, 1].set_title("Q5 – Average Coach vs First-Class Price")
axes[1, 1].set_ylabel("Avg Price ($)")

plt.suptitle("Q5 – Coach vs First-Class Price Relationship", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("q5_coach_vs_first.png", dpi=150, bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Q6 – IN-FLIGHT FEATURES AND COACH PRICE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Q6 – IN-FLIGHT FEATURES AND COACH PRICE")
print("=" * 70)

inflight_features = ["inflight_meal", "inflight_entertainment", "inflight_wifi"]
feature_results   = {}

for feature in inflight_features:
    yes_group = df[df[feature] == "Yes"]["coach_price"]
    no_group  = df[df[feature] == "No"]["coach_price"]
    t_stat, p_val = stats.ttest_ind(yes_group, no_group)
    price_diff = yes_group.mean() - no_group.mean()
    feature_results[feature] = {
        "yes_mean" : yes_group.mean(),
        "no_mean"  : no_group.mean(),
        "diff"     : price_diff,
        "t"        : t_stat,
        "p"        : p_val,
    }
    sig = "✓ sig" if p_val < 0.05 else "✗ not sig"
    print(f"  {feature:<26}: Yes=${yes_group.mean():.2f}  No=${no_group.mean():.2f}  "
          f"Diff=+${price_diff:.2f}  p={p_val:.4f}  {sig}")

print("\n  Interpretation:")
print("  All three in-flight features are associated with significantly higher prices.")
print("  WiFi and Entertainment carry the largest price premiums (~+$68–71).")

# Q6 PLOTS: boxplots + mean-diff bar + grouped bar + violin per feature
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Boxplots for all three features on one axis
all_group_data  = []
all_group_ticks = []
positions       = []
pos             = 1
tick_labels     = []
for feature in inflight_features:
    yes_data = df[df[feature] == "Yes"]["coach_price"].values
    no_data  = df[df[feature] == "No" ]["coach_price"].values
    all_group_data.extend([yes_data, no_data])
    positions.extend([pos, pos + 1])
    tick_labels.extend([f"{feature.replace('inflight_', '')[:5]}\nYes",
                        f"{feature.replace('inflight_', '')[:5]}\nNo"])
    pos += 3

bp6 = axes[0, 0].boxplot(all_group_data, positions=positions, patch_artist=True,
                          medianprops=dict(color="black", linewidth=2),
                          widths=0.7)
colors6 = ["steelblue", "lightsteelblue", "darkorange", "moccasin",
           "seagreen", "lightgreen"]
for patch, color in zip(bp6["boxes"], colors6):
    patch.set_facecolor(color)
axes[0, 0].set_xticks(positions)
axes[0, 0].set_xticklabels(tick_labels, fontsize=8)
axes[0, 0].set_title("Q6 – Coach Price by In-Flight Feature (Yes vs No)")
axes[0, 0].set_ylabel("Coach Price ($)")

# Price premium bar chart
feature_short = ["Meal", "Entertainment", "WiFi"]
diffs = [feature_results[f]["diff"] for f in inflight_features]
axes[0, 1].bar(feature_short, diffs, color=["steelblue", "darkorange", "seagreen"],
               edgecolor="white")
for i, d in enumerate(diffs):
    axes[0, 1].text(i, d + 0.5, f"+${d:.2f}", ha="center", fontsize=10, fontweight="bold")
axes[0, 1].set_title("Q6 – Price Premium per In-Flight Feature")
axes[0, 1].set_ylabel("Price Difference ($)")
axes[0, 1].set_xlabel("Feature")

# Grouped bar: Yes vs No mean prices
x_pos  = np.arange(len(feature_short))
width  = 0.35
yes_means = [feature_results[f]["yes_mean"] for f in inflight_features]
no_means  = [feature_results[f]["no_mean"]  for f in inflight_features]
axes[1, 0].bar(x_pos - width / 2, yes_means, width, label="With Feature",
               color="steelblue", edgecolor="white")
axes[1, 0].bar(x_pos + width / 2, no_means,  width, label="Without Feature",
               color="lightsteelblue", edgecolor="white")
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(feature_short)
axes[1, 0].set_title("Q6 – Mean Coach Price: With vs Without Feature")
axes[1, 0].set_ylabel("Avg Coach Price ($)")
axes[1, 0].legend()
axes[1, 0].set_ylim(300, 420)

# Violin for WiFi (largest premium)
wifi_yes = df[df["inflight_wifi"] == "Yes"]["coach_price"].values
wifi_no  = df[df["inflight_wifi"] == "No" ]["coach_price"].values
vp6 = axes[1, 1].violinplot([wifi_yes, wifi_no], positions=[1, 2],
                             showmedians=True, showmeans=True)
for i, body in enumerate(vp6["bodies"]):
    body.set_facecolor(["seagreen", "lightgreen"][i])
    body.set_alpha(0.7)
axes[1, 1].set_xticks([1, 2])
axes[1, 1].set_xticklabels(["WiFi: Yes", "WiFi: No"])
axes[1, 1].set_title("Q6 – Violin: Coach Price by WiFi Availability")
axes[1, 1].set_ylabel("Coach Price ($)")

plt.suptitle("Q6 – In-Flight Features and Coach Price", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("q6_inflight_features.png", dpi=150, bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Q7 – PASSENGERS VS. FLIGHT DURATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Q7 – PASSENGERS VS. FLIGHT DURATION")
print("=" * 70)

flight_hours         = df["hours"]
number_of_passengers = df["passengers"]

r_q7, p_q7 = stats.pearsonr(flight_hours, number_of_passengers)
avg_pax_by_hour = df.groupby("hours")["passengers"].mean()

print(f"  Pearson r = {r_q7:.4f},  p-value = {p_q7:.4f}")
print("\n  Average passengers by flight hours:")
print(avg_pax_by_hour.to_string())
print("\n  Interpretation:")
print(f"  r = {r_q7:.3f} — essentially no relationship between flight duration and passengers.")
print("  Passenger numbers stay flat (~207–208) across all flight durations.")

# Q7 PLOTS: scatter + bar + boxplot by hour + KDE
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Scatter
axes[0, 0].scatter(flight_hours, number_of_passengers, alpha=0.1, s=5, color="teal")
slope7, intercept7 = np.polyfit(flight_hours, number_of_passengers, 1)
x7 = np.linspace(flight_hours.min(), flight_hours.max(), 100)
axes[0, 0].plot(x7, slope7 * x7 + intercept7, color="black", linewidth=2,
                label=f"Trend (r={r_q7:.3f})")
axes[0, 0].set_title("Q7 – Passengers vs Hours")
axes[0, 0].set_xlabel("Flight Hours")
axes[0, 0].set_ylabel("Passengers")
axes[0, 0].legend()

# Bar: avg passengers per hour
axes[0, 1].bar(avg_pax_by_hour.index, avg_pax_by_hour.values,
               color="teal", edgecolor="white", alpha=0.85)
axes[0, 1].axhline(number_of_passengers.mean(), color="red", linestyle="--",
                   linewidth=1.5, label=f"Overall mean {number_of_passengers.mean():.1f}")
axes[0, 1].set_title("Q7 – Avg Passengers by Flight Duration")
axes[0, 1].set_xlabel("Flight Hours")
axes[0, 1].set_ylabel("Avg Passengers")
axes[0, 1].legend()
axes[0, 1].set_ylim(190, 220)

# Boxplot by hour
pax_by_hour_lists = [df[df["hours"] == h]["passengers"].values for h in hours_order]
bp7 = axes[1, 0].boxplot(pax_by_hour_lists, positions=hours_order, patch_artist=True,
                          widths=0.6, medianprops=dict(color="black", linewidth=2))
for patch in bp7["boxes"]:
    patch.set_facecolor("teal")
    patch.set_alpha(0.6)
axes[1, 0].set_title("Q7 – Passenger Distribution by Hour (Boxplot)")
axes[1, 0].set_xlabel("Flight Hours")
axes[1, 0].set_ylabel("Passengers")

# Histogram of passengers
axes[1, 1].hist(number_of_passengers, bins=40, color="teal", edgecolor="white", alpha=0.85)
axes[1, 1].axvline(number_of_passengers.mean(), color="red", linestyle="--",
                   linewidth=1.8, label=f"Mean {number_of_passengers.mean():.1f}")
axes[1, 1].axvline(number_of_passengers.median(), color="orange", linestyle="--",
                   linewidth=1.8, label=f"Median {number_of_passengers.median():.0f}")
axes[1, 1].set_title("Q7 – Histogram of Passengers")
axes[1, 1].set_xlabel("Passengers")
axes[1, 1].set_ylabel("Count")
axes[1, 1].legend()

plt.suptitle("Q7 – Passengers vs Flight Duration", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("q7_passengers_hours.png", dpi=150, bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Q8 – WEEKEND VS. WEEKDAY PRICING
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Q8 – WEEKEND VS. WEEKDAY PRICING")
print("=" * 70)

for price_column in ["coach_price", "firstclass_price"]:
    weekend_prices = df[df["weekend"] == "Yes"][price_column]
    weekday_prices = df[df["weekend"] == "No" ][price_column]
    t_stat, p_val  = stats.ttest_ind(weekend_prices, weekday_prices)
    sig = "✓ sig" if p_val < 0.05 else "✗ not sig"
    print(f"  {price_column}:")
    print(f"    Weekend avg : ${weekend_prices.mean():.2f}  |  Weekday avg : ${weekday_prices.mean():.2f}")
    print(f"    Diff : ${weekend_prices.mean()-weekday_prices.mean():.2f}  t = {t_stat:.3f},  p = {p_val:.4f}  {sig}")

print("\n  Interpretation:")
print("  Both coach and first-class ticket prices are significantly higher on weekends.")
print("  Weekend coach fares are ~$82 more; first-class fares are ~$300 more.")

# Q8 PLOTS: boxplot + violin + bar + scatter coach vs firstclass colored by weekend
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Boxplot for both ticket classes side by side
for i, (price_col, color) in enumerate(zip(["coach_price", "firstclass_price"],
                                            ["steelblue", "mediumpurple"])):
    ax = axes[0, i]
    wknd = df[df["weekend"] == "Yes"][price_col].values
    wkdy = df[df["weekend"] == "No" ][price_col].values
    bp8 = ax.boxplot([wknd, wkdy], labels=["Weekend", "Weekday"],
                     patch_artist=True, medianprops=dict(color="black", linewidth=2))
    for patch in bp8["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_title(f"Q8 – {price_col.replace('_', ' ').title()}: Weekend vs Weekday")
    ax.set_ylabel("Price ($)")

# Violin: coach price by weekend
wknd_cp = df[df["weekend"] == "Yes"]["coach_price"].values
wkdy_cp = df[df["weekend"] == "No" ]["coach_price"].values
vp8 = axes[1, 0].violinplot([wknd_cp, wkdy_cp], positions=[1, 2],
                             showmedians=True, showmeans=True)
for j, body in enumerate(vp8["bodies"]):
    body.set_facecolor(["coral", "steelblue"][j])
    body.set_alpha(0.75)
axes[1, 0].set_xticks([1, 2])
axes[1, 0].set_xticklabels(["Weekend", "Weekday"])
axes[1, 0].set_title("Q8 – Violin: Coach Price Weekend vs Weekday")
axes[1, 0].set_ylabel("Coach Price ($)")

# Bar: mean prices by weekend/weekday for both classes
categories = ["Coach\nWeekend", "Coach\nWeekday", "First-Class\nWeekend", "First-Class\nWeekday"]
means = [
    df[df["weekend"] == "Yes"]["coach_price"].mean(),
    df[df["weekend"] == "No" ]["coach_price"].mean(),
    df[df["weekend"] == "Yes"]["firstclass_price"].mean(),
    df[df["weekend"] == "No" ]["firstclass_price"].mean(),
]
bar_colors8 = ["coral", "steelblue", "darkorchid", "mediumpurple"]
bars8 = axes[1, 1].bar(categories, means, color=bar_colors8, edgecolor="white")
for bar, val in zip(bars8, means):
    axes[1, 1].text(bar.get_x() + bar.get_width() / 2, val + 15,
                    f"${val:,.0f}", ha="center", fontsize=9, fontweight="bold")
axes[1, 1].set_title("Q8 – Mean Prices: Weekend vs Weekday (Both Classes)")
axes[1, 1].set_ylabel("Avg Price ($)")

plt.suptitle("Q8 – Weekend vs Weekday Pricing", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("q8_weekend_weekday.png", dpi=150, bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Q9 – REDEYE VS. NON-REDEYE BY DAY OF WEEK
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Q9 – COACH PRICES: REDEYE VS. NON-REDEYE BY DAY OF WEEK")
print("=" * 70)

days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

pivot_q9 = df.pivot_table(
    values="coach_price", index="day_of_week", columns="redeye", aggfunc="mean"
).reindex(days_of_week)

print(pivot_q9.round(2))
print("\n  Interpretation:")
print("  Non-redeye flights are consistently more expensive every day.")
print("  Weekend (Fri–Sun) commands the highest prices for both flight types.")
print("  Cheapest option: weekday redeye (Monday/Wednesday).")

# Q9 PLOTS: grouped bar + line + heatmap + boxplot by day
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Grouped bar
x_q9    = np.arange(len(days_of_week))
width_q9 = 0.35
redeye_no_vals  = pivot_q9["No"].values
redeye_yes_vals = pivot_q9["Yes"].values
axes[0, 0].bar(x_q9 - width_q9 / 2, redeye_no_vals,  width_q9, label="Non-Redeye",
               color="steelblue", edgecolor="white")
axes[0, 0].bar(x_q9 + width_q9 / 2, redeye_yes_vals, width_q9, label="Redeye",
               color="tomato", edgecolor="white")
axes[0, 0].set_xticks(x_q9)
axes[0, 0].set_xticklabels([d[:3] for d in days_of_week])
axes[0, 0].set_title("Q9 – Coach Price: Redeye vs Non-Redeye by Day (Bar)")
axes[0, 0].set_ylabel("Avg Coach Price ($)")
axes[0, 0].legend()

# Line chart
axes[0, 1].plot(days_of_week, redeye_no_vals,  "o-", color="steelblue",
                linewidth=2, label="Non-Redeye", markersize=7)
axes[0, 1].plot(days_of_week, redeye_yes_vals, "s--", color="tomato",
                linewidth=2, label="Redeye",     markersize=7)
axes[0, 1].fill_between(range(len(days_of_week)), redeye_yes_vals, redeye_no_vals,
                         alpha=0.15, color="gray", label="Price gap")
axes[0, 1].set_xticks(range(len(days_of_week)))
axes[0, 1].set_xticklabels([d[:3] for d in days_of_week])
axes[0, 1].set_title("Q9 – Coach Price Trend by Day (Line)")
axes[0, 1].set_ylabel("Avg Coach Price ($)")
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Heatmap (redeye × day)
sns.heatmap(pivot_q9.T, annot=True, fmt=".0f", cmap="RdYlGn_r",
            ax=axes[1, 0], linewidths=0.5,
            xticklabels=[d[:3] for d in days_of_week])
axes[1, 0].set_title("Q9 – Heatmap: Avg Coach Price by Day & Redeye")
axes[1, 0].set_xlabel("Day of Week")
axes[1, 0].set_ylabel("Redeye")

# Discount (price gap) bar
discounts = redeye_no_vals - redeye_yes_vals
axes[1, 1].bar([d[:3] for d in days_of_week], discounts,
               color="darkred", edgecolor="white", alpha=0.8)
for i, d in enumerate(discounts):
    axes[1, 1].text(i, d + 0.5, f"${d:.0f}", ha="center", fontsize=9)
axes[1, 1].set_title("Q9 – Redeye Discount vs Non-Redeye by Day")
axes[1, 1].set_xlabel("Day of Week")
axes[1, 1].set_ylabel("Price Discount ($)")

plt.suptitle("Q9 – Coach Prices: Redeye vs Non-Redeye by Day", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("q9_redeye_by_day.png", dpi=150, bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Q10 – COMPREHENSIVE STATISTICAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Q10 – COMPREHENSIVE STATISTICAL ANALYSIS")
print("=" * 70)

numeric_columns = ["miles", "passengers", "delay", "coach_price", "firstclass_price", "hours"]
cat_columns     = ["inflight_meal", "inflight_entertainment", "inflight_wifi",
                   "redeye", "weekend", "day_of_week"]

# ── 10a. Summary Statistics ──────────────────────────────────────────────────
print("\n── 10a. Summary Statistics ──")
summary_stats = df[numeric_columns].agg(["mean", "median", "std", "min", "max",
                                         lambda x: x.skew(),
                                         lambda x: x.kurtosis()]).round(3)
summary_stats.index = ["mean", "median", "std", "min", "max", "skewness", "kurtosis"]
print(summary_stats.to_string())


# ── 10b. Visualizations ──────────────────────────────────────────────────────
print("\n── 10b. Visualizations ──")

# Histograms
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
hist_colors = ["steelblue", "darkorange", "tomato", "seagreen", "mediumpurple", "teal"]
for axis, col, color in zip(axes.flat, numeric_columns, hist_colors):
    axis.hist(df[col], bins=35, color=color, edgecolor="white", alpha=0.85)
    axis.axvline(df[col].mean(),   color="red",    linestyle="--", linewidth=1.5,
                 label=f"Mean {df[col].mean():.1f}")
    axis.axvline(df[col].median(), color="orange", linestyle=":",  linewidth=1.5,
                 label=f"Median {df[col].median():.1f}")
    axis.set_title(f"Histogram – {col}")
    axis.set_xlabel(col)
    axis.set_ylabel("Count")
    axis.legend(fontsize=7)
plt.suptitle("Q10b – Histograms of Numeric Variables", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("q10b_histograms.png", dpi=150, bbox_inches="tight")
plt.show()

# Boxplots
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for axis, col, color in zip(axes.flat, numeric_columns, hist_colors):
    axis.boxplot(df[col], patch_artist=True,
                 boxprops=dict(facecolor=color, alpha=0.7),
                 medianprops=dict(color="black", linewidth=2),
                 flierprops=dict(marker="o", color=color, alpha=0.2, markersize=3))
    axis.set_title(f"Boxplot – {col}")
    axis.set_ylabel(col)
plt.suptitle("Q10b – Boxplots of Numeric Variables", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("q10b_boxplots.png", dpi=150, bbox_inches="tight")
plt.show()

# Bar charts for categorical columns
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for axis, col in zip(axes.flat, cat_columns):
    counts = df[col].value_counts()
    bars = axis.bar(counts.index, counts.values, color="darkorange", edgecolor="white")
    for bar, val in zip(bars, counts.values):
        axis.text(bar.get_x() + bar.get_width() / 2, val + 20,
                  f"{val:,}", ha="center", fontsize=8)
    axis.set_title(f"Bar Chart – {col}")
    axis.set_ylabel("Count")
    axis.tick_params(axis="x", rotation=30)
plt.suptitle("Q10b – Bar Charts of Categorical Variables", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("q10b_barcharts.png", dpi=150, bbox_inches="tight")
plt.show()

# KDE plots for numeric columns
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for axis, col, color in zip(axes.flat, numeric_columns, hist_colors):
    df[col].plot.kde(ax=axis, color=color, linewidth=2)
    axis.axvline(df[col].mean(),   color="red",    linestyle="--", linewidth=1.3,
                 label=f"Mean {df[col].mean():.1f}")
    axis.axvline(df[col].median(), color="orange", linestyle=":",  linewidth=1.3,
                 label=f"Median {df[col].median():.1f}")
    axis.set_title(f"KDE – {col}")
    axis.set_xlabel(col)
    axis.legend(fontsize=7)
plt.suptitle("Q10b – KDE Plots of Numeric Variables", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("q10b_kde.png", dpi=150, bbox_inches="tight")
plt.show()


# ── 10c. Hypothesis Testing ──────────────────────────────────────────────────
print("\n── 10c. Hypothesis Testing ──")

cp = df["coach_price"]

# Z-test (H0: mean = $350)
z_stat = (cp.mean() - 350) / (cp.std() / np.sqrt(len(cp)))
z_pval = 2 * (1 - norm.cdf(abs(z_stat)))
print(f"\n  Z-test (H0: μ=$350): z={z_stat:.4f},  p={z_pval:.4e}  "
      f"{'Reject H0' if z_pval < 0.05 else 'Fail to reject H0'}")

# t-test (H0: mean = $380)
t_stat, t_pval = stats.ttest_1samp(cp, 380)
print(f"  t-test (H0: μ=$380): t={t_stat:.4f},  p={t_pval:.4e}  "
      f"{'Reject H0' if t_pval < 0.05 else 'Fail to reject H0'}")

# Chi-square tests
for col_a, col_b in [("redeye", "inflight_meal"), ("redeye", "weekend")]:
    ct  = pd.crosstab(df[col_a], df[col_b])
    chi2, p_chi, dof, _ = stats.chi2_contingency(ct)
    decision = "Reject H0 (dependent)" if p_chi < 0.05 else "Fail to reject H0 (independent)"
    print(f"  Chi-sq: {col_a} × {col_b}: χ²={chi2:.4f},  df={dof},  p={p_chi:.4f}  {decision}")

# Visualize hypothesis tests
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Z-distribution with shading
x_norm = np.linspace(-5, 5, 400)
axes[0].plot(x_norm, norm.pdf(x_norm), "steelblue", linewidth=2)
axes[0].fill_between(x_norm[x_norm >= abs(z_stat)],
                      norm.pdf(x_norm[x_norm >= abs(z_stat)]),
                      color="red", alpha=0.4, label=f"Reject region z>{abs(z_stat):.2f}")
axes[0].fill_between(x_norm[x_norm <= -abs(z_stat)],
                      norm.pdf(x_norm[x_norm <= -abs(z_stat)]),
                      color="red", alpha=0.4)
axes[0].axvline(z_stat, color="darkred", linestyle="--", label=f"z={z_stat:.2f}")
axes[0].set_title("10c – Z-test: H₀: μ=$350")
axes[0].set_xlabel("z-statistic")
axes[0].legend(fontsize=8)

# Chi-square: redeye × meal
ct1 = pd.crosstab(df["redeye"], df["inflight_meal"])
sns.heatmap(ct1, annot=True, fmt="d", cmap="Blues", ax=axes[1], linewidths=0.5)
axes[1].set_title("10c – Contingency: Redeye × Inflight Meal")

# Chi-square: redeye × weekend
ct2 = pd.crosstab(df["redeye"], df["weekend"])
sns.heatmap(ct2, annot=True, fmt="d", cmap="Greens", ax=axes[2], linewidths=0.5)
axes[2].set_title("10c – Contingency: Redeye × Weekend")

plt.suptitle("Q10c – Hypothesis Testing", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("q10c_hypothesis_tests.png", dpi=150, bbox_inches="tight")
plt.show()


# ── 10d. Independent t-Tests ─────────────────────────────────────────────────
print("\n── 10d. Independent t-Tests ──")

comparison_groups = [
    ("weekend",               "coach_price", "Weekend vs Weekday — Coach Price"),
    ("redeye",                "coach_price", "Redeye vs Non-Redeye — Coach Price"),
    ("inflight_meal",         "coach_price", "Meal vs No-Meal — Coach Price"),
    ("inflight_entertainment","coach_price", "Entertainment vs No-Entertainment — Coach Price"),
    ("inflight_wifi",         "coach_price", "WiFi vs No-WiFi — Coach Price"),
]

ttest_results = []
for group_col, target_col, label in comparison_groups:
    yes_g = df[df[group_col] == "Yes"][target_col]
    no_g  = df[df[group_col] == "No" ][target_col]
    t_s, p_v = stats.ttest_ind(yes_g, no_g)
    ttest_results.append({
        "label"   : label,
        "yes_mean": yes_g.mean(),
        "no_mean" : no_g.mean(),
        "t"       : t_s,
        "p"       : p_v,
    })
    sig = "✓ sig" if p_v < 0.05 else "✗ not sig"
    print(f"\n  {label}")
    print(f"  Yes: ${yes_g.mean():.2f}  No: ${no_g.mean():.2f}  "
          f"t={t_s:.3f}  p={p_v:.4f}  {sig}")

# t-test visualization: forest-plot style
fig, ax = plt.subplots(figsize=(12, 5))
short_labels = ["Weekend", "Redeye", "Meal", "Entertainment", "WiFi"]
diffs_10d    = [r["yes_mean"] - r["no_mean"] for r in ttest_results]
bar_c10d     = ["steelblue" if d > 0 else "tomato" for d in diffs_10d]
y_pos = range(len(short_labels))
ax.barh(list(y_pos), diffs_10d, color=bar_c10d, edgecolor="white")
ax.axvline(0, color="black", linewidth=1)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(short_labels)
ax.set_title("Q10d – Mean Coach Price Difference (Yes − No) per Group")
ax.set_xlabel("Difference in Avg Coach Price ($)")
for i, d in enumerate(diffs_10d):
    ax.text(d + (1 if d >= 0 else -1), i, f"${d:+.2f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("q10d_ttest_forest.png", dpi=150, bbox_inches="tight")
plt.show()


# ── 10e. Weekend vs Weekday Differences ──────────────────────────────────────
print("\n── 10e. Weekend vs Weekday Price Differences ──")

wknd_coach = df[df["weekend"] == "Yes"]["coach_price"].mean()
wkdy_coach = df[df["weekend"] == "No" ]["coach_price"].mean()
wknd_fc    = df[df["weekend"] == "Yes"]["firstclass_price"].mean()
wkdy_fc    = df[df["weekend"] == "No" ]["firstclass_price"].mean()

print(f"  Coach      — Weekend: ${wknd_coach:.2f}  Weekday: ${wkdy_coach:.2f}  Diff: ${wknd_coach-wkdy_coach:+.2f}")
print(f"  First-Class— Weekend: ${wknd_fc:.2f}  Weekday: ${wkdy_fc:.2f}  Diff: ${wknd_fc-wkdy_fc:+.2f}")


# ── 10f. Correlation Matrix ───────────────────────────────────────────────────
print("\n── 10f. Correlation Matrix ──")

corr_matrix = df[numeric_columns].corr()
print(corr_matrix.round(4).to_string())

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Full heatmap
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            ax=axes[0], linewidths=0.5, vmin=-1, vmax=1)
axes[0].set_title("Q10f – Full Correlation Heatmap")

# Upper-triangle only
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            mask=mask, ax=axes[1], linewidths=0.5, vmin=-1, vmax=1)
axes[1].set_title("Q10f – Upper-Triangle Correlation Heatmap")

plt.suptitle("Q10f – Correlation Analysis", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("q10f_correlation.png", dpi=150, bbox_inches="tight")
plt.show()


# ── 10g. Linear Regression ───────────────────────────────────────────────────
print("\n── 10g. Linear Regression: Predicting Coach Price ──")

encoded_df = df.copy()
for col in ["inflight_meal", "inflight_entertainment", "inflight_wifi", "redeye", "weekend"]:
    encoded_df[col] = (encoded_df[col] == "Yes").astype(int)

predictor_cols = ["miles", "passengers", "delay", "hours",
                  "inflight_meal", "inflight_entertainment", "inflight_wifi"]

X_reg = sm.add_constant(encoded_df[predictor_cols])
y_reg = encoded_df["coach_price"]

linear_model = sm.OLS(y_reg, X_reg).fit()
print(linear_model.summary())

# Regression plots: predicted vs actual + residuals
y_pred_lm    = linear_model.fittedvalues
residuals_lm = linear_model.resid

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Predicted vs Actual
axes[0, 0].scatter(y_reg, y_pred_lm, alpha=0.1, s=5, color="steelblue")
lims = [min(y_reg.min(), y_pred_lm.min()), max(y_reg.max(), y_pred_lm.max())]
axes[0, 0].plot(lims, lims, "r--", linewidth=2, label="Perfect fit")
axes[0, 0].set_title(f"10g – Predicted vs Actual Coach Price (R²={linear_model.rsquared:.3f})")
axes[0, 0].set_xlabel("Actual Coach Price ($)")
axes[0, 0].set_ylabel("Predicted Coach Price ($)")
axes[0, 0].legend()

# Residuals vs Fitted
axes[0, 1].scatter(y_pred_lm, residuals_lm, alpha=0.1, s=5, color="tomato")
axes[0, 1].axhline(0, color="black", linewidth=1.5, linestyle="--")
axes[0, 1].set_title("10g – Residuals vs Fitted Values")
axes[0, 1].set_xlabel("Fitted Values")
axes[0, 1].set_ylabel("Residuals")

# Histogram of residuals
axes[1, 0].hist(residuals_lm, bins=50, color="tomato", edgecolor="white", alpha=0.85)
axes[1, 0].axvline(0, color="black", linestyle="--", linewidth=1.5)
axes[1, 0].set_title("10g – Histogram of Residuals")
axes[1, 0].set_xlabel("Residual")
axes[1, 0].set_ylabel("Count")

# Coefficient plot
coef_df = pd.DataFrame({
    "coef"  : linear_model.params,
    "ci_low": linear_model.conf_int()[0],
    "ci_hi" : linear_model.conf_int()[1],
}).drop("const")
coef_df = coef_df.sort_values("coef")
axes[1, 1].barh(coef_df.index, coef_df["coef"],
                xerr=[coef_df["coef"] - coef_df["ci_low"],
                      coef_df["ci_hi"]  - coef_df["coef"]],
                color=["steelblue" if c > 0 else "tomato" for c in coef_df["coef"]],
                edgecolor="white", capsize=4)
axes[1, 1].axvline(0, color="black", linewidth=1)
axes[1, 1].set_title("10g – Regression Coefficients with 95% CI")
axes[1, 1].set_xlabel("Coefficient Value")

plt.suptitle("Q10g – Linear Regression: Coach Price", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("q10g_linear_regression.png", dpi=150, bbox_inches="tight")
plt.show()


# ── 10h. Logistic Regression ──────────────────────────────────────────────────
print("\n── 10h. Logistic Regression: Predicting Redeye Flights ──")

encoded_df["redeye_binary"] = (df["redeye"] == "Yes").astype(int)

logit_pred_cols = ["coach_price", "miles", "hours", "passengers", "delay"]
X_log = sm.add_constant(encoded_df[logit_pred_cols])
y_log = encoded_df["redeye_binary"]

logit_model = sm.Logit(y_log, X_log).fit(disp=False)
print(logit_model.summary())

pred_probs   = logit_model.predict(X_log)
pred_classes = (pred_probs >= 0.5).astype(int)
accuracy     = (pred_classes == y_log).mean()
print(f"\n  Classification Accuracy : {accuracy * 100:.2f}%")

# Logistic regression plots
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Predicted probability histogram by actual class
axes[0].hist(pred_probs[y_log == 0], bins=30, alpha=0.7, color="steelblue",
             edgecolor="white", label="Non-Redeye (actual)")
axes[0].hist(pred_probs[y_log == 1], bins=30, alpha=0.7, color="tomato",
             edgecolor="white", label="Redeye (actual)")
axes[0].axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Threshold 0.5")
axes[0].set_title("10h – Predicted Probabilities by Actual Class")
axes[0].set_xlabel("P(Redeye = 1)")
axes[0].set_ylabel("Count")
axes[0].legend(fontsize=8)

# Confusion matrix heatmap
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_log, pred_classes)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
            xticklabels=["Pred: No", "Pred: Yes"],
            yticklabels=["Actual: No", "Actual: Yes"])
axes[1].set_title(f"10h – Confusion Matrix (Acc={accuracy*100:.2f}%)")

# Log-odds (coefficients) bar chart
log_coef = logit_model.params.drop("const")
axes[2].barh(log_coef.index, log_coef.values,
             color=["steelblue" if c > 0 else "tomato" for c in log_coef.values],
             edgecolor="white")
axes[2].axvline(0, color="black", linewidth=1)
axes[2].set_title("10h – Logistic Regression Coefficients (Log-Odds)")
axes[2].set_xlabel("Log-Odds Coefficient")

plt.suptitle("Q10h – Logistic Regression: Redeye Prediction", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("q10h_logistic_regression.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE — All plots saved.")
print("=" * 70)
