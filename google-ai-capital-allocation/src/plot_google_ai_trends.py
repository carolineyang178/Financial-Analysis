from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


# ==================================================
# 1. Configuration & Styling Setup
# ==================================================

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DATA = OUTPUT_DIR / "data"
CHART_DIR = OUTPUT_DIR / "charts"
DATA_DIR = ROOT / "data"

# Source files
ANNUAL_SOURCE_FILE = DATA_DIR / "Annual_dataset_FY2019_FY2025.csv"
QUARTERLY_SOURCE_FILE = OUTPUT_DIR / "data/source_observations.csv"

CHART_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 2023
END_YEAR = 2026
END_QUARTER = 2

# Source CSV values are in US$ millions.
MILLIONS_PER_BILLION = 1000.0

CASH_FLOW_METRICS = {
    "operating_cash_flow",
    "capex",
    "ppe_depreciation",
}

OPERATING_METRICS = {
    "revenue",
    "operating_income",
    "google_cloud_revenue",
    "google_cloud_operating_income",
    "search_other_revenue",
}

# --- Modern & Professional Palette ---
COLOR_PRIMARY_BLUE = "#3B82F6"  # Bar chart / primary series (blue).
COLOR_ACCENT_ORANGE = "#F97316" # Line chart / secondary series (orange).
COLOR_DARK = "#1E293B"          # Text color.
COLOR_MUTED = "#64748B"         # Axis and grid color.

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "text.color": COLOR_DARK,
    "axes.labelcolor": COLOR_DARK,
    "xtick.color": COLOR_MUTED,
    "ytick.color": COLOR_MUTED,
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 1.0,
})


# ==================================================
# 2. Load Data Sources (Annual & Quarterly)
# ==================================================

def load_annual_data():
    if not ANNUAL_SOURCE_FILE.exists():
        raise FileNotFoundError(f"Cannot find annual dataset at {ANNUAL_SOURCE_FILE}")
    
    df = pd.read_csv(ANNUAL_SOURCE_FILE)
    df["fiscal_year"] = pd.to_numeric(df["fiscal_year"], errors="raise").astype(int)
    return df.sort_values("fiscal_year")


def load_quarterly_source_data():
    if not QUARTERLY_SOURCE_FILE.exists():
        # Fallback or empty DataFrame if quarterly source observations are not present
        return pd.DataFrame(columns=["fiscal_year", "period", "metric", "value", "basis"])

    df = pd.read_csv(QUARTERLY_SOURCE_FILE)
    required = {"fiscal_year", "period", "metric", "value", "basis"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=required)

    df["fiscal_year"] = pd.to_numeric(df["fiscal_year"], errors="raise").astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["period"] = df["period"].str.upper().str.strip()
    df["metric"] = df["metric"].str.lower().str.strip()
    df["basis"] = df["basis"].str.lower().str.strip()
    return df


# ==================================================
# 3. Build standalone quarterly series (Kept intact)
# ==================================================

def get_single_value(df, year, period, metric, basis):
    if df.empty:
        return np.nan
    rows = df[
        (df["fiscal_year"] == year)
        & (df["period"] == period)
        & (df["metric"] == metric)
        & (df["basis"] == basis)
    ]
    if len(rows) != 1:
        return np.nan
    return rows.iloc[0]["value"]


def build_quarterly_data(raw):
    records = []
    if raw.empty:
        return pd.DataFrame(columns=["year", "quarter", "period_id", "metric", "value", "method", "value_bn"])

    for year in range(START_YEAR, END_YEAR + 1):
        quarters = ["Q1", "Q2", "Q3", "Q4"]

        for metric in CASH_FLOW_METRICS:
            q1_ytd = get_single_value(raw, year, "Q1", metric, "ytd")
            q2_ytd = get_single_value(raw, year, "Q2", metric, "ytd")
            q3_ytd = get_single_value(raw, year, "Q3", metric, "ytd")
            annual = get_single_value(raw, year, "FY", metric, "annual")

            values = {
                "Q1": q1_ytd,
                "Q2": (q2_ytd - q1_ytd if pd.notna(q2_ytd) and pd.notna(q1_ytd) else np.nan),
                "Q3": (q3_ytd - q2_ytd if pd.notna(q3_ytd) and pd.notna(q2_ytd) else np.nan),
                "Q4": (annual - q3_ytd if pd.notna(annual) and pd.notna(q3_ytd) else np.nan),
            }

            for quarter, value in values.items():
                if year == END_YEAR and quarter in ["Q3", "Q4"]:
                    continue

                records.append({
                    "year": year,
                    "quarter": quarter,
                    "period_id": f"{year}{quarter}",
                    "metric": metric,
                    "value": value,
                    "method": "YTD-derived" if quarter != "Q1" else "Q1 YTD",
                })

        for metric in OPERATING_METRICS:
            annual = get_single_value(raw, year, "FY", metric, "annual")
            reported_q4 = get_single_value(raw, year, "Q4", metric, "quarter")

            quarter_values = {}
            for quarter in ["Q1", "Q2", "Q3"]:
                quarter_values[quarter] = get_single_value(raw, year, quarter, metric, "quarter")

            if pd.notna(reported_q4):
                quarter_values["Q4"] = reported_q4
                q4_method = "Reported quarter"
            elif pd.notna(annual) and all(pd.notna(quarter_values[q]) for q in ["Q1", "Q2", "Q3"]):
                quarter_values["Q4"] = annual - sum(quarter_values[q] for q in ["Q1", "Q2", "Q3"])
                q4_method = "Annual less Q1-Q3"
            else:
                quarter_values["Q4"] = np.nan
                q4_method = "Unverified / incomplete"

            for quarter in quarters:
                if year == END_YEAR and quarter in ["Q3", "Q4"]:
                    continue

                records.append({
                    "year": year,
                    "quarter": quarter,
                    "period_id": f"{year}{quarter}",
                    "metric": metric,
                    "value": quarter_values[quarter],
                    "method": q4_method if quarter == "Q4" else "Reported quarter",
                })

    quarterly = pd.DataFrame(records)
    if not quarterly.empty:
        quarterly["value_bn"] = quarterly["value"] / MILLIONS_PER_BILLION
    return quarterly


# ==================================================
# 4. Plot formatting helpers
# ==================================================

def format_axis(ax, ylabel="US$ billions", title=""):
    ax.set_ylabel(ylabel, fontsize=11, fontweight="semibold", labelpad=10)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15, loc="center")
    
    ax.grid(True, axis="y", linestyle="--", alpha=0.5, color="#CBD5E1")
    ax.grid(False, axis="x")
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    ax.tick_params(axis="both", which="major", labelsize=10, length=0)


def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(
        CHART_DIR / filename,
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor='none',
    )
    plt.close(fig)


def quarterly_series(quarterly, metric):
    if quarterly.empty:
        return pd.DataFrame()
    subset = quarterly[quarterly["metric"] == metric].copy()
    return subset.sort_values(["year", "quarter"])


# ==================================================
# 5. Annual Charts (Using Annual_dataset_FY2019_FY2025.csv)
# ==================================================

def plot_annual_revenue(annual_df):
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="white")
    df_bn = annual_df.copy()
    for col in ["revenue", "search_other_revenue", "google_cloud_revenue"]:
        df_bn[f"{col}_bn"] = df_bn[col] / MILLIONS_PER_BILLION

    ax.plot(
        df_bn["fiscal_year"].astype(str),
        df_bn["revenue_bn"],
        marker="o",
        markersize=6,
        linewidth=2.5,
        color=COLOR_PRIMARY_BLUE,
        label="Total Revenue",
        zorder=3,
    )
    ax.plot(
        df_bn["fiscal_year"].astype(str),
        df_bn["search_other_revenue_bn"],
        marker="s",
        markersize=6,
        linewidth=2.0,
        color=COLOR_ACCENT_ORANGE,
        label="Search & Other Revenue",
        zorder=3,
    )

    format_axis(ax, ylabel="US$ billions", title="Alphabet Annual Revenue Trend (FY2019 - FY2025)")
    ax.set_xlabel("Fiscal Year", fontsize=11, fontweight="semibold", labelpad=10)
    ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor="none", fontsize=10)

    save_figure(fig, "annual_revenue_trend.png")


def plot_annual_capex_fcf(annual_df):
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="white")
    df_bn = annual_df.copy()
    df_bn["capex_bn"] = df_bn["capex"] / MILLIONS_PER_BILLION
    df_bn["fcf_bn"] = df_bn["fcf"] / MILLIONS_PER_BILLION

    ax.plot(
        df_bn["fiscal_year"].astype(str),
        df_bn["capex_bn"],
        marker="o",
        markersize=6,
        linewidth=2.5,
        color=COLOR_ACCENT_ORANGE,
        label="Capital Expenditure (CapEx)",
        zorder=3,
    )
    ax.plot(
        df_bn["fiscal_year"].astype(str),
        df_bn["fcf_bn"],
        marker="s",
        markersize=6,
        linewidth=2.5,
        color=COLOR_PRIMARY_BLUE,
        label="Free Cash Flow (FCF)",
        zorder=3,
    )
    ax.axhline(0, linewidth=1.2, linestyle="--", color=COLOR_MUTED, zorder=2)

    format_axis(ax, ylabel="US$ billions", title="Alphabet Annual CapEx & Free Cash Flow (FY2019 - FY2025)")
    ax.set_xlabel("Fiscal Year", fontsize=11, fontweight="semibold", labelpad=10)
    ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor="none", fontsize=10)

    save_figure(fig, "annual_capex_fcf.png")


def plot_annual_cloud_profitability(annual_df):
    df = annual_df.copy()
    years = df["fiscal_year"].astype(str)
    x_indices = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")

    ax.bar(
        x_indices,
        df["google_cloud_revenue"],
        color=COLOR_PRIMARY_BLUE,
        width=0.55,
        label="Google Cloud Revenue",
        zorder=3
    )

    ax.plot(
        x_indices,
        df["google_cloud_operating_income"],
        color=COLOR_ACCENT_ORANGE,
        linewidth=2.5,
        marker="o",
        label="Google Cloud Operating Income",
        zorder=4
    )

    ax.axhline(0, linewidth=1, color="#94A3B8", linestyle="-", zorder=2)
    ax.grid(True, which="both", linestyle="-", alpha=0.3, color="#CBD5E1", zorder=1)

    ax.set_title("Google Cloud Annual Results (FY2019 - FY2025, US$ millions)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(years, fontsize=10)
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False,
        fontsize=10
    )

    save_figure(fig, "annual_cloud_profitability.png")


# ==================================================
# 6. Quarter-Related Image Generation (Kept Intact)
# ==================================================

def plot_revenue(quarterly, annual):
    q = quarterly_series(quarterly, "revenue")
    if q.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="white")

    ax.plot(
        q["period_id"],
        q["value_bn"],
        marker="o",
        markersize=6,
        linewidth=2.5,
        color=COLOR_PRIMARY_BLUE,
        label="Quarterly revenue",
        zorder=3,
    )

    derived = q[q["method"] == "Annual less Q1-Q3"]
    if not derived.empty:
        ax.scatter(
            derived["period_id"],
            derived["value_bn"],
            marker="s",
            s=55,
            color=COLOR_ACCENT_ORANGE,
            label="Q4 derived from annual totals",
            zorder=4,
        )

    format_axis(ax, ylabel="US$ billions", title="Alphabet Quarterly Revenue Trend")
    ax.set_xlabel("Fiscal quarter", fontsize=11, fontweight="semibold", labelpad=10)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor="none", fontsize=10)

    save_figure(fig, "01_revenue_quarterly.png")


def plot_capex(quarterly, annual):
    q = quarterly_series(quarterly, "capex")
    if q.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="white")

    ax.plot(
        q["period_id"],
        q["value_bn"],
        marker="o",
        markersize=6,
        linewidth=2.5,
        color=COLOR_PRIMARY_BLUE,
        label="Quarterly CapEx",
        zorder=3,
    )

    format_axis(ax, ylabel="US$ billions", title="Alphabet Capital Expenditure Trend")
    ax.set_xlabel("Fiscal quarter", fontsize=11, fontweight="semibold", labelpad=10)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor="none", fontsize=10)

    save_figure(fig, "02_capex_quarterly.png")


def plot_fcf(quarterly, annual):
    q = quarterly_series(quarterly, "operating_cash_flow").rename(columns={"value_bn": "ocf_bn"})
    c = quarterly_series(quarterly, "capex").rename(columns={"value_bn": "capex_bn"})

    if q.empty or c.empty:
        return

    fcf = q[["period_id", "year", "quarter", "ocf_bn"]].merge(
        c[["period_id", "capex_bn"]],
        on="period_id",
        how="outer",
    )
    fcf["fcf_bn"] = fcf["ocf_bn"] - fcf["capex_bn"]

    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="white")
    ax.plot(
        fcf["period_id"],
        fcf["fcf_bn"],
        marker="o",
        markersize=6,
        linewidth=2.5,
        color=COLOR_PRIMARY_BLUE,
        label="Calculated FCF",
        zorder=3,
    )
    ax.axhline(0, linewidth=1.2, linestyle="--", color=COLOR_MUTED, zorder=2)

    format_axis(ax, ylabel="US$ billions", title="Alphabet Free Cash Flow Trend")
    ax.set_xlabel("Fiscal quarter", fontsize=11, fontweight="semibold", labelpad=10)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor="none", fontsize=10)

    save_figure(fig, "03_fcf_quarterly.png")


def plot_cloud_profitability(quarterly):
    revenue = quarterly_series(quarterly, "google_cloud_revenue").rename(columns={"value": "cloud_revenue_m"})
    income = quarterly_series(quarterly, "google_cloud_operating_income").rename(columns={"value": "cloud_operating_income_m"})

    if revenue.empty or income.empty:
        return

    cloud = revenue[["period_id", "year", "quarter", "cloud_revenue_m"]].merge(
        income[["period_id", "cloud_operating_income_m"]],
        on="period_id",
        how="outer",
    )
    cloud = cloud.sort_values("period_id")
    cloud["x_label"] = cloud["quarter"] + "-" + cloud["year"].astype(str)

    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
    x_indices = np.arange(len(cloud))

    ax.bar(
        x_indices,
        cloud["cloud_revenue_m"],
        color=COLOR_PRIMARY_BLUE,
        width=0.65,
        label="Google Cloud Revenue",
        zorder=3
    )

    ax.plot(
        x_indices,
        cloud["cloud_operating_income_m"],
        color=COLOR_ACCENT_ORANGE,
        linewidth=2.5,
        marker=None,
        label="Google Cloud Operating Income",
        zorder=4
    )

    ax.axhline(0, linewidth=1, color="#94A3B8", linestyle="-", zorder=2)
    ax.grid(True, which="both", linestyle="-", alpha=0.3, color="#CBD5E1", zorder=1)

    ax.set_title("Google Cloud Results (x $1 million)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(cloud["x_label"], rotation=90, fontsize=9)
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False,
        fontsize=10
    )

    save_figure(fig, "04_cloud_profitability.png")


# ==================================================
# 7. Main Execution
# ==================================================

def main():
    # Load annual data from Annual_dataset_FY2019_FY2025.csv
    annual_df = load_annual_data()

    # Load quarterly source data (if available)
    quarterly_raw = load_quarterly_source_data()
    quarterly = build_quarterly_data(quarterly_raw)

    # Export CSV data summaries
    annual_df.to_csv(OUTPUT_DATA / "annual_chart_data.csv", index=False)
    if not quarterly.empty:
        quarterly.to_csv(OUTPUT_DATA / "quarterly_chart_data.csv", index=False)

    # Plot Annual Images (using Annual_dataset_FY2019_FY2025.csv)
    plot_annual_revenue(annual_df)
    plot_annual_capex_fcf(annual_df)
    plot_annual_cloud_profitability(annual_df)

    # Plot Quarterly Images (keeping quarter-related logic and generation intact)
    plot_revenue(quarterly, annual_df)
    plot_capex(quarterly, annual_df)
    plot_fcf(quarterly, annual_df)
    plot_cloud_profitability(quarterly)

    print(f"Charts successfully saved to: {CHART_DIR.resolve()}")
    print("\nGenerated chart files:")
    for file in sorted(CHART_DIR.glob("*.png")):
        print(f"  {file.name}")


if __name__ == "__main__":
    main()