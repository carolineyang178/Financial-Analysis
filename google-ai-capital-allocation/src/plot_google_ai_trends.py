
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


# ==================================================
# 1. Configuration
# ==================================================

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"

SOURCE_FILE = OUTPUT_DIR / "data/source_observations.csv"

CHART_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 2023
END_YEAR = 2026
END_QUARTER = 2

# Source CSV values are in US$ millions.
# Charts display US$ billions.
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


# ==================================================
# 2. Load source observations
# ==================================================

def load_source_data():
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {SOURCE_FILE}\n"
            "Run google_ai_capital_allocation.py first."
        )

    df = pd.read_csv(SOURCE_FILE)

    required = {
        "fiscal_year",
        "period",
        "metric",
        "value",
        "basis",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["fiscal_year"] = pd.to_numeric(
        df["fiscal_year"], errors="raise"
    ).astype(int)

    df["value"] = pd.to_numeric(
        df["value"], errors="coerce"
    )

    df["period"] = df["period"].str.upper().str.strip()
    df["metric"] = df["metric"].str.lower().str.strip()
    df["basis"] = df["basis"].str.lower().str.strip()

    # Avoid silently using duplicate observations.
    key = ["fiscal_year", "period", "metric", "basis"]

    if df.duplicated(key).any():
        duplicated = df.loc[
            df.duplicated(key, keep=False), key
        ]
        raise ValueError(
            "Duplicate source observations found:\n"
            + duplicated.to_string(index=False)
        )

    return df


# ==================================================
# 3. Build standalone quarterly series
# ==================================================

def get_single_value(df, year, period, metric, basis):
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
    """
    Cash-flow metrics:
      Q1 = Q1 YTD
      Q2 = Q2 YTD - Q1 YTD
      Q3 = Q3 YTD - Q2 YTD
      Q4 = FY - Q3 YTD

    Income-statement / segment metrics:
      Q1-Q3 = reported standalone quarter
      Q4 = FY - Q1 - Q2 - Q3

    Missing observations remain NaN.
    """

    records = []

    for year in range(START_YEAR, END_YEAR + 1):
        quarters = ["Q1", "Q2", "Q3", "Q4"]

        # Cash-flow observations
        for metric in CASH_FLOW_METRICS:
            q1_ytd = get_single_value(
                raw, year, "Q1", metric, "ytd"
            )
            q2_ytd = get_single_value(
                raw, year, "Q2", metric, "ytd"
            )
            q3_ytd = get_single_value(
                raw, year, "Q3", metric, "ytd"
            )
            annual = get_single_value(
                raw, year, "FY", metric, "annual"
            )

            values = {
                "Q1": q1_ytd,
                "Q2": (
                    q2_ytd - q1_ytd
                    if pd.notna(q2_ytd)
                    and pd.notna(q1_ytd)
                    else np.nan
                ),
                "Q3": (
                    q3_ytd - q2_ytd
                    if pd.notna(q3_ytd)
                    and pd.notna(q2_ytd)
                    else np.nan
                ),
                "Q4": (
                    annual - q3_ytd
                    if pd.notna(annual)
                    and pd.notna(q3_ytd)
                    else np.nan
                ),
            }

            for quarter, value in values.items():
                if year == END_YEAR and quarter == "Q3":
                    continue
                if year == END_YEAR and quarter == "Q4":
                    continue

                records.append({
                    "year": year,
                    "quarter": quarter,
                    "period_id": f"{year}{quarter}",
                    "metric": metric,
                    "value": value,
                    "method": (
                        "YTD-derived"
                        if quarter != "Q1"
                        else "Q1 YTD"
                    ),
                })

        # Revenue, operating income, and segment metrics
        for metric in OPERATING_METRICS:
            annual = get_single_value(
                raw, year, "FY", metric, "annual"
            )
            reported_q4 = get_single_value(
                raw, year, "Q4", metric, "quarter"
            )

            quarter_values = {}

            for quarter in ["Q1", "Q2", "Q3"]:
                quarter_values[quarter] = get_single_value(
                    raw, year, quarter, metric, "quarter"
                )

            if pd.notna(reported_q4):
                quarter_values["Q4"] = reported_q4
                q4_method = "Reported quarter"
            # Derive Q4 only when the annual and all three
            # standalone quarterly figures are available.
            elif (
                pd.notna(annual)
                and all(
                    pd.notna(quarter_values[q])
                    for q in ["Q1", "Q2", "Q3"]
                )
            ):
                quarter_values["Q4"] = (
                    annual
                    - quarter_values["Q1"]
                    - quarter_values["Q2"]
                    - quarter_values["Q3"]
                )
                q4_method = "Annual less Q1-Q3"
            else:
                quarter_values["Q4"] = np.nan
                q4_method = "Unverified / incomplete"

            for quarter in quarters:
                if year == END_YEAR and quarter in ["Q3", "Q4"]:
                    continue

                value = quarter_values[quarter]

                records.append({
                    "year": year,
                    "quarter": quarter,
                    "period_id": f"{year}{quarter}",
                    "metric": metric,
                    "value": value,
                    "method": (
                        q4_method if quarter == "Q4"
                        else "Reported quarter"
                    ),
                })

    quarterly = pd.DataFrame(records)

    # Convert source values from US$ millions to US$ billions.
    quarterly["value_bn"] = (
        quarterly["value"] / MILLIONS_PER_BILLION
    )

    return quarterly


# ==================================================
# 4. Build annual data for the long-term revenue chart
# ==================================================

def build_annual_data(raw):
    annual = raw[
        (raw["period"] == "FY")
        & (raw["basis"] == "annual")
        & raw["metric"].isin(OPERATING_METRICS | CASH_FLOW_METRICS)
    ].copy()

    annual["value_bn"] = (
        annual["value"] / MILLIONS_PER_BILLION
    )

    return annual


# ==================================================
# 5. Plot formatting helpers
# ==================================================

def format_axis(ax, ylabel="US$ billions"):
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(
        mtick.StrMethodFormatter("{x:,.0f}")
    )


def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(
        CHART_DIR / filename,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def quarterly_series(quarterly, metric):
    subset = quarterly[
        quarterly["metric"] == metric
    ].copy()

    subset = subset.sort_values(
        ["year", "quarter"]
    )

    return subset


# ==================================================
# 6. Chart 1: Revenue trend
# ==================================================

def plot_revenue(quarterly, annual):
    q = quarterly_series(quarterly, "revenue")

    fig, ax = plt.subplots(figsize=(11, 5.5))

    if not q.empty:
        ax.plot(
            q["period_id"],
            q["value_bn"],
            marker="o",
            linewidth=2,
            label="Quarterly revenue",
        )

        # Mark derived Q4 observations differently.
        derived = q[
            q["method"] == "Annual less Q1-Q3"
        ]
        if not derived.empty:
            ax.scatter(
                derived["period_id"],
                derived["value_bn"],
                marker="s",
                s=45,
                label="Q4 derived from annual totals",
            )

    ax.set_title("Alphabet Revenue Trend")
    ax.set_xlabel("Fiscal quarter")
    ax.tick_params(axis="x", rotation=55)
    format_axis(ax)
    ax.legend(frameon=False)

    save_figure(fig, "01_revenue_quarterly.png")

    # Annual chart provides longer-term context.
    a = annual[annual["metric"] == "revenue"].sort_values(
        "fiscal_year"
    )

    if not a.empty:
        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(
            a["fiscal_year"].astype(str),
            a["value_bn"],
            marker="o",
            linewidth=2,
        )

        ax.set_title("Alphabet Annual Revenue: FY2019–FY2025")
        ax.set_xlabel("Fiscal year")
        format_axis(ax)

        save_figure(fig, "01b_revenue_annual.png")


# ==================================================
# 7. Chart 2: Capital expenditure trend
# ==================================================

def plot_capex(quarterly, annual):
    q = quarterly_series(quarterly, "capex")

    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.plot(
        q["period_id"],
        q["value_bn"],
        marker="o",
        linewidth=2,
        label="Quarterly CapEx",
    )

    ax.set_title("Alphabet Capital Expenditure Trend")
    ax.set_xlabel("Fiscal quarter")
    ax.tick_params(axis="x", rotation=55)
    format_axis(ax)
    ax.legend(frameon=False)

    save_figure(fig, "02_capex_quarterly.png")

    a = annual[annual["metric"] == "capex"].sort_values(
        "fiscal_year"
    )

    if not a.empty:
        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(
            a["fiscal_year"].astype(str),
            a["value_bn"],
            marker="o",
            linewidth=2,
        )

        ax.set_title("Alphabet Annual CapEx: FY2019–FY2025")
        ax.set_xlabel("Fiscal year")
        format_axis(ax)

        save_figure(fig, "02b_capex_annual.png")


# ==================================================
# 8. Chart 3: Free cash flow trend
# ==================================================

def plot_fcf(quarterly, annual):
    q = quarterly_series(
        quarterly, "operating_cash_flow"
    ).rename(columns={"value_bn": "ocf_bn"})

    c = quarterly_series(
        quarterly, "capex"
    ).rename(columns={"value_bn": "capex_bn"})

    fcf = q[["period_id", "year", "quarter", "ocf_bn"]].merge(
        c[["period_id", "capex_bn"]],
        on="period_id",
        how="outer",
    )

    fcf["fcf_bn"] = fcf["ocf_bn"] - fcf["capex_bn"]

    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.plot(
        fcf["period_id"],
        fcf["fcf_bn"],
        marker="o",
        linewidth=2,
        label="Calculated FCF",
    )

    # A zero line makes negative FCF quarters immediately visible.
    ax.axhline(0, linewidth=1, linestyle="--")

    ax.set_title("Alphabet Free Cash Flow Trend")
    ax.set_xlabel("Fiscal quarter")
    ax.tick_params(axis="x", rotation=55)
    format_axis(ax)
    ax.legend(frameon=False)

    save_figure(fig, "03_fcf_quarterly.png")

    # Annual FCF calculated using annual OCF - annual CapEx.
    ocf_a = annual[
        annual["metric"] == "operating_cash_flow"
    ][["fiscal_year", "value_bn"]].rename(
        columns={"value_bn": "ocf_bn"}
    )

    capex_a = annual[
        annual["metric"] == "capex"
    ][["fiscal_year", "value_bn"]].rename(
        columns={"value_bn": "capex_bn"}
    )

    annual_fcf = ocf_a.merge(
        capex_a,
        on="fiscal_year",
        how="outer",
    )

    annual_fcf["fcf_bn"] = (
        annual_fcf["ocf_bn"] - annual_fcf["capex_bn"]
    )

    annual_fcf = annual_fcf.sort_values("fiscal_year")

    if not annual_fcf.empty:
        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(
            annual_fcf["fiscal_year"].astype(str),
            annual_fcf["fcf_bn"],
            marker="o",
            linewidth=2,
        )

        ax.set_title("Alphabet Annual FCF: FY2019–FY2025")
        ax.set_xlabel("Fiscal year")
        ax.axhline(0, linewidth=1, linestyle="--")
        format_axis(ax)

        save_figure(fig, "03b_fcf_annual.png")


# ==================================================
# 9. Chart 4: Google Cloud profitability
# ==================================================

def plot_cloud_profitability(quarterly):
    revenue = quarterly_series(
        quarterly, "google_cloud_revenue"
    ).rename(columns={"value_bn": "cloud_revenue_bn"})

    income = quarterly_series(
        quarterly, "google_cloud_operating_income"
    ).rename(columns={"value_bn": "cloud_operating_income_bn"})

    cloud = revenue[
        ["period_id", "cloud_revenue_bn", "method"]
    ].merge(
        income[
            ["period_id", "cloud_operating_income_bn"]
        ],
        on="period_id",
        how="outer",
    )

    cloud["cloud_operating_margin"] = (
        cloud["cloud_operating_income_bn"]
        / cloud["cloud_revenue_bn"].where(
            cloud["cloud_revenue_bn"] != 0
        )
    )

    cloud = cloud.sort_values("period_id")

    # Revenue and operating income share the same units.
    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.plot(
        cloud["period_id"],
        cloud["cloud_revenue_bn"],
        marker="o",
        linewidth=2,
        label="Google Cloud revenue",
    )

    ax.plot(
        cloud["period_id"],
        cloud["cloud_operating_income_bn"],
        marker="s",
        linewidth=2,
        label="Google Cloud operating income",
    )

    ax.axhline(0, linewidth=1, linestyle="--")
    ax.set_title("Google Cloud Revenue and Operating Income")
    ax.set_xlabel("Fiscal quarter")
    ax.tick_params(axis="x", rotation=55)
    format_axis(ax)
    ax.legend(frameon=False)

    save_figure(fig, "04_cloud_profitability.png")

    # Separate margin chart avoids confusing dollars and percentages.
    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.plot(
        cloud["period_id"],
        cloud["cloud_operating_margin"] * 100,
        marker="o",
        linewidth=2,
    )

    ax.axhline(0, linewidth=1, linestyle="--")
    ax.set_title("Google Cloud Operating Margin")
    ax.set_xlabel("Fiscal quarter")
    ax.set_ylabel("Operating margin (%)")
    ax.yaxis.set_major_formatter(
        mtick.PercentFormatter(xmax=100)
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=55)

    save_figure(fig, "04b_cloud_margin.png")

    cloud.to_csv(
        CHART_DIR / "cloud_profitability_chart_data.csv",
        index=False,
    )


# ==================================================
# 10. Export plotting data and run
# ==================================================

def main():
    raw = load_source_data()

    quarterly = build_quarterly_data(raw)
    annual = build_annual_data(raw)

    quarterly.to_csv(
        CHART_DIR / "quarterly_chart_data.csv",
        index=False,
    )

    annual.to_csv(
        CHART_DIR / "annual_chart_data.csv",
        index=False,
    )

    plot_revenue(quarterly, annual)
    plot_capex(quarterly, annual)
    plot_fcf(quarterly, annual)
    plot_cloud_profitability(quarterly)

    print(f"Charts saved to: {CHART_DIR.resolve()}")
    print("\nGenerated chart files:")
    for file in sorted(CHART_DIR.glob("*.png")):
        print(f"  {file.name}")

    # Display missing observations that may affect charts.
    required = [
        "revenue",
        "capex",
        "operating_cash_flow",
        "google_cloud_revenue",
        "google_cloud_operating_income",
    ]

    print("\nMissing quarterly observations:")
    for metric in required:
        s = quarterly[quarterly["metric"] == metric]
        missing = s.loc[s["value"].isna(), "period_id"].tolist()

        if missing:
            print(f"  {metric}: {', '.join(missing)}")


if __name__ == "__main__":
    main()
