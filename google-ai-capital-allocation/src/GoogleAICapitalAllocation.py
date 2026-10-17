
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs/data"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

RAW_FILE = DATA_DIR / "sec_observations.csv"

CASH_FLOW_METRICS = {
    "operating_cash_flow",
    "capex",
    "ppe_depreciation",
}

QUARTERS = [
    f"{year}Q{quarter}"
    for year in range(2023, 2027)
    for quarter in range(1, 5)
]
QUARTERS = [
    q for q in QUARTERS
    if q <= "2026Q2"
]

ANNUAL_YEARS = range(2019, 2026)

# Each row is one observation from an official SEC filing.
# Required columns:
# fiscal_year, period, metric, value, basis, source_url
#
# period: Q1, Q2, Q3, or FY
# basis: quarter, ytd, or annual
#
# Example:
# 2023,Q1,operating_cash_flow,23509,ytd,https://...
#
# Use US$ millions for all numeric values.
# Leave unverified observations out; never enter zero
# to represent missing data.


# --------------------------------------------------
# 2. Load and validate SEC source observations
# --------------------------------------------------

def load_observations(path=RAW_FILE):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file: {path}\n"
            "Create the SEC observation CSV first."
        )

    df = pd.read_csv(path)

    required = {
        "fiscal_year",
        "period",
        "metric",
        "value",
        "basis",
        "source_url",
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

    valid_periods = {"Q1", "Q2", "Q3", "Q4", "FY"}
    valid_basis = {"quarter", "ytd", "annual"}

    if not set(df["period"]).issubset(valid_periods):
        raise ValueError("Unexpected period value.")

    if not set(df["basis"]).issubset(valid_basis):
        raise ValueError("Unexpected basis value.")

    # A repeated key can silently double-count data.
    key = ["fiscal_year", "period", "metric", "basis"]
    duplicates = df[df.duplicated(key, keep=False)]

    if not duplicates.empty:
        raise ValueError(
            "Duplicate observations found:\n"
            + duplicates[key].to_string(index=False)
        )

    # Non-null observations need traceable sources.
    bad_sources = df[
        df["value"].notna()
        & (
            df["source_url"].isna()
            | df["source_url"].str.strip().eq("")
        )
    ]

    if not bad_sources.empty:
        raise ValueError(
            "Numeric observations are missing source URLs."
        )

    return df


# --------------------------------------------------
# 3. Convert YTD cash flows into standalone quarters
# --------------------------------------------------

def derive_quarterly_cash_flows(raw):
    """
    For cash-flow metrics:
      Q1 = Q1 YTD
      Q2 = Q2 YTD - Q1 YTD
      Q3 = Q3 YTD - Q2 YTD
      Q4 = FY annual - Q3 YTD

    Missing inputs remain NaN. They are never treated as zero.
    """

    records = []

    for metric in CASH_FLOW_METRICS:
        m = raw[raw["metric"] == metric]

        for year in sorted(m["fiscal_year"].unique()):
            year_data = m[m["fiscal_year"] == year]

            ytd_values = {}
            for q in ["Q1", "Q2", "Q3"]:
                rows = year_data[
                    (year_data["period"] == q)
                    & (year_data["basis"] == "ytd")
                ]
                ytd_values[q] = (
                    rows.iloc[0]["value"]
                    if len(rows) == 1
                    else np.nan
                )

            annual_rows = year_data[
                (year_data["period"] == "FY")
                & (year_data["basis"] == "annual")
            ]
            annual_value = (
                annual_rows.iloc[0]["value"]
                if len(annual_rows) == 1
                else np.nan
            )

            q_values = {
                "Q1": ytd_values["Q1"],
                "Q2": (
                    ytd_values["Q2"] - ytd_values["Q1"]
                    if pd.notna(ytd_values["Q2"])
                    and pd.notna(ytd_values["Q1"])
                    else np.nan
                ),
                "Q3": (
                    ytd_values["Q3"] - ytd_values["Q2"]
                    if pd.notna(ytd_values["Q3"])
                    and pd.notna(ytd_values["Q2"])
                    else np.nan
                ),
                "Q4": (
                    annual_value - ytd_values["Q3"]
                    if pd.notna(annual_value)
                    and pd.notna(ytd_values["Q3"])
                    else np.nan
                ),
            }

            for quarter, value in q_values.items():
                records.append({
                    "fiscal_year": year,
                    "quarter": quarter,
                    "period_id": f"{year}{quarter}",
                    "metric": metric,
                    "value": value,
                })

    return pd.DataFrame(records)


# --------------------------------------------------
# 4. Build reported quarterly operating metrics
# --------------------------------------------------

def build_quarterly_operating_metrics(raw):
    """
    Revenue, operating income, Cloud metrics and Search
    revenue should use standalone quarterly observations.
    Do not subtract YTD values for these income-statement
    metrics when the filing already reports a quarter.
    """

    q = raw[
        (raw["basis"] == "quarter")
        & raw["period"].isin(["Q1", "Q2", "Q3", "Q4"])
    ].copy()

    q["period_id"] = (
        q["fiscal_year"].astype(str) + q["period"]
    )

    q = q.rename(columns={"value": "value_usd_millions"})

    return q[
        [
            "fiscal_year",
            "period",
            "period_id",
            "metric",
            "value_usd_millions",
            "source_url",
        ]
    ]


# --------------------------------------------------
# 5. Calculate FCF and analytical ratios
# --------------------------------------------------

def build_financial_metrics(cash_flows, operating):
    cf = cash_flows.pivot(
        index="period_id",
        columns="metric",
        values="value",
    )

    for metric in CASH_FLOW_METRICS:
        if metric not in cf.columns:
            cf[metric] = np.nan

    cf["fcf"] = (
        cf["operating_cash_flow"] - cf["capex"]
    )

    # Ratio: retain missing values if the denominator is
    # unavailable or zero.
    cf["capex_to_ocf"] = (
        cf["capex"] / cf["operating_cash_flow"].where(
            cf["operating_cash_flow"] != 0
        )
    )

    op = operating.pivot(
        index="period_id",
        columns="metric",
        values="value_usd_millions",
    )

    result = cf.join(op, how="outer")

    if "revenue" in result.columns:
        result["operating_margin"] = (
            result["operating_income"]
            / result["revenue"]
            if "operating_income" in result.columns
            else np.nan
        )

    return result.reset_index()


# --------------------------------------------------
# 6. Reconciliation checks
# --------------------------------------------------

def reconciliation_checks(raw, quarterly_cf, operating):
    checks = []

    def add_check(name, expected, actual, note=""):
        if pd.isna(expected) or pd.isna(actual):
            status = "INCOMPLETE"
            difference = np.nan
        else:
            difference = actual - expected
            status = (
                "PASS"
                if np.isclose(
                    expected, actual, atol=0.002, rtol=0
                )
                else "FAIL"
            )

        checks.append({
            "check": name,
            "expected_usd_millions": expected,
            "actual_usd_millions": actual,
            "difference_usd_millions": difference,
            "status": status,
            "note": note,
        })

    # Annual cash flow = sum of standalone quarters.
    for metric in CASH_FLOW_METRICS:
        annual = raw[
            (raw["metric"] == metric)
            & (raw["period"] == "FY")
            & (raw["basis"] == "annual")
        ]

        for _, row in annual.iterrows():
            year = row["fiscal_year"]

            q = quarterly_cf[
                (quarterly_cf["fiscal_year"] == year)
                & (quarterly_cf["metric"] == metric)
            ]

            expected = row["value"]
            actual = (
                q["value"].sum()
                if len(q) == 4 and q["value"].notna().all()
                else np.nan
            )

            add_check(
                f"{year} annual tie-out: {metric}",
                expected,
                actual,
                "Annual filing vs sum of standalone quarters",
            )

    # Quarterly operating metrics = annual values, where
    # both the annual and all four quarterly observations exist.
    annual_metrics = raw[
        (raw["period"] == "FY")
        & (raw["basis"] == "annual")
    ]

    quarter_metrics = raw[
        (raw["period"].isin(["Q1", "Q2", "Q3"]))
        & (raw["basis"] == "quarter")
    ]

    for _, row in annual_metrics.iterrows():
        metric = row["metric"]
        year = row["fiscal_year"]

        qs = quarter_metrics[
            (quarter_metrics["metric"] == metric)
            & (quarter_metrics["fiscal_year"] == year)
        ]

        if len(qs) == 3 and qs["value"].notna().all():
            # Q4 may be reported separately or derived from
            # annual less Q1-Q3. Derive it only if all are known.
            q4 = row["value"] - qs["value"].sum()
            actual = qs["value"].sum() + q4
        else:
            actual = np.nan

        add_check(
            f"{year} annual arithmetic: {metric}",
            row["value"],
            actual,
            "Arithmetic completeness check; not independent validation",
        )

    return pd.DataFrame(checks)


# --------------------------------------------------
# 7. Run the project pipeline
# --------------------------------------------------

def main():
    raw = load_observations()

    quarterly_cf = derive_quarterly_cash_flows(raw)
    operating = build_quarterly_operating_metrics(raw)

    metrics = build_financial_metrics(
        quarterly_cf, operating
    )

    checks = reconciliation_checks(
        raw, quarterly_cf, operating
    )

    raw.to_csv(
        OUTPUT_DIR / "source_observations.csv",
        index=False,
    )
    quarterly_cf.to_csv(
        OUTPUT_DIR / "quarterly_cash_flows.csv",
        index=False,
    )
    operating.to_csv(
        OUTPUT_DIR / "quarterly_operating_metrics.csv",
        index=False,
    )
    metrics.to_csv(
        OUTPUT_DIR / "quarterly_financial_metrics.csv",
        index=False,
    )
    checks.to_csv(
        OUTPUT_DIR / "reconciliation_checks.csv",
        index=False,
    )

    print("\nReconciliation summary")
    print(checks["status"].value_counts(dropna=False))

    failures = checks[checks["status"] == "FAIL"]
    if not failures.empty:
        print("\nChecks requiring investigation:")
        print(failures.to_string(index=False))

    print(f"\nOutput files saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
