from pathlib import Path

import numpy as np
import pandas as pd

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs/data"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILE = DATA_DIR / "sec_observations.csv"
ANNUAL_RAW_FILE = DATA_DIR / "Annual_dataset_FY2019_FY2025.csv"

CASH_FLOW_METRICS = {
    "operating_cash_flow",
    "capex",
    "ppe_depreciation",
}

CORE_METRICS = [
    "revenue", "operating_income", "operating_cash_flow", "capex",
    "ppe_depreciation", "google_cloud_revenue", "google_cloud_operating_income",
    "search_other_revenue"
]

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

PERIOD_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "FY": 5}

# Each row is one observation from an official SEC filing.
# Required columns:
# fiscal_year, period, metric, value, basis, source_url
#
# period: Q1, Q2, Q3, or FY
# basis: quarter, ytd, or annual


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

    key = ["fiscal_year", "period", "metric", "basis"]
    duplicates = df[df.duplicated(key, keep=False)]

    if not duplicates.empty:
        raise ValueError(
            "Duplicate observations found:\n"
            + duplicates[key].to_string(index=False)
        )

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
# 2.1 Load and process Annual dataset (FY2019-FY2025) [NEW ADDITION]
# --------------------------------------------------

def load_annual_dataset(path=ANNUAL_RAW_FILE):
    if not path.exists():
        return None

    # df = None
    # for enc in ['cp932', 'utf-8-sig', 'latin1']:
    #     try:
    #         df = pd.read_csv(path, encoding=enc)
    #         break
    #     except Exception:
    #         continue

    # if df is None:
    #     return None
    df = pd.read_csv(path)
    # Clean column names.
   # df.columns = [c.replace('ｹ', '').strip() for c in df.columns]
    rename_map = {
        'Fiscal year': 'fiscal_year',
        'Revenue': 'revenue',
        'Operating income': 'operating_income',
        'Operating cash flow': 'operating_cash_flow',
        'CapEx': 'capex',
        'PPE depreciation': 'ppe_depreciation',
        'FCF': 'fcf',
        'Cloud revenue': 'google_cloud_revenue',
        'Cloud operating income': 'google_cloud_operating_income',
        'Search & other': 'search_other_revenue'
    }
  #  df = df.rename(columns=rename_map)

    df["fiscal_year"] = pd.to_numeric(df["fiscal_year"], errors="coerce").astype(int)
    for col in rename_map.values():
        if col != 'fiscal_year' and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("fiscal_year").reset_index(drop=True)


def build_annual_dataset_metrics(annual_df):
    if annual_df is None or annual_df.empty:
        return pd.DataFrame()
    
    out = annual_df.copy()
    out["operating_margin"] = out["operating_income"] / out["revenue"].replace(0, np.nan)
    if "google_cloud_operating_income" in out.columns and "google_cloud_revenue" in out.columns:
        out["google_cloud_operating_margin"] = out["google_cloud_operating_income"] / out["google_cloud_revenue"].replace(0, np.nan)
    if "search_other_revenue" in out.columns and "revenue" in out.columns:
        out["search_other_revenue_share"] = out["search_other_revenue"] / out["revenue"].replace(0, np.nan)
    out["capex_to_revenue"] = out["capex"] / out["revenue"].replace(0, np.nan)
    out["capex_to_operating_cash_flow"] = out["capex"] / out["operating_cash_flow"].replace(0, np.nan)
    if "fcf" in out.columns and "revenue" in out.columns:
        out["fcf_margin"] = out["fcf"] / out["revenue"].replace(0, np.nan)
    out["ppe_depreciation_to_revenue"] = out["ppe_depreciation"] / out["revenue"].replace(0, np.nan)

    for col in ["revenue", "operating_income", "operating_cash_flow", "capex", "google_cloud_revenue"]:
        if col in out.columns:
            out[f"{col}_growth_yoy"] = out[col].pct_change()

    return out


# --------------------------------------------------
# 3. Convert YTD cash flows into standalone quarters
# --------------------------------------------------

def derive_quarterly_cash_flows(raw):
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
# 6.5 Build Forecast Assumptions & Book (Step 3 Constraints)
# --------------------------------------------------

def finite_values(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df:
        return np.array([], dtype=float)
    vals = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    return vals[np.isfinite(vals)]


def range_stats(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"n": 0, "min": np.nan, "p25": np.nan, "median": np.nan, "p75": np.nan, "max": np.nan}
    return {
        "n": int(len(values)),
        "min": float(np.min(values)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values))
    }


def assumption_row(name, metric_col, data, unit, proposed_use, limitations, extra_evidence=None):
    values = finite_values(data, metric_col)
    s = range_stats(values)
    if s["n"] >= 3:
        low, base, high = s["p25"], s["median"], s["p75"]
        range_method = "Historical interquartile range (Q1-Q3) and median; analyst must assess structural breaks before use."
    elif s["n"] > 0:
        low, base, high = s["min"], s["median"], s["max"]
        range_method = "Sparse-history min/median/max; insufficient observations for a robust range. Do not treat as forecast."
    else:
        low = base = high = np.nan
        range_method = "No verified observations available; leave blank until sourced."
    return {
        "assumption": name, "historical_metric": metric_col, "unit": unit,
        "historical_observations_n": s["n"], "historical_min": s["min"], "historical_p25": s["p25"],
        "historical_median": s["median"], "historical_p75": s["p75"], "historical_max": s["max"],
        "candidate_low": low, "candidate_base": base, "candidate_high": high,
        "range_method": range_method, "proposed_use_in_step3": proposed_use,
        "limitations": limitations, "additional_evidence_needed": extra_evidence or "Review SEC filing commentary, management guidance and capacity / demand evidence; document the source and date."
    }


def build_period_metrics(obs: pd.DataFrame, flags: list[dict]) -> pd.DataFrame:
    rows = []
    cash_flow_metrics = {"operating_cash_flow", "capex", "ppe_depreciation"}
    accepted_ytd_basis = {"ytd", "year_to_date", "year-to-date"}

    for year in sorted(obs.fiscal_year.unique()):
        year_obs = obs[obs.fiscal_year.eq(year)]
        for period in ["Q1", "Q2", "Q3", "Q4", "FY"]:
            g = year_obs[year_obs.period.eq(period)]
            if g.empty:
                if period != "Q4" or not year_obs.period.isin(["Q1", "Q2", "Q3", "FY"]).any():
                    continue
                g = year_obs.iloc[0:0]

            rec = {"fiscal_year": int(year), "period": period, "period_order": PERIOD_ORDER[period]}
            for metric in CORE_METRICS:
                metric_rows = g[g.metric.eq(metric)]
                if len(metric_rows) == 1:
                    rec[metric] = float(metric_rows.iloc[0].value) if pd.notna(metric_rows.iloc[0].value) else np.nan
                else:
                    rec[metric] = np.nan

            if period in {"Q1", "Q2", "Q3", "Q4"}:
                for metric in cash_flow_metrics:
                    current_rows = g[g.metric.eq(metric)]
                    current_value = float(current_rows.iloc[0].value) if len(current_rows) == 1 and pd.notna(current_rows.iloc[0].value) else np.nan
                    current_basis = str(current_rows.iloc[0].basis).lower() if len(current_rows) == 1 else ""

                    if current_basis in accepted_ytd_basis:
                        if period == "Q1":
                            rec[metric] = current_value
                        else:
                            previous_period = f"Q{int(period[-1]) - 1}"
                            previous_rows = year_obs[(year_obs.period.eq(previous_period)) & (year_obs.metric.eq(metric))]
                            if len(previous_rows) == 1 and str(previous_rows.iloc[0].basis).lower() in accepted_ytd_basis and pd.notna(current_value) and pd.notna(previous_rows.iloc[0].value):
                                rec[metric] = current_value - float(previous_rows.iloc[0].value)
                            else:
                                rec[metric] = np.nan
                                flags.append({"severity": "WARNING", "flag": "cannot_derive_standalone_quarter", "fiscal_year": int(year), "period": period, "metric": metric, "detail": "YTD value present but the prior YTD observation is missing; standalone quarter left blank."})
                    elif period == "Q4":
                        annual_rows = year_obs[(year_obs.period.eq("FY")) & (year_obs.metric.eq(metric)) & (year_obs.basis.eq("annual"))]
                        q3_rows = year_obs[(year_obs.period.eq("Q3")) & (year_obs.metric.eq(metric)) & (year_obs.basis.isin(accepted_ytd_basis))]
                        if len(annual_rows) == 1 and len(q3_rows) == 1 and pd.notna(annual_rows.iloc[0].value) and pd.notna(q3_rows.iloc[0].value):
                            rec[metric] = float(annual_rows.iloc[0].value) - float(q3_rows.iloc[0].value)
                        elif current_basis in {"quarter"}:
                            rec[metric] = current_value
                        else:
                            rec[metric] = np.nan
                    elif current_basis == "quarter":
                        rec[metric] = current_value
                    elif current_basis == "annual":
                        rec[metric] = np.nan

            rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["fiscal_year", "period_order"]).reset_index(drop=True)

    out["operating_margin"] = out["operating_income"] / out["revenue"].replace(0, np.nan)
    out["google_cloud_operating_margin"] = out["google_cloud_operating_income"] / out["google_cloud_revenue"].replace(0, np.nan)
    out["search_other_revenue_share"] = out["search_other_revenue"] / out["revenue"].replace(0, np.nan)
    out["capex_to_revenue"] = out["capex"] / out["revenue"].replace(0, np.nan)
    out["capex_to_operating_cash_flow"] = out["capex"] / out["operating_cash_flow"].replace(0, np.nan)
    out["fcf_ocf_less_capex"] = out["operating_cash_flow"] - out["capex"]
    out["fcf_margin_ocf_less_capex"] = out["fcf_ocf_less_capex"] / out["revenue"].replace(0, np.nan)
    out["ppe_depreciation_to_revenue"] = out["ppe_depreciation"] / out["revenue"].replace(0, np.nan)
    return out


def build_annual_trends(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics


def build_assumption_book_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    fy_df = raw[raw["period"] == "FY"].pivot(index="fiscal_year", columns="metric", values="value")
    if not fy_df.empty:
        fy_df = fy_df.sort_index()
        for col in fy_df.columns:
            fy_df[f"{col}_growth_yoy"] = fy_df[col].pct_change(fill_method=None)
        if "operating_income" in fy_df.columns and "revenue" in fy_df.columns:
            fy_df["operating_margin"] = fy_df["operating_income"] / fy_df["revenue"]
        if "capex" in fy_df.columns and "revenue" in fy_df.columns:
            fy_df["capex_to_revenue"] = fy_df["capex"] / fy_df["revenue"]
        if "ppe_depreciation" in fy_df.columns and "revenue" in fy_df.columns:
            fy_df["ppe_depreciation_to_revenue"] = fy_df["ppe_depreciation"] / fy_df["revenue"]

    annual = fy_df.reset_index()
    rows = []
    rows.append(assumption_row(
        "Group revenue growth", "revenue_growth_yoy", annual, "% per year",
        "Starting evidence for 2027-2030 consolidated revenue growth. Base case should reflect normalized growth and current business mix.",
        "Reported total revenue is not AI revenue. Historical growth may reflect acquisitions, FX, macro and business mix.",
        "Segment trends, Search & other growth, Cloud growth, disclosed AI product adoption and management commentary."
    ))
    rows.append(assumption_row(
        "Group operating margin", "operating_margin", annual, "% of revenue",
        "Constrain consolidated operating-margin assumptions; model margin expansion only when supported by cost leverage.",
        "Historical GAAP operating margin includes all businesses and corporate costs; it cannot isolate AI profitability.",
        "R&D and infrastructure cost trends, stock-based compensation, restructuring, and management commentary."
    ))
    rows.append(assumption_row(
        "CapEx intensity", "capex_to_revenue", annual, "% of revenue",
        "Anchor the range for 2027-2030 CapEx / revenue, then adjust for disclosed infrastructure build-out and management guidance.",
        "Total CapEx is not AI-only CapEx. A structural AI infrastructure cycle can make the past range a poor forward guide.",
        "Most recent annual and quarterly CapEx guidance, data-center build-out, servers/networking commitments."
    ))
    rows.append(assumption_row(
        "PPE depreciation intensity", "ppe_depreciation_to_revenue", annual, "% of revenue",
        "Use as a historical diagnostic only; forecast D&A separately with asset-life and in-service timing assumptions.",
        "PPE depreciation may not equal total D&A; depreciation lags CapEx and depends on useful lives.",
        "SEC note on useful lives, total depreciation and amortization, gross PP&E."
    ))

    latest_year = int(annual.fiscal_year.max()) if not annual.empty else None
    for r in rows:
        r["annual_data_through"] = latest_year
        r["status"] = "EVIDENCE_AVAILABLE" if r.get("historical_observations_n", 0) > 0 else "REQUIRES_ADDITIONAL_EVIDENCE"
        r["analyst_override_required"] = True
    return pd.DataFrame(rows), annual


def export_forecast_assumption_book_excel(
    assumptions: pd.DataFrame,
    annual_trends: pd.DataFrame,
    quality_flags: pd.DataFrame,
    output_path: Path
) -> None:
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    font_family = "Segoe UI"
    header_fill = PatternFill(start_color="1B365D", end_color="1B365D", fill_type="solid")
    header_font = Font(name=font_family, size=11, bold=True, color="FFFFFF")
    
    zebra_fill = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
    accent_fill = PatternFill(start_color="FEF3C7", end_color="FEF3C7", fill_type="solid")
    
    status_green = PatternFill(start_color="D1FAE5", end_color="D1FAE5", fill_type="solid")
    status_orange = PatternFill(start_color="FFEDD5", end_color="FFEDD5", fill_type="solid")

    thin_border_side = Side(style="thin", color="CBD5E1")
    border_all = Border(left=thin_border_side, right=thin_border_side, top=thin_border_side, bottom=thin_border_side)
    
    align_left = Alignment(horizontal="left", vertical="center", wrap_text=True)
    align_right = Alignment(horizontal="right", vertical="center")
    align_center = Alignment(horizontal="center", vertical="center")

    # Sheet 1: Forecast Assumptions
    ws_assump = wb.create_sheet(title="Forecast Assumptions")
    ws_assump.views.sheetView[0].showGridLines = True

    ws_assump.merge_cells("A1:N1")
    title_cell = ws_assump["A1"]
    title_cell.value = "Alphabet AI Capital Allocation — Forecast Assumption Book (Step 3 Constraints)"
    title_cell.font = Font(name=font_family, size=14, bold=True, color="1B365D")
    title_cell.alignment = align_left
    ws_assump.row_dimensions[1].height = 28

    headers_assump = list(assumptions.columns)
    ws_assump.append([])
    ws_assump.append(headers_assump)
    ws_assump.row_dimensions[3].height = 26

    for col_num, col_name in enumerate(headers_assump, start=1):
        cell = ws_assump.cell(row=3, column=col_num)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = align_center if col_num > 3 else align_left
        cell.border = border_all

    for r_idx, row_data in enumerate(assumptions.itertuples(index=False), start=4):
        ws_assump.row_dimensions[r_idx].height = 36
        is_even = (r_idx % 2 == 0)
        row_fill = zebra_fill if is_even else PatternFill(fill_type=None)

        unit_str = str(getattr(row_data, "unit", "")).lower()
        is_pct = "%" in unit_str

        for c_idx, val in enumerate(row_data, start=1):
            cell = ws_assump.cell(row=r_idx, column=c_idx)
            col_name = headers_assump[c_idx - 1]

            if pd.isna(val):
                cell.value = "-"
                cell.alignment = align_center
            else:
                cell.value = val

            cell.font = Font(name=font_family, size=10)
            cell.border = border_all
            cell.fill = row_fill

            if col_name in ["historical_min", "historical_p25", "historical_median", "historical_p75", "historical_max", "candidate_low", "candidate_base", "candidate_high"]:
                cell.alignment = align_right
                if isinstance(val, (int, float)) and not pd.isna(val):
                    cell.number_format = "0.0%" if is_pct else "#,##0.0"
            elif col_name in ["historical_observations_n", "annual_data_through"]:
                cell.alignment = align_center
                if isinstance(val, (int, float)) and not pd.isna(val):
                    cell.number_format = "0"
            elif col_name == "status":
                cell.alignment = align_center
                if val == "EVIDENCE_AVAILABLE":
                    cell.fill = status_green
                else:
                    cell.fill = status_orange
            elif col_name == "analyst_override_required":
                cell.alignment = align_center
                cell.fill = accent_fill
            else:
                cell.alignment = align_left

    ws_assump.freeze_panes = "C4"

    # Sheet 2: Annual Trends
    ws_trends = wb.create_sheet(title="Annual Trends")
    ws_trends.views.sheetView[0].showGridLines = True

    ws_trends.merge_cells("A1:H1")
    t_cell = ws_trends["A1"]
    t_cell.value = "Alphabet Historical Annual Financial Metrics & Trends"
    t_cell.font = Font(name=font_family, size=14, bold=True, color="1B365D")
    ws_trends.row_dimensions[1].height = 28

    if not annual_trends.empty:
        headers_trends = list(annual_trends.columns)
        ws_trends.append([])
        ws_trends.append(headers_trends)
        ws_trends.row_dimensions[3].height = 24

        for col_num in range(1, len(headers_trends) + 1):
            cell = ws_trends.cell(row=3, column=col_num)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = align_center
            cell.border = border_all

        for r_idx, row_data in enumerate(annual_trends.itertuples(index=False), start=4):
            ws_trends.row_dimensions[r_idx].height = 20
            row_fill = zebra_fill if (r_idx % 2 == 0) else PatternFill(fill_type=None)
            for c_idx, val in enumerate(row_data, start=1):
                cell = ws_trends.cell(row=r_idx, column=c_idx)
                col_name = headers_trends[c_idx - 1]
                cell.font = Font(name=font_family, size=10)
                cell.border = border_all
                cell.fill = row_fill

                if pd.isna(val):
                    cell.value = "-"
                    cell.alignment = align_center
                else:
                    cell.value = val
                    if "margin" in col_name or "growth" in col_name or "share" in col_name or "to_" in col_name:
                        cell.number_format = "0.0%"
                        cell.alignment = align_right
                    elif col_name == "fiscal_year":
                        cell.number_format = "0"
                        cell.alignment = align_center
                    else:
                        cell.number_format = "#,##0.0"
                        cell.alignment = align_right

    ws_trends.freeze_panes = "B4"

    # Sheet 3: Data Quality Flags
    ws_flags = wb.create_sheet(title="Data Quality Flags")
    ws_flags.views.sheetView[0].showGridLines = True

    ws_flags.merge_cells("A1:F1")
    q_cell = ws_flags["A1"]
    q_cell.value = "Data Quality Checks & Reconciliation Flags"
    q_cell.font = Font(name=font_family, size=14, bold=True, color="1B365D")
    ws_flags.row_dimensions[1].height = 28

    if not quality_flags.empty:
        headers_flags = list(quality_flags.columns)
        ws_flags.append([])
        ws_flags.append(headers_flags)
        ws_flags.row_dimensions[3].height = 24

        for col_num in range(1, len(headers_flags) + 1):
            cell = ws_flags.cell(row=3, column=col_num)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = align_center
            cell.border = border_all

        for r_idx, row_data in enumerate(quality_flags.itertuples(index=False), start=4):
            ws_flags.row_dimensions[r_idx].height = 22
            row_fill = zebra_fill if (r_idx % 2 == 0) else PatternFill(fill_type=None)
            for c_idx, val in enumerate(row_data, start=1):
                cell = ws_flags.cell(row=r_idx, column=c_idx)
                cell.font = Font(name=font_family, size=10)
                cell.border = border_all
                cell.fill = row_fill
                cell.value = val if pd.notna(val) else "-"
                cell.alignment = align_left if c_idx == len(row_data) else align_center

    for ws in wb.worksheets:
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                if cell.row == 1:
                    continue
                val_str = str(cell.value or "")
                if len(val_str) > max_len:
                    max_len = len(val_str)
            ws.column_dimensions[col_letter].width = min(max(max_len + 4, 12), 50)

    wb.save(output_path)


def export_historical_metrics_excel(
    historical_metrics: pd.DataFrame,
    quality_flags: pd.DataFrame,
    output_path: Path,
) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        historical_metrics.to_excel(writer, sheet_name="Historical Metrics", index=False)
        quality_flags.to_excel(writer, sheet_name="Data Quality Flags", index=False)

        wb = writer.book
        header_fill = PatternFill(start_color="1B365D", end_color="1B365D", fill_type="solid")
        header_font = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")
        thin_side = Side(style="thin", color="CBD5E1")
        border = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)
        zebra_fill = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")

        for ws in wb.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            ws.sheet_view.showGridLines = True

            for cell in ws[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                cell.border = border
            ws.row_dimensions[1].height = 30

            for row_idx in range(2, ws.max_row + 1):
                for cell in ws[row_idx]:
                    cell.border = border
                    if row_idx % 2 == 0:
                        cell.fill = zebra_fill
                    cell.alignment = Alignment(vertical="top", wrap_text=(ws.title == "Data Quality Flags"))

            if ws.title == "Historical Metrics":
                for col_idx, cell in enumerate(ws[1], start=1):
                    col_name = str(cell.value or "").lower()
                    if any(token in col_name for token in ("margin", "share", "to_revenue", "to_operating_cash_flow")):
                        for row_idx in range(2, ws.max_row + 1):
                            ws.cell(row=row_idx, column=col_idx).number_format = "0.0%"
                    elif col_name in {"fiscal_year", "period_order"}:
                        for row_idx in range(2, ws.max_row + 1):
                            ws.cell(row=row_idx, column=col_idx).number_format = "0"

            for col_cells in ws.columns:
                max_len = max((len(str(c.value)) if c.value is not None else 0) for c in col_cells)
                col_letter = get_column_letter(col_cells[0].column)
                ws.column_dimensions[col_letter].width = min(max(max_len + 2, 12), 48)


# --------------------------------------------------
# 7. Run the project pipeline
# --------------------------------------------------

def main():
    # 1. Existing SEC observations and quarterly/historical logic.
    raw = load_observations()

    quarterly_cf = derive_quarterly_cash_flows(raw)
    operating = build_quarterly_operating_metrics(raw)

    metrics = build_financial_metrics(
        quarterly_cf, operating
    )

    checks = reconciliation_checks(
        raw, quarterly_cf, operating
    )

    historical_flags = []
    historical_metrics = build_period_metrics(raw, historical_flags)
    historical_flags_df = pd.DataFrame(
        historical_flags,
        columns=["severity", "flag", "fiscal_year", "period", "metric", "detail"],
    )

    assumptions, annual_trends = build_assumption_book_from_raw(raw)

    # 2. Read and generate annual datasets (FY2019-FY2025).
    annual_raw_df = load_annual_dataset()
    annual_dataset_metrics = build_annual_dataset_metrics(annual_raw_df)

    # Save the existing output files.
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
    historical_metrics.to_csv(
        OUTPUT_DIR / "historical_metrics.csv",
        index=False,
    )
    historical_flags_df.to_csv(
        OUTPUT_DIR / "historical_metrics_flags.csv",
        index=False,
    )

    historical_excel_path = OUTPUT_DIR / "historical_metrics.xlsx"
    export_historical_metrics_excel(
        historical_metrics,
        historical_flags_df,
        historical_excel_path,
    )
    checks.to_csv(
        OUTPUT_DIR / "reconciliation_checks.csv",
        index=False,
    )
    annual_trends.to_csv(
        OUTPUT_DIR / "annual_trends.csv",
        index=False,
    )

    # Save the new annual CSV files.
    if annual_raw_df is not None and not annual_raw_df.empty:
        annual_raw_df.to_csv(OUTPUT_DIR / "annual_dataset_cleaned.csv", index=False)
        annual_dataset_metrics.to_csv(OUTPUT_DIR / "annual_dataset_metrics.csv", index=False)
        print("Successfully generated annual CSV files: annual_dataset_cleaned.csv and annual_dataset_metrics.csv")

    # Export forecast assumption book to styled Excel
    excel_path = OUTPUT_DIR / "forecast_assumption_book.xlsx"
    export_forecast_assumption_book_excel(assumptions, annual_trends, checks, excel_path)

    print("\nReconciliation summary")
    print(checks["status"].value_counts(dropna=False))

    failures = checks[checks["status"] == "FAIL"]
    if not failures.empty:
        print("\nChecks requiring investigation:")
        print(failures.to_string(index=False))

    print(f"\nOutput files saved to: {OUTPUT_DIR.resolve()}")
    print(f"Assumption Book for modelling generated: {excel_path.name}")
    print(f"Historical metrics CSV generated: {(OUTPUT_DIR / 'historical_metrics.csv').resolve()}")
    print(f"Historical metrics Excel generated: {historical_excel_path.resolve()}")


if __name__ == "__main__":
    main()