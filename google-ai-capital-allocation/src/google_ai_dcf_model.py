"""
Alphabet / Google AI Capital Allocation — Forecast + DCF + Reverse DCF
All monetary values are USD billions unless otherwise stated.

Updated to export directly to an Excel Spreadsheet (.xlsx)
matching the style and layout of Alphabet_AI_Capital_Allocation_Valuation_Model.xlsx.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parent.parent 

MODEL_DIR = ROOT / "outputs/models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT = MODEL_DIR / "dcf"
OUT.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = ROOT / "config" / "DCF_model_assumptions.json"
ASSUMPTIONS = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

MARKET_PRICE = ASSUMPTIONS["market_price_usd"]
DILUTED_SHARES_B = ASSUMPTIONS["diluted_shares_billions"]
NET_CASH_B = ASSUMPTIONS["net_cash_usd_billions"]
MARKET_CAP_B = MARKET_PRICE * DILUTED_SHARES_B
TAX_RATE = ASSUMPTIONS["tax_rate"]
BASE_YEAR_REVENUE = ASSUMPTIONS["base_year_revenue_usd_billions"]
BASE_YEAR_OPERATING_MARGIN = ASSUMPTIONS["base_year_operating_margin"]
BASE_YEAR_DA = ASSUMPTIONS["base_year_da_usd_billions"]
BASE_YEAR_CAPEX = ASSUMPTIONS["base_year_capex_usd_billions"]
BASE_YEAR = ASSUMPTIONS["base_year"]
FORECAST_YEARS = ASSUMPTIONS["forecast_years"]
SCENARIOS = ASSUMPTIONS["scenarios"]
SENSITIVITY = ASSUMPTIONS["sensitivity"]

def project_scenario(name, a):
    rows = []
    revenue = BASE_YEAR_REVENUE
    for i, year in enumerate(FORECAST_YEARS):
        revenue *= 1 + a["revenue_growth"][i]
        margin = a["operating_margin"][i]
        da = revenue * a["da_pct_revenue"][i]
        capex = revenue * a["capex_pct_revenue"][i]
        ebit = revenue * margin
        nopat = ebit * (1 - TAX_RATE)
        fcf = nopat + da - capex
        rows.append({
            "scenario": name, "year": year, "revenue_usd_bn": revenue,
            "revenue_growth": a["revenue_growth"][i],
            "operating_margin": margin, "operating_income_usd_bn": ebit,
            "tax_rate": TAX_RATE, "nopat_usd_bn": nopat,
            "da_pct_revenue": a["da_pct_revenue"][i], "da_usd_bn": da,
            "capex_pct_revenue": a["capex_pct_revenue"][i], "capex_usd_bn": capex,
            "fcf_usd_bn": fcf, "wacc": a["wacc"], "terminal_growth": a["terminal_growth"]
        })
    return pd.DataFrame(rows)

def dcf_value(forecast, wacc, terminal_growth, net_cash=NET_CASH_B, shares=DILUTED_SHARES_B):
    fcfs = forecast["fcf_usd_bn"].to_numpy(dtype=float)
    years = np.arange(1, len(fcfs) + 1)
    pv_fcfs = np.sum(fcfs / ((1 + wacc) ** years))
    terminal_fcf = fcfs[-1] * (1 + terminal_growth)
    if wacc <= terminal_growth:
        terminal_value = np.nan
        equity_value = np.nan
        value_per_share = np.nan
    else:
        terminal_value = terminal_fcf / (wacc - terminal_growth)
        pv_terminal = terminal_value / ((1 + wacc) ** len(fcfs))
        equity_value = pv_fcfs + pv_terminal + net_cash
        value_per_share = equity_value / shares
    return {
        "pv_explicit_fcf_usd_bn": pv_fcfs,
        "terminal_value_usd_bn": terminal_value,
        "equity_value_usd_bn": equity_value,
        "value_per_share_usd": value_per_share,
        "market_price_usd": MARKET_PRICE,
        "upside_downside_pct": value_per_share / MARKET_PRICE - 1 if pd.notna(value_per_share) else np.nan,
        "wacc": wacc, "terminal_growth": terminal_growth
    }

def reverse_dcf_growth(forecast, target_equity_value, wacc, net_cash=NET_CASH_B, shares=DILUTED_SHARES_B):
    fcfs = forecast["fcf_usd_bn"].to_numpy(dtype=float)
    years = np.arange(1, len(fcfs) + 1)
    pv_explicit = np.sum(fcfs / ((1 + wacc) ** years))
    required_pv_terminal = target_equity_value - net_cash - pv_explicit
    if required_pv_terminal <= 0:
        return {"implied_terminal_fcf_growth": np.nan, "note": "Explicit-period PV plus net cash exceeds target."}
    pv_terminal_target = required_pv_terminal
    terminal_value_target = pv_terminal_target * ((1 + wacc) ** len(fcfs))
    fcf_2030 = fcfs[-1]
    if fcf_2030 <= 0:
        return {"implied_terminal_fcf_growth": np.nan, "note": "FCF in 2030 is non-positive."}
    g = (terminal_value_target * wacc - fcf_2030) / (terminal_value_target + fcf_2030)
    note = "Mathematical solution."
    if g >= wacc:
        note = "Implied growth is at/above WACC."
    return {"implied_terminal_fcf_growth": g, "note": note}

# Run model calculations
forecasts = pd.concat([project_scenario(k, v) for k, v in SCENARIOS.items()], ignore_index=True)
valuation_rows = []
reverse_rows = []
for name, a in SCENARIOS.items():
    f = forecasts[forecasts.scenario == name].sort_values("year")
    v = dcf_value(f, a["wacc"], a["terminal_growth"])
    v["scenario"] = name
    valuation_rows.append(v)
    r = reverse_dcf_growth(f, MARKET_CAP_B, a["wacc"])
    reverse_rows.append({
        "scenario": name,
        "target_market_cap_usd_bn": MARKET_CAP_B,
        "target_equity_value_usd_bn": MARKET_CAP_B,
        "wacc": a["wacc"],
        "explicit_period_pv_fcf_usd_bn": v["pv_explicit_fcf_usd_bn"],
        "fcf_2030_usd_bn": float(f.iloc[-1]["fcf_usd_bn"]),
        **r
    })

val_df = pd.DataFrame(valuation_rows)
rev_df = pd.DataFrame(reverse_rows)

# Save CSV backups
forecasts.to_csv(OUT / "scenario_forecasts.csv", index=False)
val_df.to_csv(OUT / "dcf_valuation.csv", index=False)
rev_df.to_csv(OUT / "reverse_dcf.csv", index=False)

# Export the formatted Excel workbook
excel_path = OUT / "Alphabet_AI_Capital_Allocation_Valuation_Model_Output.xlsx"

wb = openpyxl.Workbook()
wb.remove(wb.active) # Remove the default worksheet

# Configure styles
font_title = Font(name="Calibri", size=14, bold=True, color="1F4E78")
font_section = Font(name="Calibri", size=11, bold=True, color="1F4E78")
font_header = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
font_bold = Font(name="Calibri", size=10, bold=True)
font_regular = Font(name="Calibri", size=10)

fill_header = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")

thin_border = Border(
    left=Side(style='thin', color='D9D9D9'),
    right=Side(style='thin', color='D9D9D9'),
    top=Side(style='thin', color='D9D9D9'),
    bottom=Side(style='thin', color='D9D9D9')
)

# Sheet 1: Executive Summary
ws1 = wb.create_sheet(title="Executive Summary")
ws1.append(["Alphabet Inc. (GOOGL) - AI Capital Allocation & Institutional Valuation Model"])
ws1.cell(1, 1).font = font_title
ws1.append([])
ws1.append(["1. Executive Summary & Calibration Baseline"])
ws1.cell(3, 1).font = font_section

summary_items = [
    ("• Current Stock Price (GOOGL):", f"${MARKET_PRICE:.2f}"),
    ("• Diluted Shares Outstanding:", f"{DILUTED_SHARES_B*1000:,.0f} Million"),
    ("• Current Market Capitalization:", f"${MARKET_CAP_B:,.2f} Billion"),
]
for scenario in valuation_rows:
    s_name = scenario["scenario"]
    v_share = scenario["value_per_share_usd"]
    up_dn = scenario["upside_downside_pct"] * 100
    summary_items.append((f"• {s_name} Case Intrinsic Target Price:", f"${v_share:.2f} ({up_dn:+.1f}%)"))

for label, val in summary_items:
    ws1.append([label, val])
    r = ws1.max_row
    ws1.cell(r, 1).font = font_bold
    ws1.cell(r, 2).font = font_regular

ws1.append([])
ws1.append(["2. Scenario Matrix (2027–2030E)"])
ws1.cell(ws1.max_row, 1).font = font_section

headers1 = ["Metric / Parameter", "Bear Case", "Base Case", "Bull Case"]
ws1.append(headers1)
r_head = ws1.max_row
for col_idx in range(1, 5):
    cell = ws1.cell(r_head, col_idx)
    cell.font = font_header
    cell.fill = fill_header

matrix_rows = [
    ["Revenue CAGR (2027E-2030E)", f"{SCENARIOS['Bear']['revenue_growth'][0]*100:.1f}%", f"{SCENARIOS['Base']['revenue_growth'][0]*100:.1f}%", f"{SCENARIOS['Bull']['revenue_growth'][0]*100:.1f}%"],
    ["Terminal Operating Margin (2030E)", f"{SCENARIOS['Bear']['operating_margin'][-1]*100:.1f}%", f"{SCENARIOS['Base']['operating_margin'][-1]*100:.1f}%", f"{SCENARIOS['Bull']['operating_margin'][-1]*100:.1f}%"],
    ["CapEx Peak as % of Revenue", f"{max(SCENARIOS['Bear']['capex_pct_revenue'])*100:.1f}%", f"{max(SCENARIOS['Base']['capex_pct_revenue'])*100:.1f}%", f"{max(SCENARIOS['Bull']['capex_pct_revenue'])*100:.1f}%"],
    ["Weighted Average Cost of Capital (WACC)", f"{SCENARIOS['Bear']['wacc']*100:.1f}%", f"{SCENARIOS['Base']['wacc']*100:.1f}%", f"{SCENARIOS['Bull']['wacc']*100:.1f}%"],
    ["Terminal Growth Rate (g)", f"{SCENARIOS['Bear']['terminal_growth']*100:.1f}%", f"{SCENARIOS['Base']['terminal_growth']*100:.1f}%", f"{SCENARIOS['Bull']['terminal_growth']*100:.1f}%"],
    ["Implied Share Value ($)", f"${val_df[val_df.scenario=='Bear']['value_per_share_usd'].values[0]:.2f}", f"${val_df[val_df.scenario=='Base']['value_per_share_usd'].values[0]:.2f}", f"${val_df[val_df.scenario=='Bull']['value_per_share_usd'].values[0]:.2f}"],
    ["Upside / (Downside) vs Current Price", f"{val_df[val_df.scenario=='Bear']['upside_downside_pct'].values[0]*100:+.1f}%", f"{val_df[val_df.scenario=='Base']['upside_downside_pct'].values[0]*100:+.1f}%", f"{val_df[val_df.scenario=='Bull']['upside_downside_pct'].values[0]*100:+.1f}%"],
]

for row_data in matrix_rows:
    ws1.append(row_data)
    r = ws1.max_row
    for c in range(1, 5):
        ws1.cell(r, c).font = font_regular
        ws1.cell(r, c).border = thin_border

# Sheet 2: DCF & Projections
ws2 = wb.create_sheet(title="DCF & Projections")
ws2.append(["Alphabet Inc. (GOOGL) - Scenario DCF Valuation Model"])
ws2.cell(1, 1).font = font_title
ws2.append([])

headers2 = ["Scenario", "Year", "Revenue ($B)", "Rev Growth", "EBIT Margin", "EBIT ($B)", "NOPAT ($B)", "D&A ($B)", "CapEx ($B)", "FCF ($B)"]
ws2.append(headers2)
r_head2 = ws2.max_row
for c in range(1, len(headers2)+1):
    cell = ws2.cell(r_head2, c)
    cell.font = font_header
    cell.fill = fill_header

for idx, row in forecasts.iterrows():
    ws2.append([
        row["scenario"], int(row["year"]),
        round(row["revenue_usd_bn"], 2),
        f"{row['revenue_growth']*100:.1f}%",
        f"{row['operating_margin']*100:.1f}%",
        round(row["operating_income_usd_bn"], 2),
        round(row["nopat_usd_bn"], 2),
        round(row["da_usd_bn"], 2),
        round(row["capex_usd_bn"], 2),
        round(row["fcf_usd_bn"], 2)
    ])
    r = ws2.max_row
    for c in range(1, len(headers2)+1):
        ws2.cell(r, c).font = font_regular
        ws2.cell(r, c).border = thin_border

# Sheet 3: Sensitivity Analysis
ws3 = wb.create_sheet(title="Sensitivity Analysis")
ws3.append(["Sensitivity Analysis: Base Case Valuation Matrix"])
ws3.cell(1, 1).font = font_title
ws3.append([])

base_f = forecasts[forecasts.scenario == "Base"].sort_values("year")
wacc_list = SENSITIVITY["wacc"]
tg_list = SENSITIVITY["terminal_growth"]

ws3.append(["Sensitivity to WACC and Terminal Growth Rate (g)"])
ws3.cell(3, 1).font = font_section

headers3 = ["WACC \\ g"] + [f"{g*100:.1f}%" for g in tg_list]
ws3.append(headers3)
r_head3 = ws3.max_row
for c in range(1, len(headers3)+1):
    cell = ws3.cell(r_head3, c)
    cell.font = font_header
    cell.fill = fill_header

for w in wacc_list:
    row_vals = [f"{w*100:.1f}%"]
    for g in tg_list:
        res = dcf_value(base_f, w, g)
        val = res["value_per_share_usd"]
        row_vals.append(f"${val:.2f}" if pd.notna(val) else "N/A")
    ws3.append(row_vals)
    r = ws3.max_row
    for c in range(1, len(headers3)+1):
        ws3.cell(r, c).font = font_regular
        ws3.cell(r, c).border = thin_border

# Automatically adjust column widths
for sheet in wb.worksheets:
    for col in sheet.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = get_column_letter(col[0].column)
        sheet.column_dimensions[col_letter].width = max(max_len + 3, 12)

# Save the workbook
wb.save(excel_path)
print(f"Spreadsheet saved successfully to: {excel_path}")