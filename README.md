I am a student in the Queen’s Commerce program, and this repository represents an independent personal project dedicated to the analysis of financial markets, economic developments, and contemporary financial topics. The content combines professional analytical methodologies with my own independent research, interpretations, and perspectives.



All content is provided for informational and educational purposes only and does not constitute financial, investment, legal, or other professional advice. While I strive to maintain analytical rigor and accuracy, I make no representations or warranties regarding the completeness, reliability, or timeliness of the information presented. Any reliance on the content is solely at the user’s own discretion and risk. I expressly disclaim liability for any losses, damages, or consequences arising from the use of, or reliance upon, the analyses, opinions, or information contained herein.



# Google AI Capital Allocation Analysis

This repository analyzes Alphabet (Google) financial and operating results, with a focus on AI-related capital allocation. It combines SEC-sourced observations, historical metric calculations, scenario-based discounted cash flow (DCF) valuation, and annual and quarterly charts.

## Project Structure

```text
google-ai-capital-allocation/
  config/       DCF assumptions and scenario inputs
  data/         SEC observations and annual financial dataset
  notebooks/    Exploratory analysis notebooks
  outputs/      Generated data, charts, valuation files, and reports
  src/          Data pipeline, DCF model, and chart scripts
```

## Requirements

- Python 3
- NumPy
- pandas
- openpyxl
- Matplotlib

Install the dependencies from the repository root:

```bash
cd google-ai-capital-allocation
python -m pip install numpy pandas openpyxl matplotlib
```

## Run the Analysis

Run these commands from the `google-ai-capital-allocation` directory, in order:

```bash
python src/google_ai_capital_allocation.py
python src/google_ai_dcf_model.py
python src/plot_google_ai_trends.py
```

The first script validates the source observations and builds quarterly and historical metrics. The DCF script creates bear, base, and bull forecasts, valuations, reverse-DCF results, and a WACC/terminal-growth sensitivity analysis. The chart script uses the annual dataset and the quarterly observations produced by the first script.

The notebooks in `google-ai-capital-allocation/notebooks/` are exploratory analysis files. The scripts above are the recommended workflow for producing the current outputs.

## Input Data

- `google-ai-capital-allocation/data/sec_observations.csv` contains SEC-sourced observations. It requires `fiscal_year`, `period`, `metric`, `value`, `basis`, and `source_url` columns. Valid periods include `Q1`–`Q4` and `FY`; bases include `quarter`, `ytd`, and `annual`.
- `google-ai-capital-allocation/data/Annual_dataset_FY2019_FY2025.csv` contains annual financial data for charting and historical analysis.
- Values in these source CSV files are in US$ millions. DCF assumptions and model outputs use US$ billions, except per-share values.
- `google-ai-capital-allocation/config/DCF_model_assumptions.json` contains the market snapshot, base-year assumptions, scenario forecasts, and sensitivity ranges. Review and update these assumptions before running the DCF when appropriate.

Numeric SEC observations are expected to have a source URL. The data pipeline checks required columns, valid period and basis values, duplicate observations, and missing source URLs.

## Generated Outputs

- `google-ai-capital-allocation/outputs/data/` contains normalized observations, quarterly cash flows and metrics, historical metrics, reconciliation checks, and Excel reports.
- `google-ai-capital-allocation/outputs/models/dcf/` contains scenario forecasts, DCF valuation and reverse-DCF CSV files, and the valuation workbook.
- `google-ai-capital-allocation/outputs/charts/` contains generated chart images and chart data exports.

The DCF results depend on the assumptions in the JSON configuration and are analytical estimates, not investment advice.
