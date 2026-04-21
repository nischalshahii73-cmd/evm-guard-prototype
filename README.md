# EVM-Guard Prototype (Streamlit) — Updated

This repository implements **EVM-Guard**: an explainable decision-support prototype for early prediction of cost/schedule overrun risk using Earned Value Management (EVM) indicators.

## Input
Upload **one or more** Excel workbooks in the **SetA_Project*.xlsx** format (each contains a `Budget_Costs` sheet with EV/PV/AC/CPI/SPI time-phased by Month).

## Quick start
### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Notes on labels
Targets are derived from end-of-project outcomes available in the workbooks:
- **Cost overrun ratio** = (Final CumActual - BAC) / BAC (BAC = final CumPlanned)
- **Schedule slip proxy** = 1 - Final SPI (finish-date variance not provided)
- **Risk level** is defined from the two outcomes (see `src/evm_guard/targets.py`)

These definitions must be documented transparently in your dissertation.
