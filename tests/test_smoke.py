import pandas as pd
from src.evm_guard.features import add_evm_derived_features, compute_eac_baselines
from src.evm_guard.targets import add_targets


def test_feature_pipeline_smoke():
    df = pd.DataFrame({
        "ProjectID": ["P1","P1","P1"],
        "Month": pd.to_datetime(["2027-01-01","2027-02-01","2027-03-01"]),
        "PlannedCost_USD": [100,100,100],
        "ActualCost_USD": [110,90,120],
        "ForecastCost_USD": [105,110,130],
        "CumPlanned_USD": [100,200,300],
        "CumActual_USD": [110,200,320],
        "CumForecast_USD": [105,215,340],
        "EV": [90,190,280],
        "PV": [100,200,300],
        "AC": [110,200,320],
        "CPI": [0.82,0.95,0.88],
        "SPI": [0.90,0.97,0.92],
    })
    df = add_evm_derived_features(df)
    df = compute_eac_baselines(df)
    df = add_targets(df)

    assert "BAC" in df.columns
    assert "CV" in df.columns
    assert "SV" in df.columns
    assert "EAC_cpi" in df.columns
    assert "y_risk_level" in df.columns
