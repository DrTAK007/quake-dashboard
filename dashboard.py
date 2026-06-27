# ============================================================
# Inland Empire / Cajon Pass Earthquake Dashboard
# Daily automated M4.0+ probability + 7-day activation score
#
# Output:
#   docs/index.html
#
# Designed for GitHub Actions + GitHub Pages
# ============================================================

import os
import time
import warnings
from pathlib import Path
from datetime import date, timedelta

import requests
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ============================================================
# SETTINGS
# ============================================================

START_YEAR = 2008
TARGET_MAG = 4.0
FEATURE_WINDOW_DAYS = 7
FORECAST_HORIZONS = [7, 30, 60, 90]

# Inland Empire / Cajon Pass / San Bernardino / Riverside / San Jacinto corridor
REGION = {
    "minlatitude": 33.4,
    "maxlatitude": 35.0,
    "minlongitude": -118.0,
    "maxlongitude": -116.0,
}

# Approximate Cajon Pass center
CAJON_LAT = 34.32
CAJON_LON = -117.47

USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

OUTPUT_DIR = Path("docs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# BASIC HELPERS
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )

    return 2 * R * np.arcsin(np.sqrt(a))


def get_usgs_earthquakes_chunk(starttime, endtime, minmagnitude=0.0, limit=20000):
    params = {
        "format": "geojson",
        "starttime": starttime,
        "endtime": endtime,
        "minmagnitude": minmagnitude,
        "minlatitude": REGION["minlatitude"],
        "maxlatitude": REGION["maxlatitude"],
        "minlongitude": REGION["minlongitude"],
        "maxlongitude": REGION["maxlongitude"],
        "orderby": "time-asc",
        "limit": limit,
    }

    r = requests.get(USGS_URL, params=params, timeout=90)
    r.raise_for_status()
    data = r.json()

    rows = []

    for f in data.get("features", []):
        prop = f.get("properties", {})
        geom = f.get("geometry", {})

        if not geom or "coordinates" not in geom:
            continue

        lon, lat, depth = geom["coordinates"]

        rows.append({
            "id": f.get("id"),
            "time": pd.to_datetime(prop.get("time"), unit="ms", utc=True),
            "mag": prop.get("mag"),
            "place": prop.get("place"),
            "lat": lat,
            "lon": lon,
            "depth_km": depth,
            "type": prop.get("type"),
            "status": prop.get("status"),
            "net": prop.get("net"),
            "magType": prop.get("magType"),
        })

    return pd.DataFrame(rows)


def month_ranges(start_year, end_date):
    """
    Monthly ranges reduce risk of hitting USGS result limits.
    """
    start = pd.Timestamp(f"{start_year}-01-01", tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC")

    ranges = []
    current = start

    while current < end:
        next_month = current + pd.DateOffset(months=1)
        this_end = min(next_month, end)
        ranges.append((current.strftime("%Y-%m-%d"), this_end.strftime("%Y-%m-%d")))
        current = next_month

    return ranges


def download_catalog():
    end_date = (date.today() + timedelta(days=1)).isoformat()
    dfs = []

    for starttime, endtime in month_ranges(START_YEAR, end_date):
        print(f"Downloading {starttime} to {endtime}")

        try:
            df = get_usgs_earthquakes_chunk(
                starttime=starttime,
                endtime=endtime,
                minmagnitude=0.0,
                limit=20000,
            )

            if len(df) > 0:
                dfs.append(df)

            time.sleep(0.1)

        except Exception as e:
            print(f"WARNING: failed {starttime} to {endtime}: {e}")

    if not dfs:
        raise RuntimeError("No earthquake data downloaded.")

    eq = pd.concat(dfs, ignore_index=True)
    eq = add_basic_features(eq)
    eq = (
        eq.drop_duplicates(subset=["id"])
        .sort_values("time")
        .reset_index(drop=True)
    )

    return eq


def add_basic_features(eq):
    eq = eq.copy()

    eq = eq.dropna(subset=["time", "mag", "lat", "lon"])
    eq["mag"] = pd.to_numeric(eq["mag"], errors="coerce")
    eq = eq.dropna(subset=["mag"])

    eq["date"] = eq["time"].dt.floor("D")

    eq["mag_bin"] = pd.cut(
        eq["mag"],
        bins=[-10, 1, 2, 3, 4, 5, 6, 10],
        labels=["<M1", "M1-M2", "M2-M3", "M3-M4", "M4-M5", "M5-M6", "M6+"],
    )

    eq["is_under3"] = eq["mag"] < 3
    eq["is_m1plus"] = eq["mag"] >= 1
    eq["is_m2plus"] = eq["mag"] >= 2
    eq["is_m3plus"] = eq["mag"] >= 3
    eq["is_m4plus"] = eq["mag"] >= 4
    eq["is_m45plus"] = eq["mag"] >= 4.5
    eq["is_m5plus"] = eq["mag"] >= 5
    eq["is_m65plus"] = eq["mag"] >= 6.5

    eq["dist_to_cajon_km"] = haversine_km(
        eq["lat"], eq["lon"], CAJON_LAT, CAJON_LON
    )

    return eq


# ============================================================
# COMPLETENESS + B-VALUE
# ============================================================

def estimate_mc_max_curvature(magnitudes, bin_width=0.1, min_mag=-1.0, max_mag=6.0):
    mags = pd.Series(magnitudes).dropna()
    mags = mags[(mags >= min_mag) & (mags <= max_mag)]

    if len(mags) < 50:
        return np.nan

    bins = np.arange(min_mag, max_mag + bin_width, bin_width)
    counts, edges = np.histogram(mags, bins=bins)

    if len(counts) == 0:
        return np.nan

    idx = np.argmax(counts)
    mc = edges[idx]

    return round(float(mc), 2)


def b_value_aaki(magnitudes, mc, delta_m=0.1):
    mags = pd.Series(magnitudes).dropna()

    if pd.isna(mc):
        return np.nan

    mags = mags[mags >= mc]

    if len(mags) < 20:
        return np.nan

    mean_mag = mags.mean()
    denom = mean_mag - (mc - delta_m / 2.0)

    if denom <= 0:
        return np.nan

    b = np.log10(np.e) / denom

    return float(b)


# ============================================================
# DAILY FORECAST DATASET
# ============================================================

daily_features = [
    "n_total",
    "n_under3",
    "n_m1_m2",
    "n_m2_m3",
    "max_mag_under3",
    "n_m3plus",
    "n_m4plus",
    "max_mag",
    "n_total_anomaly_z",
    "n_under3_anomaly_z",
    "n_m1_m2_anomaly_z",
    "n_m2_m3_anomaly_z",
    "n_m3plus_anomaly_z",
    "n_m4plus_anomaly_z",
    "min_dist_to_cajon",
    "median_dist_to_cajon",
    "median_depth",
    "depth_sd",
    "local_mc",
    "local_b_value",
]


def make_daily_forecast_dataset(
    eq,
    feature_window_days=7,
    future_window_days=7,
    target_mag=4.0,
):
    df = eq.copy().sort_values("time")
    df["date"] = df["time"].dt.floor("D")

    start_date = df["date"].min() + pd.Timedelta(days=feature_window_days)
    end_date = df["date"].max() - pd.Timedelta(days=future_window_days)

    if end_date <= start_date:
        raise RuntimeError("Not enough data to create daily forecast dataset.")

    all_dates = pd.date_range(start_date, end_date, freq="D", tz="UTC")

    rows = []

    for current_date in all_dates:
        past_start = current_date - pd.Timedelta(days=feature_window_days)
        past_end = current_date

        future_start = current_date
        future_end = current_date + pd.Timedelta(days=future_window_days)

        past = df[(df["time"] >= past_start) & (df["time"] < past_end)]
        future = df[(df["time"] >= future_start) & (df["time"] < future_end)]

        under3 = past[past["mag"] < 3]
        m1_m2 = past[(past["mag"] >= 1) & (past["mag"] < 2)]
        m2_m3 = past[(past["mag"] >= 2) & (past["mag"] < 3)]
        m3plus = past[past["mag"] >= 3]
        m4plus = past[past["mag"] >= 4]

        if len(past) >= 30:
            local_mc = estimate_mc_max_curvature(
                past["mag"],
                bin_width=0.1,
                min_mag=-1,
                max_mag=5,
            )
            local_b = b_value_aaki(past["mag"], mc=local_mc) if pd.notna(local_mc) else np.nan
        else:
            local_mc = np.nan
            local_b = np.nan

        target = int((future["mag"] >= target_mag).any())

        rows.append({
            "date": current_date,
            "feature_window_days": feature_window_days,
            "future_window_days": future_window_days,
            "target_mag": target_mag,
            "target_event_next_window": target,

            "n_total": len(past),
            "n_under3": len(under3),
            "n_m1_m2": len(m1_m2),
            "n_m2_m3": len(m2_m3),
            "max_mag_under3": under3["mag"].max() if len(under3) else 0,

            "n_m3plus": len(m3plus),
            "n_m4plus": len(m4plus),
            "max_mag": past["mag"].max() if len(past) else 0,

            "min_dist_to_cajon": past["dist_to_cajon_km"].min() if len(past) else np.nan,
            "median_dist_to_cajon": past["dist_to_cajon_km"].median() if len(past) else np.nan,

            "median_depth": past["depth_km"].median() if len(past) else np.nan,
            "depth_sd": past["depth_km"].std() if len(past) > 1 else np.nan,
            "local_mc": local_mc,
            "local_b_value": local_b,
        })

    out = pd.DataFrame(rows)

    rate_cols = [
        "n_total",
        "n_under3",
        "n_m1_m2",
        "n_m2_m3",
        "n_m3plus",
        "n_m4plus",
    ]

    for col in rate_cols:
        baseline_mean = out[col].shift(1).rolling(365, min_periods=60).mean()
        baseline_sd = out[col].shift(1).rolling(365, min_periods=60).std()

        out[f"{col}_anomaly_z"] = (
            (out[col] - baseline_mean) / baseline_sd.replace(0, np.nan)
        )

    return out


# ============================================================
# DAILY PROBABILITY MODEL
# ============================================================

def train_daily_model_positive_aware_calibration(
    daily_df,
    feature_cols,
    target_col="target_event_next_window",
    test_frac=0.15,
    cal_frac_within_train=0.20,
    random_state=42,
):
    daily_df = daily_df.sort_values("date").reset_index(drop=True)

    n = len(daily_df)
    test_start = int(n * (1 - test_frac))

    traincal_df = daily_df.iloc[:test_start].copy()
    test_df = daily_df.iloc[test_start:].copy()

    y_traincal = traincal_df[target_col].astype(int)

    if y_traincal.nunique() < 2:
        raise ValueError("Training/calibration period has only one class.")

    train_df, cal_df = train_test_split(
        traincal_df,
        test_size=cal_frac_within_train,
        stratify=y_traincal,
        random_state=random_state,
    )

    train_df = train_df.sort_values("date").reset_index(drop=True)
    cal_df = cal_df.sort_values("date").reset_index(drop=True)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(int)

    X_cal = cal_df[feature_cols]
    y_cal = cal_df[target_col].astype(int)

    X_test = test_df[feature_cols]
    y_test = test_df[target_col].astype(int)

    base_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=1000,
            random_state=random_state,
            class_weight="balanced_subsample",
            min_samples_leaf=3,
            max_features="sqrt",
        )),
    ])

    base_model.fit(X_train, y_train)

    raw_test_prob = base_model.predict_proba(X_test)[:, 1]

    try:
        calibrator = CalibratedClassifierCV(
            estimator=base_model,
            method="sigmoid",
            cv="prefit",
        )
        calibrator.fit(X_cal, y_cal)
    except Exception:
        try:
            from sklearn.frozen import FrozenEstimator
            calibrator = CalibratedClassifierCV(
                estimator=FrozenEstimator(base_model),
                method="sigmoid",
            )
            calibrator.fit(X_cal, y_cal)
        except Exception:
            # Fallback: use base model directly if calibration fails
            calibrator = base_model

    if hasattr(calibrator, "predict_proba"):
        cal_test_prob = calibrator.predict_proba(X_test)[:, 1]
        full_prob = calibrator.predict_proba(daily_df[feature_cols])[:, 1]
    else:
        cal_test_prob = raw_test_prob
        full_prob = base_model.predict_proba(daily_df[feature_cols])[:, 1]

    def safe_auc(y, p):
        return roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan

    def safe_ap(y, p):
        return average_precision_score(y, p) if len(np.unique(y)) > 1 else np.nan

    metrics = {
        "n_rows": len(daily_df),
        "n_train": len(train_df),
        "n_cal": len(cal_df),
        "n_test": len(test_df),

        "positive_days_total": int(daily_df[target_col].sum()),
        "positive_rate_total": float(daily_df[target_col].mean()),

        "train_positive_rate": float(y_train.mean()),
        "cal_positive_rate": float(y_cal.mean()),
        "test_positive_rate": float(y_test.mean()),

        "raw_auc_test": safe_auc(y_test, raw_test_prob),
        "raw_ap_test": safe_ap(y_test, raw_test_prob),
        "raw_brier_test": brier_score_loss(y_test, raw_test_prob),

        "cal_auc_test": safe_auc(y_test, cal_test_prob),
        "cal_ap_test": safe_ap(y_test, cal_test_prob),
        "cal_brier_test": brier_score_loss(y_test, cal_test_prob),
    }

    return {
        "base_model": base_model,
        "calibrator": calibrator,
        "train_df": train_df,
        "cal_df": cal_df,
        "test_df": test_df,
        "raw_test_prob": raw_test_prob,
        "cal_test_prob": cal_test_prob,
        "metrics": metrics,
        "full_prob": full_prob,
    }


# ============================================================
# LATEST ROW + ANOMALIES
# ============================================================

def make_latest_prediction_row(eq, feature_window_days=7):
    df = eq.copy().sort_values("time")
    df["date"] = df["time"].dt.floor("D")

    current_date = df["date"].max() + pd.Timedelta(days=1)
    past_start = current_date - pd.Timedelta(days=feature_window_days)
    past_end = current_date

    past = df[(df["time"] >= past_start) & (df["time"] < past_end)]

    under3 = past[past["mag"] < 3]
    m1_m2 = past[(past["mag"] >= 1) & (past["mag"] < 2)]
    m2_m3 = past[(past["mag"] >= 2) & (past["mag"] < 3)]
    m3plus = past[past["mag"] >= 3]
    m4plus = past[past["mag"] >= 4]

    if len(past) >= 30:
        local_mc = estimate_mc_max_curvature(
            past["mag"],
            bin_width=0.1,
            min_mag=-1,
            max_mag=5,
        )
        local_b = b_value_aaki(past["mag"], mc=local_mc) if pd.notna(local_mc) else np.nan
    else:
        local_mc = np.nan
        local_b = np.nan

    row = pd.DataFrame([{
        "date": current_date,
        "feature_window_days": feature_window_days,

        "n_total": len(past),
        "n_under3": len(under3),
        "n_m1_m2": len(m1_m2),
        "n_m2_m3": len(m2_m3),
        "max_mag_under3": under3["mag"].max() if len(under3) else 0,

        "n_m3plus": len(m3plus),
        "n_m4plus": len(m4plus),
        "max_mag": past["mag"].max() if len(past) else 0,

        "min_dist_to_cajon": past["dist_to_cajon_km"].min() if len(past) else np.nan,
        "median_dist_to_cajon": past["dist_to_cajon_km"].median() if len(past) else np.nan,

        "median_depth": past["depth_km"].median() if len(past) else np.nan,
        "depth_sd": past["depth_km"].std() if len(past) > 1 else np.nan,
        "local_mc": local_mc,
        "local_b_value": local_b,
    }])

    return row


def add_latest_anomalies_from_history(latest_row, historical_daily_df):
    latest_row = latest_row.copy()

    rate_cols = [
        "n_total",
        "n_under3",
        "n_m1_m2",
        "n_m2_m3",
        "n_m3plus",
        "n_m4plus",
    ]

    for col in rate_cols:
        hist_mean = historical_daily_df[col].tail(365).mean()
        hist_sd = historical_daily_df[col].tail(365).std()

        if hist_sd == 0 or pd.isna(hist_sd):
            latest_row[f"{col}_anomaly_z"] = np.nan
        else:
            latest_row[f"{col}_anomaly_z"] = (
                latest_row[col] - hist_mean
            ) / hist_sd

    return latest_row


# ============================================================
# ACTIVATION MODEL
# ============================================================

def assign_cajon_distance_band(dist):
    if pd.isna(dist):
        return np.nan
    if dist < 25:
        return "0-25km"
    elif dist < 50:
        return "25-50km"
    elif dist < 75:
        return "50-75km"
    else:
        return "75km+"


def extract_window_features(w):
    under3 = w[w["mag"] < 3]
    m1_m2 = w[(w["mag"] >= 1) & (w["mag"] < 2)]
    m2_m3 = w[(w["mag"] >= 2) & (w["mag"] < 3)]
    m3plus = w[w["mag"] >= 3]
    m4plus = w[w["mag"] >= 4]

    if len(w) >= 50:
        local_mc = estimate_mc_max_curvature(
            w["mag"],
            bin_width=0.1,
            min_mag=-1,
            max_mag=5,
        )
        local_b = b_value_aaki(w["mag"], mc=local_mc) if pd.notna(local_mc) else np.nan
    else:
        local_mc = np.nan
        local_b = np.nan

    return {
        "n_total": len(w),
        "n_under3": len(under3),
        "n_m1_m2": len(m1_m2),
        "n_m2_m3": len(m2_m3),
        "n_m3plus": len(m3plus),
        "n_m4plus": len(m4plus),
        "max_mag": w["mag"].max() if len(w) else np.nan,
        "max_mag_under3": under3["mag"].max() if len(under3) else np.nan,
        "median_depth": w["depth_km"].median() if len(w) else np.nan,
        "depth_sd": w["depth_km"].std() if len(w) > 1 else np.nan,
        "median_dist_to_cajon": w["dist_to_cajon_km"].median() if len(w) else np.nan,
        "min_dist_to_cajon": w["dist_to_cajon_km"].min() if len(w) else np.nan,
        "local_mc": local_mc,
        "local_b_value": local_b,
    }


def make_case_control_windows_spatial_matched(
    eq,
    target_mag=4.0,
    window_days=7,
    controls_per_case=10,
    min_gap_days=60,
):
    eq = eq.copy().sort_values("time")
    target_events = eq[eq["mag"] >= target_mag].copy()

    if len(target_events) == 0:
        raise RuntimeError("No target events found for activation model.")

    target_events["target_dist_to_cajon"] = target_events["dist_to_cajon_km"]
    target_events["target_cajon_band"] = target_events["target_dist_to_cajon"].apply(
        assign_cajon_distance_band
    )

    rows = []
    all_dates = pd.date_range(eq["time"].min(), eq["time"].max(), freq="D", tz="UTC")
    target_times = target_events["time"].tolist()

    # Precompute safe dates far from target events
    safe_dates_all = []

    for d in all_dates:
        too_close = any(abs((d - t).days) < min_gap_days for t in target_times)
        if not too_close:
            safe_dates_all.append(d)

    rng = np.random.default_rng(42)

    for _, ev in target_events.iterrows():
        event_time = ev["time"]
        event_band = ev["target_cajon_band"]

        start = event_time - pd.Timedelta(days=window_days)
        w = eq[(eq["time"] >= start) & (eq["time"] < event_time)]

        rows.append({
            "anchor_time": event_time,
            "target_case": 1,
            "target_mag": ev["mag"],
            "target_place": ev["place"],
            "target_dist_to_cajon": ev["target_dist_to_cajon"],
            "target_cajon_band": event_band,
            **extract_window_features(w),
        })

        candidate_controls = []

        for d in safe_dates_all:
            c_start = d - pd.Timedelta(days=window_days)
            cw = eq[(eq["time"] >= c_start) & (eq["time"] < d)]

            if len(cw) == 0:
                continue

            c_min_dist = cw["dist_to_cajon_km"].min()
            c_band = assign_cajon_distance_band(c_min_dist)

            if c_band == event_band:
                candidate_controls.append(d)

        if len(candidate_controls) == 0:
            continue

        n_controls = min(controls_per_case, len(candidate_controls))
        chosen = rng.choice(candidate_controls, size=n_controls, replace=False)

        for d in chosen:
            c_start = d - pd.Timedelta(days=window_days)
            cw = eq[(eq["time"] >= c_start) & (eq["time"] < d)]

            rows.append({
                "anchor_time": d,
                "target_case": 0,
                "target_mag": np.nan,
                "target_place": None,
                "target_dist_to_cajon": np.nan,
                "target_cajon_band": event_band,
                **extract_window_features(cw),
            })

    return pd.DataFrame(rows)


activation_features = [
    "n_m3plus",
    "n_m4plus",
    "max_mag",
    "median_depth",
    "depth_sd",
    "n_total",
    "n_under3",
    "n_m1_m2",
    "n_m2_m3",
    "max_mag_under3",
    "local_mc",
    "local_b_value",
]


# ============================================================
# RISK LABELS
# ============================================================

def operational_risk_label(ratio):
    if pd.isna(ratio):
        return "Unknown"
    elif ratio < 0.75:
        return "Below baseline"
    elif ratio < 1.25:
        return "Baseline"
    elif ratio < 2:
        return "Mildly elevated"
    elif ratio < 3:
        return "Elevated"
    else:
        return "Strongly elevated"


def activation_category(score):
    if pd.isna(score):
        return "Unknown"
    elif score < 0.25:
        return "Low activation"
    elif score < 0.50:
        return "Moderate activation"
    elif score < 0.75:
        return "High activation"
    else:
        return "Very high activation"


# ============================================================
# HTML HELPERS
# ============================================================

def category_class(label):
    label = str(label).lower()

    if "below" in label:
        return "below"
    if label == "baseline":
        return "baseline"
    if "mild" in label or "elevated" in label:
        return "elevated"
    if "strong" in label or "very" in label:
        return "strong"

    return ""


def dataframe_to_html_table(df):
    return df.to_html(index=False, escape=False, classes="data-table")


# ============================================================
# MAIN WORKFLOW
# ============================================================

def main():
    print("Downloading earthquake catalog...")
    eq = download_catalog()

    print("Catalog range:", eq["time"].min(), "to", eq["time"].max())
    print("Total events:", len(eq))

    # --------------------------------------------------------
    # Multi-horizon probability models
    # --------------------------------------------------------

    multi_horizon_results = {}
    summary_rows = []

    for horizon in FORECAST_HORIZONS:
        print(f"Training M{TARGET_MAG}+ next {horizon} days model...")

        daily_df = make_daily_forecast_dataset(
            eq,
            feature_window_days=FEATURE_WINDOW_DAYS,
            future_window_days=horizon,
            target_mag=TARGET_MAG,
        )

        result = train_daily_model_positive_aware_calibration(
            daily_df=daily_df,
            feature_cols=daily_features,
            target_col="target_event_next_window",
            test_frac=0.15,
            cal_frac_within_train=0.20,
            random_state=42,
        )

        prob_col = f"prob_m40_next_{horizon}d"
        daily_df[prob_col] = result["full_prob"]

        multi_horizon_results[horizon] = {
            "daily_df": daily_df,
            "result": result,
            "prob_col": prob_col,
        }

        summary_rows.append({
            "target_mag": TARGET_MAG,
            "feature_window_days": FEATURE_WINDOW_DAYS,
            "forecast_horizon_days": horizon,
            **result["metrics"],
        })

    summary_df = pd.DataFrame(summary_rows)

    # --------------------------------------------------------
    # Latest true forecast row
    # --------------------------------------------------------

    latest_rows = []

    for horizon in FORECAST_HORIZONS:
        obj = multi_horizon_results[horizon]
        historical_daily_df = obj["daily_df"]
        calibrator = obj["result"]["calibrator"]

        latest_row = make_latest_prediction_row(
            eq,
            feature_window_days=FEATURE_WINDOW_DAYS,
        )

        latest_row = add_latest_anomalies_from_history(
            latest_row,
            historical_daily_df,
        )

        prob = calibrator.predict_proba(latest_row[daily_features])[:, 1][0]
        baseline = historical_daily_df["target_event_next_window"].mean()

        latest_rows.append({
            "date": latest_row["date"].iloc[0],
            "target": "M4.0+",
            "forecast_horizon_days": horizon,
            "predicted_probability": prob,
            "baseline_rate_for_horizon": baseline,
            "risk_ratio_vs_baseline": prob / baseline if baseline > 0 else np.nan,

            "n_under3_last7d": latest_row["n_under3"].iloc[0],
            "n_m1_m2_last7d": latest_row["n_m1_m2"].iloc[0],
            "n_m2_m3_last7d": latest_row["n_m2_m3"].iloc[0],
            "n_m3plus_last7d": latest_row["n_m3plus"].iloc[0],
            "n_m4plus_last7d": latest_row["n_m4plus"].iloc[0],
            "max_mag_last7d": latest_row["max_mag"].iloc[0],
            "min_dist_to_cajon_last7d": latest_row["min_dist_to_cajon"].iloc[0],
            "local_mc_last7d": latest_row["local_mc"].iloc[0],
            "local_b_value_last7d": latest_row["local_b_value"].iloc[0],
        })

    latest_forecast = pd.DataFrame(latest_rows)

    latest_forecast = latest_forecast.sort_values(
        "forecast_horizon_days"
    ).reset_index(drop=True)

    # Force monotonic cumulative horizon probabilities
    latest_forecast["monotonic_probability"] = latest_forecast[
        "predicted_probability"
    ].cummax()

    latest_forecast["monotonic_risk_ratio_vs_baseline"] = (
        latest_forecast["monotonic_probability"]
        / latest_forecast["baseline_rate_for_horizon"]
    )

    latest_forecast["operational_risk_category"] = latest_forecast[
        "monotonic_risk_ratio_vs_baseline"
    ].apply(operational_risk_label)

    # --------------------------------------------------------
    # Activation score
    # --------------------------------------------------------

    print("Training activation model...")

    try:
        cc_activation = make_case_control_windows_spatial_matched(
            eq,
            target_mag=TARGET_MAG,
            window_days=7,
            controls_per_case=10,
            min_gap_days=60,
        )

        activation_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=1000,
                random_state=42,
                class_weight="balanced_subsample",
                min_samples_leaf=3,
                max_features="sqrt",
            )),
        ])

        X_act = cc_activation[activation_features]
        y_act = cc_activation["target_case"].astype(int)

        activation_model.fit(X_act, y_act)

        latest_activation_row = make_latest_prediction_row(
            eq,
            feature_window_days=7,
        )

        activation_score = activation_model.predict_proba(
            latest_activation_row[activation_features]
        )[:, 1][0]

    except Exception as e:
        print("WARNING: activation model failed:", e)
        latest_activation_row = make_latest_prediction_row(eq, feature_window_days=7)
        activation_score = np.nan

    activation_result = pd.DataFrame([{
        "date": latest_activation_row["date"].iloc[0],
        "target": "M4.0+",
        "activation_window": "previous 7 days",
        "activation_score": activation_score,
        "activation_category": activation_category(activation_score),

        "n_under3_last7d": latest_activation_row["n_under3"].iloc[0],
        "n_m1_m2_last7d": latest_activation_row["n_m1_m2"].iloc[0],
        "n_m2_m3_last7d": latest_activation_row["n_m2_m3"].iloc[0],
        "n_m3plus_last7d": latest_activation_row["n_m3plus"].iloc[0],
        "n_m4plus_last7d": latest_activation_row["n_m4plus"].iloc[0],
        "max_mag_last7d": latest_activation_row["max_mag"].iloc[0],
        "min_dist_to_cajon_last7d": latest_activation_row["min_dist_to_cajon"].iloc[0],
        "local_mc_last7d": latest_activation_row["local_mc"].iloc[0],
        "local_b_value_last7d": latest_activation_row["local_b_value"].iloc[0],
    }])

    # --------------------------------------------------------
    # Final dashboard table
    # --------------------------------------------------------
    final_dashboard = latest_forecast[[
    "date",
    "target",
    "forecast_horizon_days",
    "predicted_probability",
    "monotonic_probability",
    "baseline_rate_for_horizon",
    "risk_ratio_vs_baseline",
    "monotonic_risk_ratio_vs_baseline",
    "operational_risk_category",
    ]].copy()

    final_dashboard["activation_score_7d"] = activation_score
    final_dashboard["activation_category_7d"] = activation_category(activation_score)

    # Display copies
    display_dashboard = final_dashboard.copy()
    display_dashboard["date"] = display_dashboard["date"].astype(str)

    # Format probability columns
    for col in [
        "predicted_probability",
        "monotonic_probability",
        "baseline_rate_for_horizon",
    ]:
        display_dashboard[col] = (
            display_dashboard[col] * 100
        ).round(2).astype(str) + "%"

    # Format risk-ratio columns
    for col in [
        "risk_ratio_vs_baseline",
        "monotonic_risk_ratio_vs_baseline",
    ]:
        display_dashboard[col] = (
            display_dashboard[col].round(2).astype(str) + "x"
        )

    display_dashboard["activation_score_7d"] = display_dashboard[
        "activation_score_7d"
    ].round(3)

    # Rename columns for readability
    display_dashboard = display_dashboard.rename(columns={
        "date": "Forecast date",
        "target": "Target",
        "forecast_horizon_days": "Forecast horizon",
        "predicted_probability": "Raw model probability",
        "monotonic_probability": "Conservative adjusted probability",
        "baseline_rate_for_horizon": "Historical baseline",
        "risk_ratio_vs_baseline": "Raw risk ratio",
        "monotonic_risk_ratio_vs_baseline": "Adjusted risk ratio",
        "operational_risk_category": "Risk category",
        "activation_score_7d": "7-day activation score",
        "activation_category_7d": "7-day activation category",
    })

    # Add CSS classes for categories
    display_dashboard["Risk category"] = display_dashboard[
        "Risk category"
    ].apply(lambda x: f'<span class="{category_class(x)}">{x}</span>')

    display_dashboard["7-day activation category"] = display_dashboard[
        "7-day activation category"
    ].apply(lambda x: f'<span class="{category_class(x)}">{x}</span>')

    activation_display = activation_result.copy()
    activation_display["date"] = activation_display["date"].astype(str)
    activation_display["activation_score"] = activation_display["activation_score"].round(3)
    activation_display["max_mag_last7d"] = activation_display["max_mag_last7d"].round(2)
    activation_display["min_dist_to_cajon_last7d"] = activation_display[
        "min_dist_to_cajon_last7d"
    ].round(2)
    activation_display["local_b_value_last7d"] = activation_display[
        "local_b_value_last7d"
    ].round(3)

    summary_display = summary_df.copy()
    numeric_cols = summary_display.select_dtypes(include=[np.number]).columns
    summary_display[numeric_cols] = summary_display[numeric_cols].round(4)

    latest_date = str(final_dashboard["date"].iloc[0])
    activation_cat = activation_result["activation_category"].iloc[0]
    activation_score_fmt = (
        "NA" if pd.isna(activation_score) else round(float(activation_score), 3)
    )

    latest_features = activation_result.iloc[0]
    # --------------------------------------------------------
    # Plain-English current readout
    # --------------------------------------------------------

    short_term_category = latest_forecast.loc[
        latest_forecast["forecast_horizon_days"] == 7,
        "operational_risk_category"
    ].iloc[0]

    thirty_day_category = latest_forecast.loc[
        latest_forecast["forecast_horizon_days"] == 30,
        "operational_risk_category"
    ].iloc[0]

    if "Below baseline" in short_term_category and "Below baseline" in thirty_day_category:
        current_readout = (
            "Moderate microseismic activation may be present, but the calibrated "
            "M4.0+ probability remains below historical baseline in the 7-day and "
            "30-day horizons. The model does not currently indicate elevated "
            "short-term M4.0+ probability."
        )
    elif "Baseline" in short_term_category and "Baseline" in thirty_day_category:
        current_readout = (
            "The model currently estimates M4.0+ probability near historical baseline. "
            "Recent seismicity does not indicate a clearly elevated short-term state."
        )
    elif "elevated" in short_term_category.lower() or "elevated" in thirty_day_category.lower():
        current_readout = (
            "The model currently estimates elevated M4.0+ probability relative to "
            "historical baseline. This should be interpreted as an experimental "
            "probabilistic signal, not a deterministic earthquake prediction."
        )
    else:
        current_readout = (
            "The model output is mixed. Review both the calibrated probability and "
            "the 7-day activation score before interpreting the current state."
        )

    # Add activation nuance
    if activation_cat == "Moderate activation":
        current_readout += (
            " The 7-day activation score is moderate, meaning the recent microseismic "
            "pattern deserves monitoring but is not classified as high activation."
        )
    elif activation_cat in ["High activation", "Very high activation"]:
        current_readout += (
            " The 7-day activation score is high, meaning recent microseismicity "
            "resembles pre-M4.0+ activation windows more strongly."
        )
    elif activation_cat == "Low activation":
        current_readout += (
            " The 7-day activation score is low."
        )

    # --------------------------------------------------------
    # Write HTML
    # --------------------------------------------------------

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Inland Empire / Cajon Pass Earthquake Monitor</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f7f7f7;
            color: #222;
        }}

        .card {{
            background: white;
            padding: 24px;
            border-radius: 14px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            margin-bottom: 24px;
        }}

        h1 {{
            margin-bottom: 5px;
        }}

        .subtitle {{
            color: #666;
            margin-bottom: 20px;
        }}

        table {{
            border-collapse: collapse;
            width: 100%;
            background: white;
            font-size: 14px;
        }}

        th, td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
            text-align: left;
        }}

        th {{
            background-color: #eeeeee;
        }}

        .below {{
            color: #2e7d32;
            font-weight: bold;
        }}

        .baseline {{
            color: #555;
            font-weight: bold;
        }}

        .elevated {{
            color: #c77800;
            font-weight: bold;
        }}

        .strong {{
            color: #b00020;
            font-weight: bold;
        }}

        .note {{
            font-size: 0.92em;
            color: #555;
            line-height: 1.5;
        }}

        .big-number {{
            font-size: 1.4em;
            font-weight: bold;
        }}

        .big-readout {{
            font-size: 1.08em;
            line-height: 1.6;
            background: #f4f7fb;
            border-left: 5px solid #4b6cb7;
            padding: 16px;
            border-radius: 8px;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
        }}

        .mini {{
            background: #fafafa;
            border-radius: 10px;
            padding: 14px;
            border: 1px solid #e5e5e5;
        }}
    </style>
</head>

<body>

<div class="card">
    <h1>Inland Empire / Cajon Pass Earthquake Monitor</h1>
    <div class="subtitle">Daily automated experimental model output for local M4.0+ activity</div>

    <p><strong>Latest forecast date:</strong> {latest_date}</p>
    <p><strong>Target:</strong> M4.0+ earthquake within the Inland Empire / Cajon Pass study box</p>
    <p><strong>7-day activation score:</strong>
       <span class="big-number">{activation_score_fmt}</span>
       — <strong>{activation_cat}</strong>
    </p>
</div>

<div class="card">
    <h2>Current Readout</h2>
    <p class="big-readout">
        {current_readout}
    </p>
</div>

<div class="card">
    <h2>How to Read This Dashboard</h2>
    <p>
    The table below compares the raw model probability and the conservative adjusted
    probability against the historical baseline for each forecast horizon.
    The activation score is separate and asks whether the last 7 days of seismicity
    resemble pre-M4.0+ activation windows.
    </p>
</div>

<div class="card">
    <h2>Multi-Horizon Forecast</h2>
    {dataframe_to_html_table(display_dashboard)}
</div>

<div class="card">
    <h2>Latest 7-Day Seismic Features</h2>

    <div class="grid">
        <div class="mini"><strong>&lt;M3 events:</strong><br>{latest_features["n_under3_last7d"]}</div>
        <div class="mini"><strong>M1–M2 events:</strong><br>{latest_features["n_m1_m2_last7d"]}</div>
        <div class="mini"><strong>M2–M3 events:</strong><br>{latest_features["n_m2_m3_last7d"]}</div>
        <div class="mini"><strong>M3+ events:</strong><br>{latest_features["n_m3plus_last7d"]}</div>
        <div class="mini"><strong>M4+ events:</strong><br>{latest_features["n_m4plus_last7d"]}</div>
        <div class="mini"><strong>Max magnitude:</strong><br>{round(float(latest_features["max_mag_last7d"]), 2)}</div>
        <div class="mini"><strong>Closest to Cajon Pass:</strong><br>{round(float(latest_features["min_dist_to_cajon_last7d"]), 2)} km</div>
        <div class="mini"><strong>Local b-value:</strong><br>{round(float(latest_features["local_b_value_last7d"]), 3) if pd.notna(latest_features["local_b_value_last7d"]) else "NA"}</div>
    </div>

    <br>
    {dataframe_to_html_table(activation_display)}
</div>

<div class="card">
    <h2>Model Performance Summary</h2>
    {dataframe_to_html_table(summary_display)}
</div>

<div class="card note">
    <h2>Important Interpretation Note</h2>
    <p>
    This dashboard is an experimental probabilistic model for local M4.0+ activity.
    It is not a deterministic earthquake prediction system and should not be used
    for emergency decisions. It is intended to monitor whether recent seismicity is
    below baseline, baseline, or elevated relative to historical activity in the selected region.
    </p>
    <p>
    The calibrated probability model and the 7-day activation score are separate.
    A moderate activation score with below-baseline probability means there is microseismic
    activity, but the full pattern has not crossed the model's threshold for elevated M4.0+ probability.
    </p>
    <p>
    Last generated automatically by GitHub Actions.
    </p>
</div>

</body>
</html>
"""

    output_path = OUTPUT_DIR / "index.html"
    output_path.write_text(html, encoding="utf-8")

    print(f"Dashboard written to {output_path}")


if __name__ == "__main__":
    main()
