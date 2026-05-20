import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "optuna", "-q"],
               capture_output=True)

import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss, brier_score_loss,
)
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")
sns.set_theme(context="talk", style="whitegrid", font_scale=0.85)
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 200)

print("✅ All libraries loaded.")


DATA_PATH = r"..\data\raw\raw_data.csv"

print(f"Loading: {DATA_PATH}")

df_raw = pd.read_csv(DATA_PATH, dtype={"agent": "Float64", "company": "Float64"})

print(f"Shape : {df_raw.shape}   ({df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns)")
print(f"\nTarget (is_canceled) distribution:")
vc = df_raw["is_canceled"].value_counts()
print(f"  Not cancelled (0) : {vc[0]:,}  ({vc[0]/len(df_raw):.1%})")
print(f"  Cancelled     (1) : {vc[1]:,}  ({vc[1]/len(df_raw):.1%})")


print("=== dtype / missing-value audit ===")
audit = pd.DataFrame({
    "dtype"    : df_raw.dtypes,
    "n_null"   : df_raw.isna().sum(),
    "pct_null" : (df_raw.isna().mean() * 100).round(1),
    "n_unique" : df_raw.nunique(),
})
print(audit.sort_values("pct_null", ascending=False).to_string())


TARGET = "is_canceled"

DATE_COLS   = ["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"]
SOURCE_COLS = [
    "lead_time", "required_car_parking_spaces", "total_of_special_requests",
    "previous_cancellations", "previous_bookings_not_canceled",
    "adr", "adults", "children", "babies",
    "deposit_type", "market_segment", "distribution_channel", "customer_type", "hotel",
] + DATE_COLS

df = df_raw[SOURCE_COLS + [TARGET]].copy()
df["children"] = df["children"].fillna(0).astype(int)

print(f"Working frame: {df.shape}")
print(f"Columns: {df.columns.tolist()}")


negative_cols = [
    "lead_time",
    "adr",
    "adults",
    "children",
    "babies",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "total_of_special_requests"
]

for col in negative_cols:
    n_neg = (df[col] < 0).sum()
    print(f"{col:<35} negative values: {n_neg}")

df["total_guests"] = df["adults"] + df["children"] + df["babies"]

zero_guest_rows = (df["total_guests"] == 0).sum()
print(f"\nRows with zero guests: {zero_guest_rows}")

missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]

if len(missing_values) == 0:
    print("\nNo missing values remaining.")
else:
    print(missing_values.sort_values(ascending=False))

duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")


before = len(df)
df = df[df["adr"] >= 0].copy()
print(f"\n  → FIX: Dropped {before - len(df)} row(s) with negative ADR.")
print(f"         Remaining: {len(df):,} rows")

before = len(df)
df = df[df["total_guests"] > 0].copy()
df = df.drop(columns=["total_guests"])
print(f"\n  → FIX: Dropped {before - len(df)} zero-guest row(s).")
print(f"         Remaining: {len(df):,} rows")

before = len(df)
df = df.drop_duplicates(keep="first").reset_index(drop=True)
print(f"\n  → FIX: Removed {before - len(df):,} duplicate row(s). Kept first occurrence.")
print(f"         Remaining: {len(df):,} rows")


df.to_csv("..\data\cleaned\cleaned_data.csv", index=False)
