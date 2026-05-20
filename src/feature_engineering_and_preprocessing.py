import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "optuna", "-q"],
               capture_output=True)

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
sns.set_theme(context="talk", style="whitegrid", font_scale=0.85)
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 200)

print("✅ All libraries loaded.")


df = pd.read_csv(r'..\data\cleaned\cleaned_data.csv')
print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
df.head(3)


TARGET = "is_canceled"

DATE_COLS   = ["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"]
SOURCE_COLS = [
    "lead_time", "required_car_parking_spaces", "total_of_special_requests",
    "previous_cancellations", "previous_bookings_not_canceled",
    "adr", "adults", "children", "babies",
    "deposit_type", "market_segment", "distribution_channel", "customer_type", "hotel",
] + DATE_COLS


def engineer_features(df_in: pd.DataFrame) -> pd.DataFrame:
    out = df_in.copy()

    pc = out["previous_cancellations"]
    pn = out["previous_bookings_not_canceled"]
    out["prior_cancel_rate"] = (pc + 1) / (pc + pn + 2)
    print("[1] Engineered: prior_cancel_rate = (PC+1) / (PC+PN+2)")

    adr_cap = float(out["adr"].quantile(0.995))
    out["adr_capped"] = out["adr"].clip(upper=adr_cap)
    guests = (out["adults"].fillna(0) + out["children"].fillna(0)
              + out["babies"].fillna(0)).clip(lower=1)
    out["adr_per_person"] = out["adr_capped"] / guests
    out = out.drop(columns=["adr_capped"])
    print(f"[2] Engineered: adr_per_person = adr / guests  (ADR capped at {adr_cap:.2f})")

    out["is_short_lead"] = (out["lead_time"] <= 7).astype(int)
    pct = out["is_short_lead"].mean()
    print(f"[3] Engineered: is_short_lead = (lead_time ≤ 7)  [{pct:.1%} of bookings]")

    top10 = set(out["market_segment"].value_counts().head(10).index)
    out["market_segment"] = out["market_segment"].where(
        out["market_segment"].isin(top10), "OTHER"
    )
    print("[4] market_segment: rare categories merged → 'OTHER'")

    months = {"January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
              "July":7,"August":8,"September":9,"October":10,"November":11,"December":12}
    m = out["arrival_date_month"].map(months).astype(int)
    out["arrival_date"] = pd.to_datetime({
        "year" : out["arrival_date_year"].astype(int),
        "month": m,
        "day"  : out["arrival_date_day_of_month"].astype(int),
    })
    out = out.drop(columns=DATE_COLS)
    print("[5] Reconstructed arrival_date (for splitting only, not a model feature)")

    BASELINE_RAWS = ["adr", "previous_cancellations", "previous_bookings_not_canceled"]

    FINAL_FEATURES = [
        "lead_time", "required_car_parking_spaces", "total_of_special_requests",
        "prior_cancel_rate", "adr_per_person", "is_short_lead",
        "deposit_type", "market_segment", "distribution_channel",
        "customer_type", "hotel",
    ]
    keep = FINAL_FEATURES + BASELINE_RAWS + ["arrival_date", TARGET]
    out = out[[c for c in keep if c in out.columns]].copy()

    print(f"\nFeature engineering complete. Shape: {out.shape}")
    return out

df_eng = engineer_features(df)
df_eng.head(3)


TRAIN_END = pd.Timestamp("2016-12-31")
VAL_END   = pd.Timestamp("2017-04-30")

train = df_eng[df_eng["arrival_date"] <= TRAIN_END].copy()
val   = df_eng[(df_eng["arrival_date"] > TRAIN_END) &
               (df_eng["arrival_date"] <= VAL_END)].copy()
test  = df_eng[df_eng["arrival_date"] > VAL_END].copy()

print("=== Time-based Split Summary ===")
for name, split in [("Train", train), ("Validation", val), ("Test", test)]:
    cr = split[TARGET].mean()
    print(f"  {name:12s}: {len(split):6,} rows  "
          f"({split['arrival_date'].min().date()} → {split['arrival_date'].max().date()})  "
          f"cancel rate = {cr:.1%}")

print(f"\n  Total: {len(train)+len(val)+len(test):,} rows")


FINAL_NUMERIC      = ["lead_time", "required_car_parking_spaces",
                      "total_of_special_requests", "prior_cancel_rate", "adr_per_person"]
FINAL_BINARY       = ["is_short_lead"]
FINAL_CATEGORICAL  = ["deposit_type", "market_segment", "distribution_channel",
                      "customer_type", "hotel"]
ALL_FEATURES       = FINAL_NUMERIC + FINAL_BINARY + FINAL_CATEGORICAL

def xy(df_split: pd.DataFrame):
    
    return df_split[ALL_FEATURES].copy(), df_split[TARGET].copy()

def make_preprocessor():
    
    return ColumnTransformer([
        ("num", StandardScaler(),                                    FINAL_NUMERIC),
        ("bin", "passthrough",                                       FINAL_BINARY),
        ("cat", OneHotEncoder(handle_unknown="ignore",
                              sparse_output=False, drop="first"),    FINAL_CATEGORICAL),
    ], remainder="drop", verbose_feature_names_out=False)

X_train, y_train = xy(train)
X_val,   y_val   = xy(val)
X_test,  y_test  = xy(test)

X_fit, y_fit = xy(pd.concat([train, val], ignore_index=True))

print("Pipeline components defined.")
print(f"  X_train : {X_train.shape}  |  X_val : {X_val.shape}  |  X_test : {X_test.shape}")
print(f"  X_fit (train+val) : {X_fit.shape}")


_pre = make_preprocessor()
_pre.fit(X_train)
X_train_check = _pre.transform(X_train)

all_col_names = _pre.get_feature_names_out()
print(f"Total columns after preprocessing: {len(all_col_names)}")
print(f"  (5 scaled numerics + 1 binary + OHE-expanded categoricals)\n")

num_idx   = list(range(len(FINAL_NUMERIC)))
arr_check = X_train_check[:, num_idx]
scale_df  = pd.DataFrame(arr_check, columns=FINAL_NUMERIC)
print("Scaled numerical stats (mean should be ~0, std should be ~1):")
print(scale_df.agg(["mean", "std"]).round(4).to_string())

n_nan = np.isnan(X_train_check).sum()
print(f"\nNaN count in transformed train set: {n_nan}  {'✅ Clean' if n_nan == 0 else '❌ Problem!'}")


df.to_csv("..\data\preprocessed\preprocessed_data.csv", index=False)
