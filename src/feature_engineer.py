import pandas as pd

def engineer_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Build engineered features, reconstruct arrival_date, return final frame.
    """
    out = df_in.copy()

    # Remove the 1 row where ADR is negative (data entry error)
    before = len(out)
    out = out[out["adr"] >= 0].copy()
    print(f"[1] Dropped {before - len(out)} row(s) with negative ADR.")

    # prior_cancel_rate ────────────────────────────────────────────────────────
    # Raw cancellation counts are noisy for guests with only 1–2 past bookings.
    # Beta smoothing adds pseudo-counts of +1 to stabilise the estimate:
    #   formula = (PC + 1) / (PC + PN + 2)
    # A first-time guest gets 0.5 (neutral); a chronic canceller gets close to 1.
    pc = out["previous_cancellations"]
    pn = out["previous_bookings_not_canceled"]
    out["prior_cancel_rate"] = (pc + 1) / (pc + pn + 2)
    print("[2] Engineered: prior_cancel_rate = (PC+1) / (PC+PN+2)")

    # adr_per_person ───────────────────────────────────────────────────────────
    # The same room rate means very different things for 1 person vs. 4 people.
    # We cap ADR at the 99.5th percentile first to remove extreme outliers,
    # then divide by group size so the price signal is comparable across bookings.
    adr_cap = float(out["adr"].quantile(0.995))
    out["adr_capped"] = out["adr"].clip(upper=adr_cap)
    guests = (out["adults"].fillna(0) + out["children"].fillna(0)
              + out["babies"].fillna(0)).clip(lower=1)
    out["adr_per_person"] = out["adr_capped"] / guests
    out = out.drop(columns=["adr_capped"])
    print(f"[3] Engineered: adr_per_person = adr / guests  (ADR capped at {adr_cap:.2f})")

    # is_short_lead ────────────────────────────────────────────────────────────
    # Last-minute bookings (≤7 days before arrival) almost never cancel —
    # the guest already needs the room imminently.
    # A binary flag captures this threshold effect cleanly.
    out["is_short_lead"] = (out["lead_time"] <= 7).astype(int)
    pct = out["is_short_lead"].mean()
    print(f"[4] Engineered: is_short_lead = (lead_time ≤ 7)  [{pct:.1%} of bookings]")

    # market_segment: merge rare categories ────────────────────────────────────
    # 'Undefined' and 'Aviation' together are < 0.5% of rows.
    # Rare categories make unstable dummy columns — we group them into 'OTHER'.
    top10 = set(out["market_segment"].value_counts().head(10).index)
    out["market_segment"] = out["market_segment"].where(
        out["market_segment"].isin(top10), "OTHER"
    )
    print("[5] market_segment: rare categories merged → 'OTHER'")

    # Reconstruct arrival_date ──────────────────────────────────────────────────
    # We need a proper datetime column to split the data by time.
    # This column will NOT be used as a model feature — only for the split.
    months = {"January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
              "July":7,"August":8,"September":9,"October":10,"November":11,"December":12}
    m = out["arrival_date_month"].map(months).astype(int)
    out["arrival_date"] = pd.to_datetime({
        "year" : out["arrival_date_year"].astype(int),
        "month": m,
        "day"  : out["arrival_date_day_of_month"].astype(int),
    })
    out = out.drop(columns=DATE_COLS)
    print("[6] Reconstructed arrival_date (for splitting only, not a model feature)")

    # Retain raw columns needed for the baseline model (Cell 10)
    # The engineered features reference them, but we keep the originals too
    # so the baseline can use unprocessed inputs.
    BASELINE_RAWS = ["adr", "previous_cancellations", "previous_bookings_not_canceled"]

    # Final column set
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



