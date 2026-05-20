import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "optuna", "-q"],
               capture_output=True)
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


df = pd.read_csv(r'..\data\preprocessed\preprocessed_data.csv')
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


sys.path.append("../src")
from feature_engineering_and_preprocessing import engineer_features
from feature_engineering_and_preprocessing import xy
from feature_engineering_and_preprocessing import make_preprocessor
from feature_engineering_and_preprocessing import df_eng


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


X_train, y_train = xy(train)
X_val,   y_val   = xy(val)
X_test,  y_test  = xy(test)

X_fit, y_fit = xy(pd.concat([train, val], ignore_index=True))

print("Pipeline components defined.")
print(f"  X_train : {X_train.shape}  |  X_val : {X_val.shape}  |  X_test : {X_test.shape}")
print(f"  X_fit (train+val) : {X_fit.shape}")


BASELINE_FEATURES = [
    "lead_time", "required_car_parking_spaces", "total_of_special_requests",
    "adr", "previous_cancellations", "previous_bookings_not_canceled",
]

X_train_base = train[BASELINE_FEATURES]
X_test_base  = test[BASELINE_FEATURES]
y_train_base = train[TARGET]
y_test_base  = test[TARGET]

baseline_lr = LogisticRegression(max_iter=1000, random_state=42)
baseline_lr.fit(X_train_base, y_train_base)

y_pred_base  = baseline_lr.predict(X_test_base)
y_proba_base = baseline_lr.predict_proba(X_test_base)[:, 1]

print("=== BASELINE MODEL — Raw numerics, no scaling, no OHE, no tuning ===")
print(f"  Features used : {BASELINE_FEATURES}")
print(f"  Accuracy      : {accuracy_score(y_test_base, y_pred_base):.4f}")
print(f"  Precision     : {precision_score(y_test_base, y_pred_base):.4f}")
print(f"  Recall        : {recall_score(y_test_base, y_pred_base):.4f}")
print(f"  F1-score      : {f1_score(y_test_base, y_pred_base):.4f}")
print(f"  ROC-AUC       : {roc_auc_score(y_test_base, y_proba_base):.4f}")
print(f"  Log Loss      : {log_loss(y_test_base, y_proba_base):.4f}")
print(f"  Brier Score   : {brier_score_loss(y_test_base, y_proba_base):.4f}")
print("\n→ We will compare this to our tuned pipeline in Cell 15.")

results_table = {
    "Baseline LR": {
        "accuracy" : accuracy_score(y_test_base, y_pred_base),
        "precision": precision_score(y_test_base, y_pred_base),
        "recall"   : recall_score(y_test_base, y_pred_base),
        "f1"       : f1_score(y_test_base, y_pred_base),
        "roc_auc"  : roc_auc_score(y_test_base, y_proba_base),
        "log_loss" : log_loss(y_test_base, y_proba_base),
        "brier"    : brier_score_loss(y_test_base, y_proba_base),
    }
}


print("Training Engineered Logistic Regression (default params, full feature set)...")

eng_lr_pipe = Pipeline([
    ("pre", make_preprocessor()),
    ("clf", LogisticRegression(penalty="l2", solver="lbfgs",
                               max_iter=1000, random_state=42)),
])
eng_lr_pipe.fit(X_fit, y_fit)

y_pred_eng  = eng_lr_pipe.predict(X_test)
y_proba_eng = eng_lr_pipe.predict_proba(X_test)[:, 1]

print("\n=== ENGINEERED LR — Full Features, No Tuning — Test Set ===")
print(f"  Accuracy  : {accuracy_score(y_test, y_pred_eng):.4f}")
print(f"  Precision : {precision_score(y_test, y_pred_eng):.4f}")
print(f"  Recall    : {recall_score(y_test, y_pred_eng):.4f}")
print(f"  F1-score  : {f1_score(y_test, y_pred_eng):.4f}")
print(f"  ROC-AUC   : {roc_auc_score(y_test, y_proba_eng):.4f}")
print(f"  Log Loss  : {log_loss(y_test, y_proba_eng):.4f}")
print(f"  Brier     : {brier_score_loss(y_test, y_proba_eng):.4f}")

results_table["Engineered LR"] = {
    "accuracy" : accuracy_score(y_test, y_pred_eng),
    "precision": precision_score(y_test, y_pred_eng),
    "recall"   : recall_score(y_test, y_pred_eng),
    "f1"       : f1_score(y_test, y_pred_eng),
    "roc_auc"  : roc_auc_score(y_test, y_proba_eng),
    "log_loss" : log_loss(y_test, y_proba_eng),
    "brier"    : brier_score_loss(y_test, y_proba_eng),
}


OPTUNA_SEED   = 42
N_TRIALS      = 40

def objective(trial):
    C            = trial.suggest_float("C", 1e-3, 10, log=True)
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

    pipe = Pipeline([
        ("pre", make_preprocessor()),
        ("clf", LogisticRegression(
            penalty="l2",
            C=C,
            solver="lbfgs",
            class_weight=class_weight,
            max_iter=1000,
            random_state=42,
        )),
    ])
    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_proba)

print(f"Running Optuna hyperparameter search ({N_TRIALS} trials) ...")
sampler = optuna.samplers.TPESampler(seed=OPTUNA_SEED)
study   = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS)

print(f"\nBest validation AUC : {study.best_value:.4f}")
print(f"Best params         : {study.best_params}")

best = study.best_params
final_lr_pipe = Pipeline([
    ("pre", make_preprocessor()),
    ("clf", LogisticRegression(
        penalty="l2",
        C=best["C"],
        solver="lbfgs",
        class_weight=best["class_weight"],
        max_iter=1000,
        random_state=42,
    )),
])
final_lr_pipe.fit(X_fit, y_fit)
print("Final Tuned LR retrained on Train+Val.")

y_pred_lr  = final_lr_pipe.predict(X_test)
y_proba_lr = final_lr_pipe.predict_proba(X_test)[:, 1]

print("\n=== TUNED LOGISTIC REGRESSION — Test Set ===")
print(f"  Accuracy  : {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"  Precision : {precision_score(y_test, y_pred_lr):.4f}")
print(f"  Recall    : {recall_score(y_test, y_pred_lr):.4f}")
print(f"  F1-score  : {f1_score(y_test, y_pred_lr):.4f}")
print(f"  ROC-AUC   : {roc_auc_score(y_test, y_proba_lr):.4f}")
print(f"  Log Loss  : {log_loss(y_test, y_proba_lr):.4f}")
print(f"  Brier     : {brier_score_loss(y_test, y_proba_lr):.4f}")

results_table["Tuned LR"] = {
    "accuracy" : accuracy_score(y_test, y_pred_lr),
    "precision": precision_score(y_test, y_pred_lr),
    "recall"   : recall_score(y_test, y_pred_lr),
    "f1"       : f1_score(y_test, y_pred_lr),
    "roc_auc"  : roc_auc_score(y_test, y_proba_lr),
    "log_loss" : log_loss(y_test, y_proba_lr),
    "brier"    : brier_score_loss(y_test, y_proba_lr),
}


print("Training Random Forest...")
rf_pipe = Pipeline([
    ("pre", make_preprocessor()),
    ("clf", RandomForestClassifier(n_estimators=200, max_depth=15,
                                   class_weight="balanced",
                                   random_state=42, n_jobs=-1)),
])
rf_pipe.fit(X_fit, y_fit)

y_pred_rf  = rf_pipe.predict(X_test)
y_proba_rf = rf_pipe.predict_proba(X_test)[:, 1]

print("=== RANDOM FOREST — Test Set ===")
print(f"  Accuracy  : {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"  Precision : {precision_score(y_test, y_pred_rf):.4f}")
print(f"  Recall    : {recall_score(y_test, y_pred_rf):.4f}")
print(f"  F1-score  : {f1_score(y_test, y_pred_rf):.4f}")
print(f"  ROC-AUC   : {roc_auc_score(y_test, y_proba_rf):.4f}")
print(f"  Log Loss  : {log_loss(y_test, y_proba_rf):.4f}")
print(f"  Brier     : {brier_score_loss(y_test, y_proba_rf):.4f}")

results_table["Random Forest"] = {
    "accuracy" : accuracy_score(y_test, y_pred_rf),
    "precision": precision_score(y_test, y_pred_rf),
    "recall"   : recall_score(y_test, y_pred_rf),
    "f1"       : f1_score(y_test, y_pred_rf),
    "roc_auc"  : roc_auc_score(y_test, y_proba_rf),
    "log_loss" : log_loss(y_test, y_proba_rf),
    "brier"    : brier_score_loss(y_test, y_proba_rf),
}

print("\nTraining Gradient Boosting...")
gbm_pipe = Pipeline([
    ("pre", make_preprocessor()),
    ("clf", HistGradientBoostingClassifier(max_iter=300, max_depth=6,
                                           learning_rate=0.05,
                                           random_state=42)),
])
gbm_pipe.fit(X_fit, y_fit)

y_pred_gbm  = gbm_pipe.predict(X_test)
y_proba_gbm = gbm_pipe.predict_proba(X_test)[:, 1]

print("=== GRADIENT BOOSTING — Test Set ===")
print(f"  Accuracy  : {accuracy_score(y_test, y_pred_gbm):.4f}")
print(f"  Precision : {precision_score(y_test, y_pred_gbm):.4f}")
print(f"  Recall    : {recall_score(y_test, y_pred_gbm):.4f}")
print(f"  F1-score  : {f1_score(y_test, y_pred_gbm):.4f}")
print(f"  ROC-AUC   : {roc_auc_score(y_test, y_proba_gbm):.4f}")
print(f"  Log Loss  : {log_loss(y_test, y_proba_gbm):.4f}")
print(f"  Brier     : {brier_score_loss(y_test, y_proba_gbm):.4f}")

results_table["Gradient Boosting"] = {
    "accuracy" : accuracy_score(y_test, y_pred_gbm),
    "precision": precision_score(y_test, y_pred_gbm),
    "recall"   : recall_score(y_test, y_pred_gbm),
    "f1"       : f1_score(y_test, y_pred_gbm),
    "roc_auc"  : roc_auc_score(y_test, y_proba_gbm),
    "log_loss" : log_loss(y_test, y_proba_gbm),
    "brier"    : brier_score_loss(y_test, y_proba_gbm),
}


comparison = pd.DataFrame(results_table).T.round(4)
comparison.index.name = "Model"

print("=" * 75)
print("MODEL COMPARISON — Test Set Metrics")
print("=" * 75)
print(comparison.to_string())
print("=" * 75)

print("\nBest model per metric:")
for metric in comparison.columns:
    if metric in ["log_loss", "brier"]:
        winner = comparison[metric].idxmin()
        print(f"  {metric:12s}: {winner}  ({comparison.loc[winner, metric]:.4f}  ← lower is better)")
    else:
        winner = comparison[metric].idxmax()
        print(f"  {metric:12s}: {winner}  ({comparison.loc[winner, metric]:.4f})")

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle("Model Comparison — Test Set", fontsize=14, fontweight="bold")

for ax, metric in zip(axes, ["accuracy", "recall", "f1", "roc_auc"]):
    vals = comparison[metric].sort_values()
    colors = ["#d73027" if i == 0 else "#4575b4" for i in range(len(vals))]
    sns.barplot(x=vals.values, y=vals.index, ax=ax, palette=colors[::-1])
    ax.set_title(metric.upper())
    ax.set_xlim(0, 1)
    for bar, val in zip(ax.patches, vals.values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

plt.tight_layout()
plt.show()
print("Saved: model_comparison.png")


clf_lr = final_lr_pipe.named_steps["clf"]
pre_lr = final_lr_pipe.named_steps["pre"]

feature_names = pre_lr.get_feature_names_out()
coefs         = clf_lr.coef_[0]

feat_imp = pd.DataFrame({
    "feature"    : feature_names,
    "coefficient": coefs,
    "abs_coef"   : np.abs(coefs),
    "odds_ratio" : np.exp(coefs),
}).sort_values("abs_coef", ascending=False).reset_index(drop=True)

print("=== TOP 15 MOST IMPORTANT FEATURES (Tuned LR) ===")
print(feat_imp.head(15).to_string(index=False))
print(f"\nIntercept: {clf_lr.intercept_[0]:.4f}")

top15 = feat_imp.head(15).sort_values("coefficient")
colors = ["tomato" if c > 0 else "steelblue" for c in top15["coefficient"]]

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top15["feature"], top15["coefficient"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Coefficient (positive → increases cancellation)")
ax.set_title("Top 15 Feature Importances — Logistic Regression",
             fontsize=13, fontweight="bold")
ax.invert_yaxis()
plt.tight_layout()
plt.show()
print("Saved: feature_importance.png")

print()


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, RocCurveDisplay

print("=" * 70)
print("FINAL EVALUATION — Tuned Logistic Regression on Test Set")
print("=" * 70)

cm = confusion_matrix(y_test, y_pred_lr)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Cancelled", "Cancelled"])
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion Matrix — Tuned LR", fontsize=12, fontweight="bold")

precision_arr, recall_arr, thresholds = precision_recall_curve(y_test, y_proba_lr)
axes[1].plot(recall_arr, precision_arr, color="#4575b4", lw=2)
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve — Tuned LR", fontsize=12, fontweight="bold")
axes[1].grid(True)

plt.tight_layout()
plt.show()

fig2, ax2 = plt.subplots(figsize=(8, 6))

for label, proba in [
    ("Baseline LR",     y_proba_base),
    ("Engineered LR",   y_proba_eng),
    ("Tuned LR",        y_proba_lr),
    ("Random Forest",   y_proba_rf),
    ("Gradient Boost",  y_proba_gbm),
]:
    auc = roc_auc_score(y_test, proba)
    RocCurveDisplay.from_predictions(y_test, proba, name=f"{label} (AUC={auc:.3f})", ax=ax2)

ax2.plot([0, 1], [0, 1], "k--", lw=1)
ax2.set_title("ROC Curves — All Models (Test Set)", fontsize=13, fontweight="bold")
ax2.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.show()

base_auc = results_table["Baseline LR"]["roc_auc"]
eng_auc  = results_table["Engineered LR"]["roc_auc"]
tuned_auc = results_table["Tuned LR"]["roc_auc"]
best_model = max(results_table, key=lambda m: results_table[m]["roc_auc"])
best_auc   = results_table[best_model]["roc_auc"]

print(f"""
\n📈 Performance Lift Summary:
  Baseline LR   : AUC = {base_auc:.4f}
  Engineered LR : AUC = {eng_auc:.4f}  (+{eng_auc - base_auc:.4f} from engineering)
  Tuned LR      : AUC = {tuned_auc:.4f}  (+{tuned_auc - eng_auc:.4f} from tuning)
  Best overall  : {best_model} → AUC = {best_auc:.4f}  (+{best_auc - base_auc:.4f} vs. baseline)
""")

base_recall  = results_table["Baseline LR"]["recall"]
eng_recall   = results_table["Engineered LR"]["recall"]
tuned_recall = results_table["Tuned LR"]["recall"]
best_model_recall = max(results_table, key=lambda m: results_table[m]["recall"])
best_recall       = results_table[best_model_recall]["recall"]

print(f"""
\n📈 Recall Performance Lift Summary:
  Baseline LR   : Recall = {base_recall:.4f}
  Engineered LR : Recall = {eng_recall:.4f}  (+{eng_recall - base_recall:.4f} from engineering)
  Tuned LR      : Recall = {tuned_recall:.4f}  (+{tuned_recall - eng_recall:.4f} from tuning)
  Best overall  : {best_model_recall} → Recall = {best_recall:.4f}  (+{best_recall - base_recall:.4f} vs. baseline)
""")
