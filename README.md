# 📊 Hotel Booking Cancellation Prediction

This project uses Python to analyze customer data, calculate important marketing metrics, and predict future purchasing behavior.

## 1. Team Members & Task Allocation

This project was carried out by Group 2, with specific contributions from each member as follows:

| Full Name | Role | Details of Work Performed |
| :--- | :--- | :--- |
| **Trần Phước Duy** | Team Leader / Machine Learning Engineer | Searched for data, came up with ideas, edited content and wrote the `04_Model_Training.ipynb` file to run the prediction algorithm and summarize accuracy results. |
| **Nguyễn Phương Anh 2** | Feature Engineer | Wrote the `03_Feature_Engineering_and_Preprocessing.ipynb` file, supported the `04_Model_Training.ipynb` file, created new features, performed scaling, built the pipeline and time-based split. |
| **Nguyễn Hoàng Lê Khánh** | Data Engineer | Responsible for the `01_Cleaning.ipynb` file, handled missing values, removed leakage data and exported a clean dataset. |
| **Cao Đức Mạnh** | Data Analyst | Searched for data, wrote the `02_EDA.ipynb` file, analyzed insights and proposed necessary actions. |

## 2. Repository Structure

```bash
.
├── data/                                    <- Dataset storage
│   ├── raw/
│   │   └── raw_data.csv                     <- Original downloaded dataset
│   │
│   ├── cleaned/
│   │   └── cleaned_data.csv                 <- Cleaned dataset after leakage removal
│   │
│   └── processed/
│       └── preprocessed_data.csv            <- Feature-engineered & preprocessed dataset
│
├── notebooks/                               <- Jupyter Notebook workflow
│   ├── 01_Cleaning.ipynb                    <- Data cleaning, leakage audit, sanity checks
│   ├── 02_EDA.ipynb                         <- Exploratory Data Analysis & visualisation
│   ├── 03_Feature_Engineering_and_Preprocessing.ipynb
│   │                                        <- Feature engineering & preprocessing pipeline
│   └── 04_Model_Training.ipynb              <- Model training, tuning, and evaluation
│
├── src/                                     <- Reusable Python scripts
│   ├── cleaning.py
│   ├── eda.py
│   ├── feature_engineering_and_preprocessing.py
│   └── model_training.py
│
├── models/                                  <- Saved trained models
│   ├── best_params.json
│   └── model.pkl
│
└── README.md                                <- Project documentation

## 3. Project Pipeline

Raw Hotel Bookings  ──►  Data Cleaning & Logic Fixing
                                   │
                                   ▼
                       Exploratory Data Analysis (EDA)
                       & Statistical Hypothesis Testing
                                   │
                                   ▼
                    Feature Engineering & Preprocessing
        (prior_cancel_rate, adr_per_person, is_short_lead, OHE, Scaling)
                                   │
                                   ▼
                       Time-based Train/Val/Test Split
                (2015-07 → 2016-12 | 2017-01→04 | 2017-05→08)
                                   │
                                   ▼
                          Model Training & Tuning
            (Logistic Regression, Random Forest, Gradient Boosting)
                                   │
                                   ▼
                    Model Evaluation & Comparison
            (ROC-AUC, Recall, Precision, Business Insights)

```
- Raw Hotel Bookings (119,390 rows): Original dataset downloaded from Kaggle containing 32 features, including booking details, customer information, and reservation status.
- Data Cleaning & Logic Fixing: Handled missing values, removed invalid records (negative ADR, zero guests), eliminated duplicate bookings, and corrected data inconsistencies to ensure high data quality.
- Exploratory Data Analysis (EDA) & Hypothesis Testing: Performed in-depth analysis of feature distributions, cancellation patterns, and relationships. Conducted statistical tests (Mann-Whitney U for numerical features and Chi-square for categorical features) to validate feature significance.
- Feature Engineering & Preprocessing: Created new powerful features: prior_cancel_rate, adr_per_person, and is_short_lead. Applied One-Hot Encoding for categorical variables and Standard Scaling for numerical features.
- Time-based Train/Val/Test Split: Split data chronologically to reflect real-world deployment:
    + Train: July 2015 – December 2016
    + Validation: January 2017 – April 2017
    + Test: May 2017 – August 2017

- Model Training & Tuning: Developed multiple models including Baseline Logistic Regression, Engineered Logistic Regression, Tuned Logistic Regression (using Optuna), Random Forest, and Gradient Boosting. Focused on optimizing both discrimination power (ROC-AUC) and business-relevant metrics (Recall).
- Model Evaluation & Comparison: Evaluated all models on the unseen test set using comprehensive metrics: ROC-AUC, Recall, Precision, F1-score, Log Loss, and Brier Score. Analyzed trade-offs and selected the best-performing models for different business priorities.
... 

## 3. Data Source
- **Source:** [Hotel Booking Demand — Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- **Total observations (rows):** 119,390
- **Total attributes (columns):** 32 *(original)* → **11 features** after selection & engineering
- **Target variable:** `is_canceled` — Binary class *(0 = Not cancelled, 1 = Cancelled)*
- **Time span:** July 2015 → August 2017
- **Overall cancellation rate:** 37.0%
- **Numerical Features:** `lead_time`, `required_car_parking_spaces`, `total_of_special_requests`,
  `prior_cancel_rate` *(engineered)*, `adr_per_person` *(engineered)*
- **Binary Feature:** `is_short_lead` *(engineered)*
- **Categorical Features:** `hotel`, `deposit_type`, `market_segment`,
  `distribution_channel`, `customer_type`

| Variable Name | Description | Feature Type | Example |
| :--- | :--- | :--- | :--- |
| **is_canceled** ⭐ | Whether the booking was cancelled *(Target)* | Binary | `0` / `1` |
| **lead_time** | Number of days between booking date and arrival date | Numerical | `45` (Days) |
| **required_car_parking_spaces** | Number of parking spaces requested by the guest | Numerical | `1` (Spaces) |
| **total_of_special_requests** | Total number of special requests made (e.g. room floor, bed type) | Numerical | `2` (Requests) |
| **prior_cancel_rate** | Guest's historical cancellation rate, computed with Beta smoothing: $(PC+1)/(PC+PN+2)$ | Numerical *(engineered)* | `0.67` |
| **adr_per_person** | Average daily room rate divided by number of guests; ADR capped at 99.5th percentile | Numerical *(engineered)* | `58.5` (€/person) |
| **is_short_lead** | Binary flag: `1` if booking was made ≤ 7 days before arrival, `0` otherwise | Binary *(engineered)* | `0` / `1` |
| **hotel** | Type of hotel property | Categorical | `Resort Hotel` / `City Hotel` |
| **deposit_type** | Financial commitment type at booking time | Categorical | `No Deposit` / `Non Refund` / `Refundable` |
| **market_segment** | Market segment the booking originated from | Categorical | `Online TA` / `Direct` / `Groups` |
| **distribution_channel** | Channel through which the booking was distributed | Categorical | `TA/TO` / `Direct` / `Corporate` |
| **customer_type** | Classification of the booking customer | Categorical | `Transient` / `Contract` / `Group` |


## 4. Metrics
Most Important Metric: ROC-AUC

In a hotel cancellation context, the class distribution is **imbalanced** and the cost of errors is asymmetric.  
A missed cancellation (False Negative) means a room is blocked and revenue is lost; a false alarm (False Positive) merely triggers an unnecessary follow-up.

**ROC-AUC** is our primary metric because it:
- Is **threshold-independent** — measures ranking ability across all decision boundaries
- Handles **class imbalance** better than raw accuracy
- Lets operations teams choose their own precision/recall trade-off at deployment

| Metric | Formula | Why We Use It |
|---|---|---|
| **ROC-AUC** *(primary)* | Area under the ROC curve (TPR vs FPR at all thresholds) | Threshold-free ranking metric; robust to class imbalance; lets the hotel choose its own operating point |
| **Recall** *(secondary)* | TP / (TP + FN) | Minimises missed cancellations, the costliest error for revenue management |
| **Precision** | TP / (TP + FP) | Reduces false alarms that waste staff resources |
| **F1-Score** | 2 · Precision · Recall / (Precision + Recall) | Harmonic mean balancing precision and recall; useful for threshold-specific comparison |
| **Accuracy** | (TP + TN) / N | Overall correctness; reported for completeness but misleading on imbalanced data |
| **Log Loss** | −mean[y·log(p) + (1−y)·log(1−p)] | Penalises overconfident wrong predictions; measures calibration of probabilities |
| **Brier Score** | mean[(y − p)²] | Squared error on probabilities; lower is better; reward well-calibrated models |


## 5. Model Comparison (Test Set: May–Aug 2017, n ≈ 22,177)
### 5.1. Models Results
| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Log Loss | Brier Score |
|---|---|---|---|---|---|---|---|
| Baseline LR | 0.6874 | 0.6039 | 0.3247 | 0.4223 | 0.7133 | 0.5734 | 0.1983 |
| Engineered LR | 0.7247 | 0.6425 | 0.4911 | 0.5567 | 0.7703 | 0.5357 | 0.1812 |
| **Tuned LR** *(Final Model)* | 0.6352 | 0.4894 | 0.8455 | 0.6200 | 0.7699 | 0.6169 | 0.2191 |
| Random Forest | 0.6756 | 0.5263 | 0.7857 | 0.6303 | 0.7727 | 0.5925 | 0.2045 |
| Gradient Boosting | 0.7205 | 0.6509 | 0.4439 | 0.5278 | 0.7759 | 0.5346 | 0.1828 |

---
### 5.2. Improvement vs Baseline LG
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Log Loss | Brier |
|---|---|---|---|---|---|---|---|
| Engineered LR | +5.4% | +6.4% | +51.2% | +31.8% | +8.0% | +6.6% | +8.6% |
| Tuned LR | -7.6% | -19.0% | +160.4% | +46.8% | +7.9% | -7.6% | -10.4% |
| Random Forest | -1.7% | -12.9% | +141.8% | +49.2% | +8.3% | -3.3% | -3.1% |
| Gradient Boosting | +4.8% | +7.8% | +36.6% | +24.9% | +8.8% | +6.8% | +7.8% |

Percentages show change relative to **Baseline LR**.  
For **Log Loss** and **Brier**, the sign is flipped so positive values still indicate better performance.

---
### 5.3. Step-by-Step Pipeline Lift
| Stage | What was added | ROC-AUC | Δ AUC | Recall | Δ Recall |
|---|---|---|---|---|---|
| Baseline LR | Raw numeric features only, no scaling, no tuning | 0.7133 | — | 0.3249 | — |
| Engineered LR | Full feature engineering + preprocessing pipeline | 0.7703 | +0.0570 | 0.4911 | +0.1662 |
| Tuned LR | Optuna hyperparameter tuning (40 trials) | 0.7700 | -0.0003 | 0.8462 | +0.3551 |
| Random Forest | Ensemble: 200 trees, balanced class weight | 0.7727 | +0.0027 | 0.7857 | -0.0605 |
| Gradient Boosting | Ensemble: 300 iterations, histogram GBM | 0.7759 | +0.0032 | 0.4439 | -0.3418 |

---
## 6. Key Findings

### About the Data & Business Insights

- The cancellation rate after data cleaning is **28.4%**.
- **`deposit_type = Non Refund`** is the strongest factor: customers who made non-refundable deposits have an extremely high cancellation rate (nearly 100% in many cases).
- Customers who book far in advance (**high lead_time**) have a significantly higher risk of cancellation. Conversely, short-term bookings (`is_short_lead`) tend to be more stable.
- Customers who make **many special requests** or **require parking spaces** are more likely to keep their bookings (stronger commitment).
- Customers with a history of cancellations (`prior_cancel_rate` high) continue to have a higher tendency to cancel in subsequent bookings.

### Hypothesis Testing Results
- All **11 selected features** are statistically significant (p-value < 0.05).
- Numerical variables (`lead_time`, `adr`, `total_of_special_requests`, ...) show clear differences between the canceled and non-canceled groups according to the Mann-Whitney U test.
- All categorical variables have a significant relationship with `is_canceled` according to the Chi-square test.

### About the Models

- **Feature Engineering** brought significant improvement: an increase of **+0.057 ROC-AUC** compared to the Baseline.
- **Gradient Boosting** is the best model in terms of ranking ability with **ROC-AUC = 0.776**.
- **Tuned Logistic Regression** achieved the **highest Recall (0.846)**, which is very suitable for business goals that prioritize minimizing lost rooms due to missed cancellations.
- The best model shows strong improvement over the Baseline:
  - ROC-AUC: **+0.063**
  - Recall: **+0.521** (from 0.325 to 0.846)

### Summary of Performance on Test Set (05–08/2017)

| Model                  | ROC-AUC | Recall  | Precision | F1     | Note |
|------------------------|---------|---------|-----------|--------|---------|
| Baseline LR            | 0.713   | 0.325   | 0.604     | 0.422  | Reference model |
| Tuned LR               | 0.770   | **0.846** | 0.489   | 0.620  | Highest Recall |
| **Gradient Boosting**  | **0.776** | 0.444 | **0.651** | 0.528  | Best AUC & Precision |
---

## 5. How to run

### Prerequisites

**Python version:** 3.8+

Install all required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy optuna ipykernel
```
---

### Running the Notebooks (recommended)

Run the notebooks **in order** from the `Notebooks/` directory. Each step depends on the output of the previous one.

**Step 1 — Data Cleaning**
```
01_Cleaning.ipynb
```
- Reads `../data/raw/raw_data.csv`
- Outputs `../data/cleaned/cleaned_data.csv`

**Step 2 — Exploratory Data Analysis**
```
02_EDA.ipynb
```
- Reads `../data/cleaned/cleaned_data.csv`
- No file output — visualisations only

**Step 3 — Feature Engineering & Preprocessing**
```
03_Feature_Engineering_and_Preprocessing.ipynb
```
- Reads `../data/cleaned/cleaned_data.csv`
- Outputs `../data/preprocessed/preprocessed_data.csv`

**Step 4 — Model Training**
```
04_Model_Training.ipynb
```
- Reads `../data/preprocessed/preprocessed_data.csv`
- Imports helper functions from `../src/feature_engineering_and_preprocessing.py`
- Outputs trained model artifacts and evaluation plots

---

### Running as Python Scripts (alternative)

If you prefer running plain `.py` files instead of notebooks:

```bash
cd requirements

python src/cleaning.py
python src/eda.py
python src/feature_engineering_and_preprocessing.py
python src/model_training.py
```

> Make sure you run them from the **project root** (`requirements/`) so that relative paths like `../data/...` resolve correctly.

---

### Notes

- **Optuna** is used for hyperparameter tuning in steps 3 and 4. The notebooks auto-install it via `pip` on the first run — or install it manually with `pip install optuna`.
- **Step 4** imports `engineer_features`, `xy`, `make_preprocessor`, and `df_eng` directly from `src/feature_engineering_and_preprocessing.py`. Ensure that file is present in `src/` before running the model training notebook.
- All file paths in the notebooks are **relative**, so always open VS Code (or Jupyter) from the project root, not from inside the `Notebooks/` folder.

