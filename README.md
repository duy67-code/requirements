# 📊 Hotel Booking Cancellation Prediction

Dự án này sử dụng Python để phân tích dữ liệu khách hàng, từ đó tính toán các chỉ số marketing quan trọng và dự đoán hành vi mua hàng trong tương lai.

## 1. Thành viên nhóm & Phân công công việc

Dự án được thực hiện bởi nhóm 2, với sự đóng góp cụ thể của từng thành viên như sau:

| Họ và Tên | Vai trò | Chi tiết công việc thực hiện |
| :--- | :--- | :--- |
| **Trần Phước Duy** | Trưởng nhóm / Machine Learning Engineer | Tìm kiếm dữ liệu, lên ý tưởng, chỉnh sửa nội dung và viết file `04_Model_Training.ipynb` chạy thuật toán phân đoán và tổng hợp kết quả độ chính xác. |
| **Nguyễn Phương Anh 2** | Feature Engineer | Viết file `03_Feature_Engineering_and_Preprocessing.ipynb`, hỗ trợ làm file `04_Model_Training.ipynb`, tạo ra các features mới, làm scaling, build pipeline and time-based split |
| **Nguyễn Hoàng Lê Khánh** | Data Engineer | Phụ trách file `01_Cleaning.ipynb`, xử lý missing values, loại bỏ leakage data và xuất ra file data chuẩn. |
| **Cao Đức Mạnh** | Data Analyst | Tìm kiếm dữ liệu, viết file `02_EDA.ipynb`, phân tích các insight và đưa ra các actions cần thiết |

## 2. Cấu trúc Repository 

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

## 3. Nguồn dữ liệu (Data Source)
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
| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Log Loss | Brier Score |
|---|---|---|---|---|---|---|---|
| Baseline LR | 0.6874 | 0.6039 | 0.3247 | 0.4223 | 0.7133 | 0.5734 | 0.1983 |
| Engineered LR | 0.7247 | 0.6425 | 0.4911 | 0.5567 | 0.7703 | 0.5357 | 0.1812 |
| **Tuned LR** *(Final Model)* | 0.6352 | 0.4894 | 0.8455 | 0.6200 | 0.7699 | 0.6169 | 0.2191 |
| Random Forest | 0.6756 | 0.5263 | 0.7857 | 0.6303 | 0.7727 | 0.5925 | 0.2045 |
| Gradient Boosting | 0.7205 | 0.6509 | 0.4439 | 0.5278 | 0.7759 | 0.5346 | 0.1828 |

## 6. Kết quả nổi bật (Key Findings)

### Về dữ liệu & Insight kinh doanh

- Tỷ lệ hủy đặt phòng sau khi làm sạch dữ liệu là **28.4%**.
- **`deposit_type = Non Refund`** là yếu tố mạnh nhất: nhóm khách hàng đặt cọc không hoàn tiền có tỷ lệ hủy cực cao (gần 100% trong nhiều trường hợp).
- Khách hàng đặt phòng trước thời gian dài (**lead_time cao**) có nguy cơ hủy cao hơn rõ rệt. Ngược lại, booking ngắn hạn (`is_short_lead`) thường ổn định hơn.
- Khách hàng yêu cầu **nhiều special requests** hoặc **cần chỗ đỗ xe** có xu hướng giữ booking cao hơn (cam kết mạnh).
- `market_segment` và `distribution_channel` có mối tương quan rất cao (0.89), cho thấy có thể giữ một trong hai để tránh đa cộng tuyến.
- Khách hàng có lịch sử hủy trước đó (`prior_cancel_rate` cao) tiếp tục có xu hướng hủy cao hơn ở các lần sau.

### Kết quả Hypothesis Testing
- Tất cả **11 features** được chọn đều có ý nghĩa thống kê mạnh (p-value < 0.05).
- Các biến số (`lead_time`, `adr`, `total_of_special_requests`, ...) đều khác biệt rõ rệt giữa nhóm hủy và không hủy theo kiểm định Mann-Whitney U.
- Tất cả biến phân loại đều có mối liên hệ có ý nghĩa với `is_canceled` theo kiểm định Chi-square.

### Về mô hình

- **Feature Engineering** mang lại cải thiện đáng kể: tăng **+0.057 ROC-AUC** so với Baseline.
- **Gradient Boosting** là mô hình tốt nhất về khả năng xếp hạng với **ROC-AUC = 0.776**.
- **Tuned Logistic Regression** đạt **Recall cao nhất (0.846)**, rất phù hợp cho mục tiêu kinh doanh ưu tiên giảm thiểu mất phòng do bỏ sót booking bị hủy.
- Mô hình tốt nhất cải thiện mạnh so với Baseline:
  - ROC-AUC: **+0.063**
  - Recall: **+0.521** (từ 0.325 lên 0.846)

### Bảng tóm tắt hiệu suất trên Test Set (05–08/2017)

| Model                  | ROC-AUC | Recall  | Precision | F1     | Ghi chú |
|------------------------|---------|---------|-----------|--------|---------|
| Baseline LR            | 0.713   | 0.325   | 0.604     | 0.422  | Mô hình tham chiếu |
| Tuned LR               | 0.770   | **0.846** | 0.489   | 0.620  | Recall cao nhất |
| **Gradient Boosting**  | **0.776** | 0.444 | **0.651** | 0.528  | AUC & Precision tốt nhất |

---

## 5. Hướng dẫn chạy code (How to run)

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

