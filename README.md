# Customer Lifetime Value Prediction

A regression model that estimates the lifetime value of auto-insurance customers, so marketing can prioritise retention and acquisition spend on the customers who actually drive portfolio value.

**Headline result:** 96.6% of the model's predictive power comes from just three features — and the strongest one had a correlation of only 0.025 with the target until it was re-encoded.

---

## Business context

Customer Lifetime Value (CLV) measures how much profit a customer generates over the whole relationship. Knowing it lets a company answer two concrete questions:

- How much can we justify spending to acquire this customer?
- Which existing customers are worth investing in to retain?

Without CLV estimates, marketing spends the same amount on every customer. In this portfolio the top 10% of customers hold roughly 31% of total value, so uniform spending systematically under-invests in the accounts that matter most.

---

## Dataset

`data_customer_lifetime_value.csv` — 5,669 auto-insurance customers, 10 features and 1 target.

| Type | Features |
|---|---|
| Categorical | Vehicle Class, Coverage, Renew Offer Type, Employment Status, Marital Status, Education |
| Numerical | Number of Policies, Monthly Premium Auto, Total Claim Amount, Income |
| **Target** | **Customer Lifetime Value** (continuous) |

Target distribution: mean 8,030 · median 5,800 · min 1,898 · max 83,325 · **skew 3.06**

No missing values. 618 duplicate rows (10.9%) were removed — see below.

---

## Project structure

```
├── clv.ipynb                          # full analysis and modelling
├── app.py                             # Streamlit prediction app
├── model_clv.sav                      # pickled pipeline (1.2 MB)
├── data_customer_lifetime_value.csv   # input data
├── requirements.txt
└── README.md
```

---

## Approach

1. **Data exploration** — duplicates, nulls, dtypes, value counts
2. **EDA** — target distribution, numeric correlations, categorical group means, multicollinearity (VIF)
3. **Cleaning** — deduplication and feature type conversion, both justified by EDA evidence
4. **Split before preprocessing** — `train_test_split` first, all transforms fitted inside a `Pipeline` to prevent leakage
5. **Model comparison** — 10 algorithms at default parameters across 6 metrics
6. **Hyperparameter tuning** — `RandomizedSearchCV`, 30 iterations, 5-fold CV, on all 9 tunable models
7. **Evaluation** — single pass on the held-out test set, plus residual analysis and feature importance
8. **Deployment** — pipeline pickled and served through Streamlit

---

## Key findings

### 1. Number of Policies is non-monotonic — and re-encoding it tripled linear model performance

Mean CLV by policy count is not a straight line:

| Policies | Mean CLV |
|---|---|
| 1 | 3,536 |
| **2** | **15,848** |
| 3–9 | ~7,100 (flat) |

Because the relationship rises then falls, Pearson correlation reads just **0.025** — the signal is invisible to a linear correlation check. Binning the feature into `1 / 2 / 3+` makes it accessible:

| Encoding | Linear Regression R² |
|---|---|
| Numeric (as-is) | **0.171** |
| Binned | **0.645** |

A **3.8× improvement from feature engineering alone**, with no change to the algorithm.

Notably, tree-based models were largely unaffected — they can already split a numeric column at multiple thresholds. Feature engineering here is **model-dependent**, not universally beneficial.

### 2. Monthly Premium is almost entirely determined by other features

VIF on the one-hot encoded feature matrix:

| Feature | VIF |
|---|---|
| **Monthly Premium Auto** | **26.98** |
| Vehicle Class — Luxury SUV | 8.67 |
| Vehicle Class — Luxury Car | 7.99 |
| Vehicle Class — SUV | 7.53 |
| EmploymentStatus — Employed | 6.79 |

Vehicle Class and Coverage together explain **96% of the variance** in Monthly Premium — unsurprising, since insurers price premiums *from* vehicle type and coverage tier. This makes linear regression coefficients unreliable for interpretation, though tree-based models are unaffected.

Checking VIF on numeric columns alone would have shown all values below 2.1 and led to the wrong conclusion. The categorical dummies must be included.

### 3. The 618 duplicates were genuine record duplication, not coincidence

832 rows were involved, collapsing to 214 distinct records repeated 2–6 times. The decisive evidence: duplicated rows matched to **six decimal places** on both Total Claim Amount and CLV (e.g. `574.024018` / `16301.9676`). Distinct customers sharing categorical attributes would still differ on continuous values. These were dropped.

### 4. Income = 0 is structural, not missing

All 1,429 zero-income rows are `EmploymentStatus = Unemployed`. No imputation was applied — replacing these with the mean would have assigned fictional salaries to customers who genuinely have none.

---

## Model comparison

### Default parameters (5-fold CV on training data)

| Model | R² | RMSE | MAE | MAPE% | MedAE |
|---|---|---|---|---|---|
| GradientBoosting | **0.667** | **4,036** | 1,800 | 14.4 | 263 |
| Lasso | 0.645 | 4,171 | 2,169 | 25.4 | 821 |
| Ridge | 0.645 | 4,171 | 2,171 | 25.4 | 822 |
| LinearRegression | 0.645 | 4,172 | 2,174 | 25.5 | 828 |
| RandomForest | 0.644 | 4,162 | **1,711** | **12.4** | **105** |
| Bagging | 0.613 | 4,342 | 1,758 | 12.7 | 112 |
| AdaBoost | 0.590 | 4,460 | 2,775 | 33.7 | 1,368 |
| XGBoost | 0.588 | 4,482 | 1,861 | 14.8 | 232 |
| KNN | 0.484 | 5,025 | 2,445 | 26.2 | 752 |
| DecisionTree | 0.318 | 5,720 | 2,020 | 14.3 | 124 |

Two observations worth noting:

- **RMSLE could not be computed for Linear, Ridge, or Lasso** — all three predicted *negative* CLV in some folds (minimum −69). Since observed CLV never falls below 1,898, this is a structural limitation of unconstrained linear models, not a scoring artefact. Tree models cannot produce negative predictions.
- **Model ranking depends on the metric.** GradientBoosting wins R²/RMSE (which square errors and therefore weight high-value customers heavily); RandomForest wins MAE/MAPE/MedAE (which weight all customers equally).

### After hyperparameter tuning

| Model | Tuned RMSE | Change vs default |
|---|---|---|
| **RandomForest** | **3,922** | −240 |
| XGBoost | 3,942 | **−540** |
| GradientBoosting | 3,953 | −83 |
| DecisionTree | 4,010 | −1,710 |
| Bagging | 4,078 | −264 |
| AdaBoost | 4,082 | −378 |
| Lasso | 4,169 | −2 |
| Ridge | 4,170 | −1 |
| KNN | 4,914 | −111 |

XGBoost moved from 8th to 2nd. Its poor default result was an artefact of aggressive default settings (`max_depth=6`, `learning_rate=0.3`) overfitting 4,040 training rows — not a property of the algorithm. Tuning only the default winner would have produced a misleading conclusion.

The top three models fall within **31 RMSE of each other**, well inside the fold-to-fold standard deviation. They are statistically equivalent; Random Forest was selected as the nominal best.

---

## Final model

```python
RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=10, random_state=42)
```

Wrapped in a `Pipeline` with `OneHotEncoder(drop='first')` for categoricals and `StandardScaler()` for numerics.

### Test set performance (1,011 unseen customers)

| Metric | Value |
|---|---|
| R² | **0.681** |
| RMSE | **3,895** |
| MAE | **1,686** |
| MAPE | **14.1%** |
| MedAE | **287** |

Test RMSE (3,895) came in slightly better than cross-validated RMSE (3,922), indicating the tuning process did not overfit the validation folds.

### Feature importance

**96.6% of predictive power comes from three features:**

| Feature | Importance |
|---|---|
| Policy Bin = 2 | ~57% |
| Monthly Premium Auto | ~32% |
| Policy Bin = 3+ | ~7% |

Demographic features — Education, Marital Status, Employment Status, and even Vehicle Class — contribute almost nothing once policy count and premium level are known.

The single most important feature is the one created during feature engineering. The EDA anomaly turned out to be the model.

---

## Limitations

Stated plainly, because they affect how the output should be used.

**1. The model systematically under-predicts high-value customers.**

Residuals segmented by actual CLV:

| Segment | RMSE | MAE | Mean bias |
|---|---|---|---|
| Overall | 3,895 | 1,686 | −56 |
| Bottom 90% | 1,989 | 971 | −767 |
| **Top 10%** | **10,728** | **8,060** | **+6,276** |

The near-zero overall bias masks two opposing errors. High-value customers are under-estimated by an average of 6,276.

**2. There is a hard prediction ceiling.** Maximum predicted CLV is 36,547 against a maximum actual of 73,226. Random Forest predictions are averages of observed training values, so extreme values are always pulled toward the group mean.

**3. Error variance grows sharply with predicted value** — residual standard deviation rises from ~200 below 5,000 to ~13,400 above 20,000. All 31 cases with residuals above 10,000 belong to the `Policy Bin = 2` segment, which itself spans 6,049 to 83,325 in actual CLV. The available features identify *which segment* a customer belongs to but contain no signal separating high from low value *within* that segment.

**4. CLV here is a derived figure, not an observed outcome.** The dataset has no tenure or date column, so the model reverse-engineers an existing business calculation rather than forecasting future behaviour. Its value lies in identifying *which attributes are associated with value*, not in predicting how value will change.

**Practical implication:** the model is reliable for **ranking customers into value tiers** and unreliable for **precise point estimates on high-value customers**.

---

## Business recommendation

Predicted CLV maps onto portfolio percentiles to give an actionable prioritisation:

| Threshold | CLV | Segment | Recommended action |
|---|---|---|---|
| — | < 5,838 | Lower half | Automated renewal only |
| median | ≥ 5,838 | Upper half | Standard servicing |
| p90 | ≥ 15,641 | Top 10% | Targeted offers |
| p95 | ≥ 21,922 | Top 5% | Retention investment |
| p99 | ≥ 36,261 | Top 1% | Priority retention |

Only 51 customers fall in the top 1%, but they carry a disproportionate share of total portfolio value.

**An open question the data cannot answer:** holding exactly two policies is the strongest predictor of high CLV — but is the relationship causal? Would encouraging a second policy actually raise a customer's value, or do already-valuable customers simply tend to buy two? Answering that requires an experiment, not this dataset.

---

## Running the project

### Notebook

```bash
pip install -r requirements.txt
jupyter notebook clv.ipynb
```

### Streamlit app

```bash
streamlit run app.py
```

The app takes customer attributes as input and returns a predicted CLV, its portfolio segment, and an indicative uncertainty range scaled to the residual analysis above.

### Requirements

```
streamlit
pandas
numpy
seaborn
matplotlib
scikit-learn==1.9.0
statsmodels
xgboost
```

The scikit-learn version is pinned because `model_clv.sav` was pickled with it. A version mismatch can cause unpickling to fail or produce subtly incorrect predictions.

---

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · statsmodels · seaborn · matplotlib · Streamlit

---

## What I'd do next

- Test a log-transformed target to address the heteroscedasticity and prediction ceiling — the remedy prescribed by Gauss-Markov diagnostics for non-constant error variance
- Add SHAP values for per-customer explanations rather than global importance only
- Investigate the two-policy segment specifically; its internal variance is the single largest source of model error
- Acquire tenure and margin data, which would allow CLV to be modelled as a forward-looking outcome rather than reconstructed from a formula
