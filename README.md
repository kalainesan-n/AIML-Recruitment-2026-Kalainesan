# Air Quality Forecasting Project

**Candidate:** Kalainesan  
**Task:** Air Quality Forecasting (one-hour-ahead CO prediction)  
**Dataset:** UCI Air Quality (hourly observations, March 2004–April 2005)

## Problem Statement
Forecast the next valid hourly CO(GT) measurement using only information available at the current hour. This is a short-term forecasting task where we predict CO concentration one hour ahead based on historical sensor readings and temporal features.

## Dataset
The UCI Air Quality dataset contains hourly averaged responses from an array of 5 metal oxide chemical sensors embedded in an Air Quality Chemical Multisensor Device. The device was located on the field in a significantly polluted area at road level within an Italian city. Data was recorded from March 2004 to February 2005.

Key characteristics:
- Timestamp columns: Date, Time
- Target variable: CO(GT) (carbon monoxide concentration)
- Sensor measurements: PT08.S1(CO), NMHC(GT), C6H6(GT), PT08.S2(NMHC), NOx(GT), PT08.S3(NOx), NO2(GT), PT08.S4(NO2), PT08.S5(O3)
- Environmental variables: T (temperature), RH (relative humidity), AH (absolute humidity)
- Missing values coded as -200

## Approach
The project follows a structured forecasting workflow:
1. **Data preprocessing**: Handle missing values (-200 sentinel) and create proper DateTime index
2. **Exploratory data analysis**: Examine distributions, temporal patterns, and sensor correlations
3. **Feature engineering**: Create lagged features, rolling window features, and temporal features
4. **Model training**: Train a Linear Regression model with time-aware validation
5. **Evaluation**: Compare against a naive persistence baseline using MAE, MSE, RMSE, and R²
6. **Interpretation**: Analyze feature importance, error patterns, and limitations

## Forecasting Setup
- **Target**: CO(GT) at hour t+1 (one hour ahead)
- **Horizon**: Exactly one hour ahead
- **Prediction time**: End of hour t, after observing CO(GT)_t
- **Input constraint**: Only information available at or before time t may be used
- **Missing values**: Not forward-filled; rows with missing required values are excluded from modeling
- **Validation strategy**: 
  - Final holdout: Latest 20% of data chronologically
  - Cross-validation: TimeSeriesSplit (5 folds) on remaining 80%
  - No random shuffling to prevent temporal leakage

## Feature Engineering
Features constructed at prediction time t:
- **Lagged measurements**: 
  - CO_current: CO(GT) at time t
  - CO_lag_1: CO(GT) at time t-1
  - CO_lag_2: CO(GT) at time t-2
- **Rolling averages**:
  - CO_roll_mean_3: Mean of CO(GT) from t-1 to t-3 (excludes current reading)
- **Temporal features**:
  - Hour_sin, Hour_cos: Cyclic encoding of hour of day
  - DayOfWeek: Day of week (0=Monday, 6=Sunday)
  - Month: Month of year (1-12)

## Model and Validation Strategy
- **Model**: Linear Regression (chosen for interpretability and simplicity)
- **Preprocessing**: StandardScaler applied to numeric features within a scikit-learn Pipeline
- **Validation**: 
  - Five-fold TimeSeriesSplit on training data (chronological folds)
  - Final evaluation on held-out test set (most recent 20%)
- **Baseline**: Naive persistence model predicting CO_target_t_plus_1 = CO_current

## Results
### Final Held-Out Test Set Performance
| Metric | Linear Regression | Naive Persistence | Improvement |
|--------|------------------|-------------------|-------------|
| MAE    | 0.4652           | 0.5208            | 10.7% ↓     |
| MSE    | 0.4815           | 0.6408            | 24.9% ↓     |
| RMSE   | 0.6939           | 0.8005            | 13.3% ↓     |
| R²     | 0.7377           | 0.6509            | 13.3% ↑     |

The linear regression model outperforms the naive persistence baseline, demonstrating that the engineered features capture predictive signal beyond simply using the current CO value.

### Time-Series Cross-Validation Performance
| Metric | Linear Regression | Naive Persistence |
|--------|------------------|-------------------|
| MAE    | 0.4927           | 0.5242            |
| MSE    | 0.5333           | 0.6313            |
| RMSE   | 0.7192           | 0.7847            |
| R²     | 0.7302           | 0.6749            |

Consistent performance improvement across validation folds confirms the model's reliability.

## Key Findings
1. **Model effectiveness**: Linear regression outperforms naive persistence baseline on held-out test set (MAE: 0.4652 vs 0.5208; RMSE: 0.6939 vs 0.8005; R²: 0.7377 vs 0.6509).
2. **Performance gains**: MAE reduced by 10.7% and RMSE by 13.3% relative to naive baseline.
3. **Temporal patterns**: Average observed CO levels are higher on weekdays (mean 2.359) than weekends (mean 1.655).
4. **Data quality impact**: Approximately 25.9% of candidate rows (2424 out of 9356) excluded due to missing CO(GT) values in current, lagged, or target variables.
5. **Feature importance**: Current CO level (CO_current) is the most influential feature (largest standardized coefficient magnitude: 1.5354).
6. **Sensor correlation**: Non-methane hydrocarbons (C6H6(GT)) exhibit strong linear correlation with CO(GT) (r = 0.931), suggesting potential predictive information.

## Limitations
1. **Missing data bias**: Exclusion of ~25.9% of rows reduces usable sample size and may introduce bias if missingness is not random.
2. **Relationship stability**: Linear model assumes time-invariant relationship between features and target, which may not hold over seasonal/long-term changes.
3. **Horizon limitation**: Evaluation limited to one-hour forecasting; multi-hour horizons would require different modeling approaches.
4. **Sensor specificity**: Model based on single pollutant (CO) and specific location/time period.

## One Proposed Improvement
- Test C6H6(GT) as an additional prediction-time feature. Its strong correlation with CO(GT) suggests it may contain useful predictive information, but its actual forecasting benefit should be evaluated rather than assumed.

## Technologies Used
- Python 3.x
- pandas (data manipulation)
- numpy (numerical computing)
- scikit-learn (machine learning, pipelines, metrics)
- matplotlib & seaborn (visualization)
- Jupyter Notebook (interactive development)

## Three Specific Things Learned
1. **Temporal leakage prevention**: Constructing time series features requires careful shifting (e.g., using shift(1) before rolling windows) to ensure only historical information is used for prediction.
2. **Chronological validation importance**: Random cross-validation would invalidate time series forecasting results by allowing future data to influence past predictions; TimeSeriesSplit maintains proper temporal order.
3. **Feature interpretability vs. performance trade-off**: While linear regression may not achieve the highest possible accuracy, its interpretability (via coefficients) provides valuable insights into which features drive predictions, which is crucial for domain understanding and trust in the model.

## Real Challenge and Solution
**Challenge**: The dataset contained significant missing values (1683 missing CO readings out of 9357 total rows, ~18%), coded as -200 sentinel values. Simply removing all rows with any missing values would drastically reduce the dataset, while imputation risks introducing bias.

**Solution**: 
1. Properly identified and converted -200 sentinel values to NaN for correct handling
2. Created the modeling dataset by requiring only that current, lagged, and target CO values were all present (rather than imputing)
3. Quantified and reported the exclusion rate (~25.9% of candidate rows) as a limitation
4. Avoided forward-fill imputation for CO values to prevent creating artificial labels from old measurements
This approach preserved data integrity while transparently communicating the impact of missing values on the analysis.