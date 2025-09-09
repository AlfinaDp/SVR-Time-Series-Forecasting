# SVR-Time-Series-Forecasting.
Forecasting nonlinear time series data using Support Vector Regression (SVR)
## Tools
- Python (scikit-learn, pandas, matplotlib)
- R (e1071, caret, ggplot2)
- Excel for additional preprocessing
## Steps
1. Data preprocessing (Min-Max normalization, lag features)
2. Parameter optimization (Grid Search + Cross Validation)
3. Model training using RBF kernel
4. Evaluation (MSE, RMSE, MAE)
5. Future prediction (one-step forecasting)
## Dataset
- Dummy seasonal additive time series data (various patterns and lengths).
- Real-world datasets:
  - Crude oil price (daily/monthly)
  - Gold price (daily/monthly)
## Results Summary
- Oil price forecasting: RMSE = 0.0278, MAE = ...
- Gold price forecasting: RMSE = 0.0165, MAE = ...
- The SVR model with RBF kernel achieved stable and accurate predictions without signs of overfitting.
## Visualization
![Oil Forecasting](results/oil_forecasting.jpg)
![Gold Forecasting](results/gold_forecasting.jpg)

## How to Run
```bash
pip install -r requirements.txt
python svr_forecast.py
