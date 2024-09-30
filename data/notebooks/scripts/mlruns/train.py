import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

mlflow.set_tracking_uri("http://localhost:8000")

df = pd.read_csv("data3.csv")

X = df.drop(columns=['medv'])
y = df['medv']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("Experiment Tracking with MLflow")

best_model = None
best_model_name = None
best_mse = float('inf')

with mlflow.start_run(run_name="Linear Regression"):
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_param("model", "Linear Regression")
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color="blue", label="Predicted")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label="Ideal Fit")
    plt.xlabel('Actual medv')
    plt.ylabel('Predicted medv')
    plt.title('Actual vs Predicted medv (Linear Regression)')
    plt.legend()
    plt.savefig("linear_regression_comparison.png")
    plt.clf()

    signature = infer_signature(X_train, y_train)
    mlflow.sklearn.log_model(reg, "linear_regression_model", signature=signature)

    if mse < best_mse:
        best_mse = mse
        best_model = reg
        best_model_name = "LinearRegressionModel"

with mlflow.start_run(run_name="Random Forest"):
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_train, y_train)
    y_pred_rf = rf_reg.predict(X_test)

    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    mlflow.log_param("model", "Random Forest")
    mlflow.log_metric("mse", mse_rf)
    mlflow.log_metric("r2", r2_rf)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_rf, color="green", label="Predicted")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label="Ideal Fit")
    plt.xlabel('Actual medv')
    plt.ylabel('Predicted medv')
    plt.title('Actual vs Predicted medv (Random Forest)')
    plt.legend()
    plt.savefig("random_forest_comparison.png")
    plt.clf()

    signature_rf = infer_signature(X_train, y_train)
    mlflow.sklearn.log_model(rf_reg, "random_forest_model", signature=signature_rf)

    if mse_rf < best_mse:
        best_mse = mse_rf
        best_model = rf_reg
        best_model_name = "RandomForestModel"

if best_model is not None:
    with mlflow.start_run(run_name="Best Model"):
        signature_best = infer_signature(X_train, y_train)
        mlflow.sklearn.log_model(best_model, "best_model", registered_model_name=best_model_name, signature=signature_best)
        mlflow.log_metric("best_mse", best_mse)
