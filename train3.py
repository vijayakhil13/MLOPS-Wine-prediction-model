from __future__ import annotations
import os
import argparse
import math
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

def parse_args():
    p = argparse.ArgumentParser("Simple MLflow demo (wine prediction)")
    p.add_argument("--csv", default="data/wine_sample.csv", help="Path to CSV")
    p.add_argument("--target", default="quality", help="Target column")
    p.add_argument("--experiment", default="wine-quality", help="MLflow experiment")
    p.add_argument("--run", default="run-1", help="MLflow run name")
    p.add_argument("--n-estimators", type=int, default=50, help="n_estimators")
    p.add_argument("--max-depth", type=int, default=5, help="max_depth")
    p.add_argument("--test-size", type=float, default=0.2, help="test_size")
    p.add_argument("--random-state", type=int, default=42, help="random_state")
    return p.parse_args()

def main():
    args = parse_args()
    
    # ✅ FILE-BASED TRACKING = NO SERVER PROBLEMS EVER
    mlflow_tracking_dir = "./mlruns"
    os.makedirs(mlflow_tracking_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{os.path.abspath(mlflow_tracking_dir)}")
    mlflow.set_experiment(args.experiment)
    
    print(f"✅ Using local MLflow tracking: {mlflow_tracking_dir}")
    print(f"✅ View results: mlflow ui (opens http://localhost:5000)")

    # Load data
    if not os.path.exists(args.csv):
        raise SystemExit(f"❌ CSV not found: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"✅ Loaded {len(df)} rows")

    if args.target not in df.columns:
        raise SystemExit(f"❌ Target '{args.target}' not found. Columns: {list(df.columns)}")

    # Prepare data
    X = df.drop(columns=[args.target])
    y = df[args.target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    print(f"✅ Train: {len(X_train)} | Test: {len(X_test)}")

    # Train and log
    with mlflow.start_run(run_name=args.run):
        print("🚀 Training started...")
        
        # Log params
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))

        # Train model
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
        )
        model.fit(X_train, y_train)

        # Predict + metrics
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        rmse = float(math.sqrt(mse))
        r2 = float(r2_score(y_test, preds))

        # Log metrics
        mlflow.log_metric("mse", float(mse))
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"✅ SUCCESS!")
        print(f"   MSE: {mse:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
        print(f"📊 View results: cd to project dir && mlflow ui")

if __name__ == "__main__":
    main()

