from __future__ import annotations
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def parse_args():
    p = argparse.ArgumentParser("Wine Prediction Training")
    p.add_argument("--csv", default="data/wine_sample.csv", help="Path to CSV")
    p.add_argument("--target", default="quality", help="Target column")
    p.add_argument("--n-estimators", type=int, default=50, help="RF n_estimators")
    p.add_argument("--max-depth", type=int, default=5, help="RF max_depth")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split")
    p.add_argument("--random-state", type=int, default=42, help="Random seed")
    return p.parse_args()

def main():
    args = parse_args()
    
    print("🚀 Starting Wine Quality Prediction Training")
    print("⏭️  MLflow SKIPPED (Kubernetes auth issue)")
    
    # Load & clean data
    if not os.path.exists(args.csv):
        raise SystemExit(f"❌ CSV not found: {args.csv}")
    
    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip()  # Fix whitespace
    
    print(f"✅ Columns: {list(df.columns)}")
    print(f"📊 Shape: {df.shape}")
    
    if args.target not in df.columns:
        raise SystemExit(f"❌ Target '{args.target}' not found")
    
    # Prepare data
    X = df.drop(columns=[args.target])
    y = df[args.target]
    
    print(f"📈 Features: {X.shape[1]}, Samples: {X.shape[0]}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    
    print(f"✅ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # Train model
    print("🤖 Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
    )
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n🎯 RESULTS:")
    print(f"   RMSE: {rmse:.3f}")
    print(f"   R²:   {r2:.3f}")
    print(f"   Params: {args.n_estimators} trees, depth {args.max_depth}")
    
    # Save model
    model_path = "wine_model.joblib"
    joblib.dump(rf, model_path)
    print(f"\n💾 Model saved: {model_path}")
    
    # Test prediction
    sample_pred = rf.predict(X_test.iloc[[0]])[0]
    print(f"🧪 Sample prediction: {sample_pred:.1f}")
    
    print("\n✅ TRAINING COMPLETE!")
    print("🚀 Ready for /predict endpoint!")
    print("📡 Test: curl -X POST http://127.0.0.1:5001/predict -d '{\"features\":[7.4,0.7,0.0,1.9,0.076]}'")

if __name__ == "__main__":
    main()

