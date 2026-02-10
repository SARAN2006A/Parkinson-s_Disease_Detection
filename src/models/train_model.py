import pandas as pd
import numpy as np
import os
import joblib
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(data_path, model_path):
    # Load Data
    logging.info("Loading dataset...")
    df = pd.read_csv(os.path.join(data_path, 'final_dataset.csv'))
    
    # Preprocessing
    # Target: UPDRS_Total (Primary) and UDysRS_Total (Secondary)
    # Let's train for UPDRS_Total first as generic Parkinson's severity
    
    target_col = 'UPDRS_Total'
    
    # Drop rows where target is missing
    df = df.dropna(subset=[target_col])
    
    # Select Features (Exclude metadata like Trial_ID, Task, Scores)
    # Features are numeric columns that are not targets
    feature_cols = [c for c in df.columns if c not in ['Trial_ID', 'Task', 'UPDRS_Total', 'UDysRS_Total']]
    
    X = df[feature_cols]
    y = df[target_col]
    
    logging.info(f"Training Data Shape: {X.shape}")
    
    # Impute missing features (some angles might be NaN if joints missing)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    logging.info("Training Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    logging.info(f"Model Performance (UPDRS):")
    logging.info(f"MAE: {mae:.4f}")
    logging.info(f"RMSE: {rmse:.4f}")
    logging.info(f"R2 Score: {r2:.4f}")
    
    # Save Model and Imputer
    os.makedirs(model_path, exist_ok=True)
    joblib.dump(rf, os.path.join(model_path, 'rf_model_updrs.pkl'))
    joblib.dump(imputer, os.path.join(model_path, 'imputer.pkl'))
    joblib.dump(feature_cols, os.path.join(model_path, 'feature_names.pkl'))
    
    logging.info("Model saved successfully.")

if __name__ == '__main__':
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    data_path = os.path.join(project_dir, 'data', 'processed')
    model_path = os.path.join(project_dir, 'models')
    
    train_model(data_path, model_path)
