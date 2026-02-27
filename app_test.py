import os
import json
import hashlib
import sqlite3
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional
import joblib
import pandas as pd
import streamlit as st
import requests

# -------------------- Config --------------------
DATA_DIR = "data"
MODELS_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "assessment_results.db")
FEATURE_COLUMNS_FILE = os.path.join(MODELS_DIR, "feature_columns.pkl")
CALIBRATED_MODEL_FILE = os.path.join(MODELS_DIR, "calibration_model.pkl")

# Replace these with your GitHub raw URLs for the .pkl files
GITHUB_BASE = "https://github.com/Omensah-15/Credit-Risk-Assessment-System/raw/main/models"
MODEL_FILES = {
    "calibration_model.pkl": f"{GITHUB_BASE}/calibration_model.pkl",
    "feature_columns.pkl": f"{GITHUB_BASE}/feature_columns.pkl"
}

# -------------------- Helper: Download Model --------------------
def download_model_if_missing():
    for fname, url in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path) or os.path.getsize(path) < 1024:
            st.info(f"Downloading {fname} from GitHub...")
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                with open(path, 'wb') as f:
                    f.write(r.content)
                st.success(f"{fname} downloaded successfully!")
            else:
                st.error(f"Failed to download {fname}. HTTP {r.status_code}")

# -------------------- Database --------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS assessment_results (
            applicant_id TEXT PRIMARY KEY,
            applicant_name TEXT,
            applicant_email TEXT,
            age INTEGER,
            data_hash TEXT,
            risk_score INTEGER,
            probability_of_default REAL,
            risk_category TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# -------------------- Hashing --------------------
def generate_data_hash(data: Dict[str, Any]) -> str:
    data_copy = dict(data)
    data_copy.pop("submission_timestamp", None)
    canonical = json.dumps(data_copy, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def verify_data_hash(data: Dict[str, Any], original_hash: str) -> bool:
    return generate_data_hash(data) == original_hash

# -------------------- Preprocessing --------------------
EMPLOYMENT_MAPPING = {'employed': 0, 'self-employed': 1, 'unemployed': 2, 'student': 3}
EDUCATION_MAPPING = {'High School': 0, 'Diploma': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
LOAN_PURPOSE_MAPPING = {'Business': 0, 'Personal': 1, 'Car Loan': 2, 'Education': 3, 'Home Loan': 4}

def preprocess_inference_data(input_data) -> pd.DataFrame:
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()

    df['employment_status'] = df.get('employment_status', pd.Series()).map(EMPLOYMENT_MAPPING).fillna(0).astype(int)
    df['education_level'] = df.get('education_level', pd.Series()).map(EDUCATION_MAPPING).fillna(0).astype(int)
    df['loan_purpose'] = df.get('loan_purpose', pd.Series()).map(LOAN_PURPOSE_MAPPING).fillna(0).astype(int)
    df['collateral_present'] = df.get('collateral_present', pd.Series()).map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    # Engineered features
    df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
    df['credit_utilization'] = df['num_previous_loans'] / (df['credit_history_length'] + 1)
    df['default_rate'] = df['num_defaults'] / (df['num_previous_loans'] + 1)

    # Load feature columns
    if not os.path.exists(FEATURE_COLUMNS_FILE):
        raise FileNotFoundError("Feature columns file missing. Download or retrain first.")
    feature_columns: List[str] = joblib.load(FEATURE_COLUMNS_FILE)

    # Ensure all columns exist
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    return df[feature_columns]

# -------------------- Load Model --------------------
@st.cache_resource
def load_models() -> Tuple[Optional[object], Optional[List[str]]]:
    download_model_if_missing()
    try:
        model = joblib.load(CALIBRATED_MODEL_FILE)
        feature_columns = joblib.load(FEATURE_COLUMNS_FILE)
        return model, feature_columns
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# -------------------- Prediction --------------------
def predict_single(input_dict: dict) -> Tuple[float, int, str, pd.DataFrame]:
    model, feature_columns = load_models()
    if model is None:
        raise RuntimeError("Model not loaded.")
    
    processed = preprocess_inference_data(input_dict)
    proba = float(model.predict_proba(processed)[:,1][0])
    risk_score = int(round((1 - proba) * 1000))
    
    if proba < 0.1: category = "Very Low Risk"
    elif proba < 0.2: category = "Low Risk"
    elif proba < 0.4: category = "Medium Risk"
    elif proba < 0.6: category = "High Risk"
    else: category = "Very High Risk"
    
    return proba, risk_score, category, processed
