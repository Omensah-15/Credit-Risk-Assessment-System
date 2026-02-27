import os
import json
import hashlib
import sqlite3
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Credit Risk Assessment",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Configuration & Paths --------------------
DATA_DIR = "data"
MODELS_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "assessment_results.db")

# Model files
MODEL_FILE = os.path.join(MODELS_DIR, "model.pkl")
SCALER_FILE = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURES_FILE = os.path.join(MODELS_DIR, "features.pkl")
CREDIT_MODEL_FILE = os.path.join(MODELS_DIR, "credit_model.pkl")
ROBUST_SCALER_FILE = os.path.join(MODELS_DIR, "robust_scaler.pkl")

# -------------------- Database Initialization --------------------
def init_db():
    """Initialize SQLite database for assessment results."""
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

# -------------------- Utility: Deterministic Hashing --------------------
def generate_data_hash(data: Dict[str, Any]) -> str:
    """
    Generate deterministic SHA-256 hash of applicant data.
    """
    data_copy = dict(data)
    data_copy.pop("submission_timestamp", None)
    canonical = json.dumps(data_copy, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def verify_data_hash(data: Dict[str, Any], original_hash: str) -> bool:
    """Verify data integrity by comparing hashes."""
    return generate_data_hash(data) == original_hash

# -------------------- Simple Model Class (No XGBoost Dependency) --------------------
class SimpleModelWrapper:
    """
    A simple wrapper that loads model files without requiring xgboost import.
    """
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_loaded = False
    
    def load(self):
        """Load model files without triggering xgboost import."""
        try:
            # First try to load credit_model.pkl (might contain all artifacts)
            if os.path.exists(CREDIT_MODEL_FILE):
                try:
                    data = joblib.load(CREDIT_MODEL_FILE)
                    if isinstance(data, dict):
                        self.model = data.get('model')
                        self.scaler = data.get('scaler')
                        self.feature_names = data.get('feature_names', [])
                        if self.model is not None:
                            self.is_loaded = True
                            return True
                except:
                    pass
            
            # Try individual files
            if os.path.exists(MODEL_FILE):
                self.model = joblib.load(MODEL_FILE)
            
            # Try scaler (prefer robust_scaler.pkl if it exists)
            if os.path.exists(ROBUST_SCALER_FILE):
                self.scaler = joblib.load(ROBUST_SCALER_FILE)
            elif os.path.exists(SCALER_FILE):
                self.scaler = joblib.load(SCALER_FILE)
            
            # Load feature names
            if os.path.exists(FEATURES_FILE):
                self.feature_names = joblib.load(FEATURES_FILE)
            
            if self.model is not None:
                self.is_loaded = True
                return True
            else:
                return False
                
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def predict_proba(self, X):
        """
        Make prediction without requiring xgboost methods.
        This handles different model types.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # If model has predict_proba, use it
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        
        # If model only has predict, convert to probability
        elif hasattr(self.model, 'predict'):
            pred = self.model.predict(X)
            # Convert binary prediction to probability-like format
            proba = np.zeros((len(pred), 2))
            for i, p in enumerate(pred):
                proba[i, int(p)] = 0.9
                proba[i, 1-int(p)] = 0.1
            return proba
        
        else:
            raise RuntimeError("Model doesn't have predict_proba or predict method")

# Global model instance
@st.cache_resource
def get_model():
    """Get or create the model wrapper instance."""
    model_wrapper = SimpleModelWrapper()
    success = model_wrapper.load()
    if success:
        return model_wrapper
    return None

# -------------------- Preprocessing --------------------
EMPLOYMENT_MAPPING = {
    'employed': 0, 'self-employed': 1, 'unemployed': 2, 'student': 3, 'retired': 4
}
EDUCATION_MAPPING = {
    'High School': 0, 'Diploma': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4
}
LOAN_PURPOSE_MAPPING = {
    'Business': 0, 'Crypto-Backed': 1, 'Car Loan': 2, 'Education': 3, 'Home Loan': 4
}
COLLATERAL_MAPPING = {'Yes': 1, 'No': 0}

def preprocess_inference_data(input_data, model_wrapper):
    """
    Preprocess input data for model inference.
    Creates all features the model expects.
    """
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()

    # Basic features
    df['age'] = df['age'].astype(float)
    df['annual_income'] = df['annual_income'].astype(float)
    df['loan_amount'] = df['loan_amount'].astype(float)
    df['credit_history_length'] = df['credit_history_length'].astype(float)
    df['num_previous_loans'] = df['num_previous_loans'].astype(float)
    df['num_defaults'] = df['num_defaults'].astype(float)
    df['avg_payment_delay_days'] = df['avg_payment_delay_days'].astype(float)
    df['current_credit_score'] = df['current_credit_score'].astype(float)
    df['loan_term_months'] = df['loan_term_months'].astype(float)

    # Categorical mappings
    df['employment_status_encoded'] = df['employment_status'].map(EMPLOYMENT_MAPPING).fillna(0)
    df['education_level_encoded'] = df['education_level'].map(EDUCATION_MAPPING).fillna(0)
    df['loan_purpose_encoded'] = df['loan_purpose'].map(LOAN_PURPOSE_MAPPING).fillna(0)
    df['collateral_present_encoded'] = df['collateral_present'].map(COLLATERAL_MAPPING).fillna(0)

    # Engineered features
    df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
    df['payment_per_month'] = df['loan_amount'] / df['loan_term_months']
    df['payment_to_income'] = df['payment_per_month'] / (df['annual_income'] / 12 + 0.001)
    df['default_rate'] = df['num_defaults'] / (df['num_previous_loans'] + 1)
    df['credit_utilization'] = (df['num_previous_loans'] * df['loan_amount']) / (df['annual_income'] + 1)
    df['payment_reliability'] = 1 / (df['avg_payment_delay_days'] + 1)
    df['credit_history_x_score'] = df['credit_history_length'] * df['current_credit_score']
    df['default_x_delay'] = df['num_defaults'] * df['avg_payment_delay_days']
    df['age_x_income'] = df['age'] * df['annual_income'] / 100000
    
    # Polynomial features
    for col in ['current_credit_score', 'annual_income', 'credit_history_length', 'age']:
        df[f'{col}_squared'] = df[col] ** 2
        df[f'{col}_log'] = np.log1p(df[col])
    
    # Dummy variables
    employment_dummies = pd.get_dummies(df['employment_status'], prefix='employment_status')
    education_dummies = pd.get_dummies(df['education_level'], prefix='education_level')
    purpose_dummies = pd.get_dummies(df['loan_purpose'], prefix='loan_purpose')
    collateral_dummies = pd.get_dummies(df['collateral_present'], prefix='collateral_present')
    
    # Combine all features
    df = pd.concat([
        df, 
        employment_dummies, 
        education_dummies, 
        purpose_dummies, 
        collateral_dummies
    ], axis=1)
    
    # If we have feature names from the model, use only those
    if model_wrapper.feature_names and len(model_wrapper.feature_names) > 0:
        for col in model_wrapper.feature_names:
            if col not in df.columns:
                df[col] = 0
        X = df[model_wrapper.feature_names].copy()
    else:
        # Otherwise use all numeric columns
        X = df.select_dtypes(include=[np.number]).copy()
    
    # Fill any missing values
    X = X.fillna(0)
    
    # Apply scaler if available
    if model_wrapper.scaler is not None:
        try:
            X_scaled = model_wrapper.scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns)
        except:
            pass  # Use unscaled if scaling fails
    
    return X

# -------------------- Prediction Function --------------------
def predict_single(input_dict: dict) -> Tuple[float, int, str]:
    """
    Predict for a single applicant.
    Returns: (probability, risk_score, category)
    """
    model_wrapper = get_model()
    
    if model_wrapper is None:
        raise RuntimeError("Model not loaded. Please check the models directory.")
    
    # Preprocess the data
    X = preprocess_inference_data(input_dict, model_wrapper)
    
    # Make prediction
    try:
        proba_array = model_wrapper.predict_proba(X)
        # Handle different output shapes
        if len(proba_array.shape) == 2 and proba_array.shape[1] >= 2:
            proba = float(proba_array[0, 1])
        else:
            proba = float(proba_array[0])
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")
    
    # Calculate risk score (higher = safer)
    risk_score = int(round((1 - proba) * 1000))
    risk_score = max(0, min(1000, risk_score))
    
    # Determine risk category
    if proba < 0.1:
        category = "Very Low Risk"
    elif proba < 0.2:
        category = "Low Risk"
    elif proba < 0.4:
        category = "Medium Risk"
    elif proba < 0.6:
        category = "High Risk"
    else:
        category = "Very High Risk"
    
    return proba, risk_score, category

# -------------------- Session State --------------------
if 'assessment_results' not in st.session_state:
    st.session_state.assessment_results = {}

# -------------------- Main UI --------------------
st.title("Credit Risk Assessment System")
st.markdown("---")

# Check model status
model_wrapper = get_model()
if model_wrapper is not None:
    st.sidebar.success("✓ Model Loaded Successfully")
    if model_wrapper.feature_names:
        st.sidebar.info(f"Features: {len(model_wrapper.feature_names)}")
else:
    st.sidebar.error("✗ Model Not Loaded")
    # Show available files
    if os.path.exists(MODELS_DIR):
        files = os.listdir(MODELS_DIR)
        st.sidebar.write("Files in models directory:")
        for f in files:
            st.sidebar.write(f"  • {f}")

# Sidebar navigation
menu = st.sidebar.selectbox(
    "Navigation",
    ["New Assessment", "Assessment History"]
)

# -------------------- New Assessment --------------------
if menu == "New Assessment":
    st.header("New Credit Risk Assessment")
    
    if model_wrapper is None:
        st.error("Model not loaded. Please check the models directory.")
        st.stop()
    
    with st.form("assessment_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal Information")
            applicant_id = st.text_input("Applicant ID *")
            applicant_name = st.text_input("Full Name *")
            applicant_email = st.text_input("Email Address *")
            age = st.slider("Age", 18, 100, 30)
            employment_status = st.selectbox(
                "Employment Status", 
                ["employed", "self-employed", "unemployed", "student", "retired"]
            )
            education_level = st.selectbox(
                "Education Level", 
                ["High School", "Diploma", "Bachelor", "Master", "PhD"]
            )
        
        with col2:
            st.subheader("Financial Information (GHS)")
            annual_income = st.number_input(
                "Annual Income (GHS) *", 
                min_value=0, 
                value=50000, 
                step=1000, 
                format="%d"
            )
            loan_amount = st.number_input(
                "Loan Amount (GHS) *", 
                min_value=0, 
                value=25000, 
                step=1000, 
                format="%d"
            )
            loan_purpose = st.selectbox(
                "Loan Purpose", 
                ["Business", "Crypto-Backed", "Car Loan", "Education", "Home Loan"]
            )
            loan_term_months = st.slider("Loan Term (months)", 12, 84, 36)
            collateral_present = st.radio(
                "Collateral Present", 
                ["Yes", "No"], 
                horizontal=True
            )
        
        st.markdown("---")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Credit History")
            credit_history_length = st.slider("Credit History (years)", 0, 30, 5)
            num_previous_loans = st.slider("Number of Previous Loans", 0, 20, 2)
            num_defaults = st.slider("Number of Defaults", 0, 10, 0)
            current_credit_score = st.slider("Current Credit Score", 300, 850, 650)
        
        with col4:
            st.subheader("Payment Behavior")
            avg_payment_delay_days = st.slider("Average Payment Delay (days)", 0, 60, 5)

        submitted = st.form_submit_button(
            "Run Assessment", 
            type="primary", 
            use_container_width=True
        )

    if submitted:
        # Validate required fields
        missing_fields = []
        if not applicant_id:
            missing_fields.append("Applicant ID")
        if not applicant_name:
            missing_fields.append("Full Name")
        if not applicant_email:
            missing_fields.append("Email Address")
        
        if missing_fields:
            st.error(f"Please provide: {', '.join(missing_fields)}")
        else:
            with st.spinner("Processing assessment..."):
                application = {
                    "applicant_id": applicant_id,
                    "applicant_name": applicant_name,
                    "applicant_email": applicant_email,
                    "age": int(age),
                    "annual_income": float(annual_income),
                    "employment_status": employment_status,
                    "education_level": education_level,
                    "credit_history_length": int(credit_history_length),
                    "num_previous_loans": int(num_previous_loans),
                    "num_defaults": int(num_defaults),
                    "avg_payment_delay_days": int(avg_payment_delay_days),
                    "current_credit_score": int(current_credit_score),
                    "loan_amount": float(loan_amount),
                    "loan_term_months": int(loan_term_months),
                    "loan_purpose": loan_purpose,
                    "collateral_present": collateral_present,
                    "submission_timestamp": datetime.utcnow().isoformat()
                }

                # Generate data hash
                data_hash = generate_data_hash(application)
                
                # Run prediction
                try:
                    proba, score, category = predict_single(application)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()

                # Store in database
                try:
                    conn = sqlite3.connect(DB_PATH)
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT OR REPLACE INTO assessment_results
                        (applicant_id, applicant_name, applicant_email, age, data_hash, 
                         risk_score, probability_of_default, risk_category, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        application['applicant_id'], 
                        application['applicant_name'], 
                        application['applicant_email'], 
                        application['age'],
                        data_hash, 
                        score, 
                        proba, 
                        category,
                        application['submission_timestamp']
                    ))
                    conn.commit()
                finally:
                    conn.close()

                # Store in session state
                st.session_state.assessment_results[applicant_id] = {
                    "data": application,
                    "hash": data_hash,
                    "probability_of_default": proba,
                    "risk_score": score,
                    "risk_category": category,
                    "timestamp": datetime.utcnow().isoformat()
                }

                # Display results
                st.markdown("### Assessment Results")
                
                with st.container():
                    col_result1, col_result2, col_result3 = st.columns(3)
                    
                    # Risk Score
                    score_color = "green" if score > 700 else "orange" if score > 400 else "red"
                    col_result1.markdown(
                        f"""
                        <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6; text-align: center;">
                            <p style="color: #666; margin-bottom: 5px;">Risk Score</p>
                            <p style="font-size: 36px; font-weight: bold; color: {score_color};">{score}</p>
                            <p style="color: #666;">/1000</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    # Default Probability
                    proba_color = "green" if proba < 0.2 else "orange" if proba < 0.4 else "red"
                    col_result2.markdown(
                        f"""
                        <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6; text-align: center;">
                            <p style="color: #666; margin-bottom: 5px;">Default Probability</p>
                            <p style="font-size: 36px; font-weight: bold; color: {proba_color};">{proba:.1%}</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    # Risk Category
                    category_colors = {
                        "Very Low Risk": "#28a745",
                        "Low Risk": "#5cb85c",
                        "Medium Risk": "#f0ad4e",
                        "High Risk": "#d9534f",
                        "Very High Risk": "#c9302c"
                    }
                    cat_color = category_colors.get(category, "#666")
                    col_result3.markdown(
                        f"""
                        <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6; text-align: center;">
                            <p style="color: #666; margin-bottom: 5px;">Risk Category</p>
                            <p style="font-size: 28px; font-weight: bold; color: {cat_color};">{category}</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

                # Financial Summary
                st.markdown("### Financial Summary")
                col_fin1, col_fin2 = st.columns(2)
                
                with col_fin1:
                    st.markdown(
                        f"""
                        <div style="padding: 15px; border-radius: 8px; background-color: #f0f2f6;">
                            <p style="color: #666; margin-bottom: 5px;">Annual Income</p>
                            <p style="font-size: 24px; font-weight: bold;">GHS {annual_income:,.0f}</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                with col_fin2:
                    st.markdown(
                        f"""
                        <div style="padding: 15px; border-radius: 8px; background-color: #f0f2f6;">
                            <p style="color: #666; margin-bottom: 5px;">Loan Amount</p>
                            <p style="font-size: 24px; font-weight: bold;">GHS {loan_amount:,.0f}</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

                # Data Integrity
                st.markdown("### Data Integrity")
                st.code(data_hash, language="text")
                
                col_verify, _ = st.columns([1, 3])
                with col_verify:
                    if st.button("Verify Data Integrity", use_container_width=True):
                        if verify_data_hash(application, data_hash):
                            st.success("Data integrity verified - Hash matches")
                        else:
                            st.error("Data integrity check failed - Hash mismatch")

# -------------------- Assessment History --------------------
elif menu == "Assessment History":
    st.header("Assessment History")
    
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM assessment_results ORDER BY timestamp DESC", conn)
    finally:
        conn.close()

    if df.empty:
        st.info("No assessment records found. Create a new assessment to get started.")
    else:
        # Summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Assessments", len(df))
        
        with col2:
            avg_score = df['risk_score'].mean()
            st.metric("Average Risk Score", f"{avg_score:.0f}")
        
        with col3:
            high_risk_count = len(df[df['risk_category'].str.contains('High', na=False)])
            st.metric("High Risk Cases", high_risk_count)

        # Assessment records table
        st.subheader("Assessment Records")
        
        display_df = df.copy()
        display_df['probability_of_default'] = display_df['probability_of_default'].apply(
            lambda x: f"{float(x):.1%}" if pd.notnull(x) else "N/A"
        )
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')

        st.dataframe(
            display_df[[
                'applicant_id', 'applicant_name', 'risk_score', 'risk_category',
                'probability_of_default', 'timestamp'
            ]],
            use_container_width=True,
            column_config={
                "applicant_id": "Applicant ID",
                "applicant_name": "Name",
                "risk_score": "Risk Score",
                "risk_category": "Risk Category",
                "probability_of_default": "Default Prob.",
                "timestamp": "Assessment Date"
            }
        )

        # Export button
        csv = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV Export",
            csv,
            "assessment_history.csv",
            "text/csv",
            use_container_width=True
        )
