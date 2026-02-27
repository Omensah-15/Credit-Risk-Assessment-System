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

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Credit Risk Assessment System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Configuration & Paths --------------------
DATA_DIR = "data"
MODELS_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "assessments.db")

# Your specific model files
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")  # The calibrated stacking model
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")  # The robust scaler
FEATURES_PATH = os.path.join(MODELS_DIR, "features.pkl")  # The feature names
CREDIT_MODEL_PATH = os.path.join(MODELS_DIR, "credit_model.pkl")  # Full artifacts

# -------------------- Database Initialization --------------------
def init_db():
    """Initialize SQLite database for assessment results."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            applicant_id TEXT NOT NULL,
            applicant_name TEXT NOT NULL,
            applicant_email TEXT NOT NULL,
            assessment_date TEXT NOT NULL,
            risk_score INTEGER NOT NULL,
            default_probability REAL NOT NULL,
            risk_category TEXT NOT NULL,
            data_hash TEXT NOT NULL,
            annual_income REAL,
            loan_amount REAL,
            employment_status TEXT,
            credit_score INTEGER,
            age INTEGER,
            loan_term INTEGER,
            num_previous_loans INTEGER,
            num_defaults INTEGER,
            payment_delay INTEGER
        )
    """)
    conn.commit()
    conn.close()

init_db()

# -------------------- Utility Functions --------------------
def generate_data_hash(data: Dict[str, Any]) -> str:
    """Generate deterministic SHA-256 hash of applicant data."""
    data_copy = dict(data)
    # Remove timestamp fields that would change
    data_copy.pop("assessment_date", None)
    data_copy.pop("submission_timestamp", None)
    canonical = json.dumps(data_copy, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def verify_data_hash(data: Dict[str, Any], original_hash: str) -> bool:
    """Verify data integrity by comparing hashes."""
    return generate_data_hash(data) == original_hash

# -------------------- Model Loading --------------------
@st.cache_resource
def load_model_artifacts():
    """Load model, scaler, and feature names from your specific files."""
    try:
        # Try loading the full artifacts first (credit_model.pkl)
        if os.path.exists(CREDIT_MODEL_PATH):
            artifacts = joblib.load(CREDIT_MODEL_PATH)
            if isinstance(artifacts, dict):
                model = artifacts.get('model')
                scaler = artifacts.get('scaler')
                feature_names = artifacts.get('feature_names', [])
                optimal_threshold = artifacts.get('optimal_threshold', 0.5)
                
                if model is not None:
                    print(f"Loaded model from {CREDIT_MODEL_PATH}")
                    return model, scaler, feature_names, optimal_threshold
        
        # Fall back to individual files
        model = None
        scaler = None
        feature_names = []
        optimal_threshold = 0.5
        
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"Loaded model from {MODEL_PATH}")
        
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print(f"Loaded scaler from {SCALER_PATH}")
        
        if os.path.exists(FEATURES_PATH):
            feature_names = joblib.load(FEATURES_PATH)
            print(f"Loaded {len(feature_names)} features from {FEATURES_PATH}")
        
        if model is None:
            st.error("No model file found. Please train the model first.")
            return None, None, None, 0.5
        
        return model, scaler, feature_names, optimal_threshold
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, 0.5

# -------------------- Data Preprocessing --------------------
def preprocess_input_data(input_data: Dict[str, Any], feature_names: List[str], scaler=None) -> pd.DataFrame:
    """
    Preprocess input data for model inference.
    Must match the preprocessing from your training notebook.
    """
    # Convert to DataFrame
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()
    
    # Create engineered features to match training
    df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
    df['payment_per_month'] = df['loan_amount'] / df['loan_term_months']
    df['payment_to_income'] = df['payment_per_month'] / (df['annual_income'] / 12 + 0.001)
    
    # Risk indicators
    df['default_rate'] = df['num_defaults'] / (df['num_previous_loans'] + 1)
    df['credit_utilization'] = (df['num_previous_loans'] * df['loan_amount']) / (df['annual_income'] + 1)
    df['payment_reliability'] = 1 / (df['avg_payment_delay_days'] + 1)
    
    # Interaction features
    df['credit_history_x_score'] = df['credit_history_length'] * df['current_credit_score']
    df['default_x_delay'] = df['num_defaults'] * df['avg_payment_delay_days']
    df['age_x_income'] = df['age'] * df['annual_income'] / 100000
    
    # Polynomial features
    for col in ['current_credit_score', 'annual_income', 'credit_history_length', 'age']:
        if col in df.columns:
            df[f'{col}_squared'] = df[col] ** 2
            df[f'{col}_log'] = np.log1p(df[col])
    
    # Categorical encodings
    employment_map = {'Employed': 0, 'Self-Employed': 1, 'Unemployed': 2, 'Retired': 3, 'Student': 4}
    education_map = {'High School': 0, 'Diploma': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
    purpose_map = {'Business': 0, 'Crypto-Backed': 1, 'Car Loan': 2, 'Education': 3, 'Home Loan': 4,
                   'Debt Consolidation': 5, 'Major Purchase': 6, 'Medical': 7, 'Vacation': 8}
    collateral_map = {'Yes': 1, 'No': 0}
    
    df['employment_status_encoded'] = df['employment_status'].map(employment_map).fillna(0)
    df['education_level_encoded'] = df['education_level'].map(education_map).fillna(0)
    df['loan_purpose_encoded'] = df['loan_purpose'].map(purpose_map).fillna(0)
    df['collateral_present_encoded'] = df['collateral_present'].map(collateral_map).fillna(0)
    
    # Create dummy variables for categorical columns (if your model expects them)
    if 'employment_status_Employed' in feature_names:
        df['employment_status_Employed'] = (df['employment_status'] == 'Employed').astype(int)
        df['employment_status_Self-Employed'] = (df['employment_status'] == 'Self-Employed').astype(int)
        df['employment_status_Unemployed'] = (df['employment_status'] == 'Unemployed').astype(int)
        df['employment_status_Retired'] = (df['employment_status'] == 'Retired').astype(int)
        df['employment_status_Student'] = (df['employment_status'] == 'Student').astype(int)
    
    if 'education_level_Bachelor' in feature_names:
        df['education_level_High School'] = (df['education_level'] == 'High School').astype(int)
        df['education_level_Diploma'] = (df['education_level'] == 'Diploma').astype(int)
        df['education_level_Bachelor'] = (df['education_level'] == 'Bachelor').astype(int)
        df['education_level_Master'] = (df['education_level'] == 'Master').astype(int)
        df['education_level_PhD'] = (df['education_level'] == 'PhD').astype(int)
    
    if 'loan_purpose_Business' in feature_names:
        df['loan_purpose_Business'] = (df['loan_purpose'] == 'Business').astype(int)
        df['loan_purpose_Crypto-Backed'] = (df['loan_purpose'] == 'Crypto-Backed').astype(int)
        df['loan_purpose_Car Loan'] = (df['loan_purpose'] == 'Car Loan').astype(int)
        df['loan_purpose_Education'] = (df['loan_purpose'] == 'Education').astype(int)
        df['loan_purpose_Home Loan'] = (df['loan_purpose'] == 'Home Loan').astype(int)
    
    if 'collateral_present_Yes' in feature_names:
        df['collateral_present_Yes'] = (df['collateral_present'] == 'Yes').astype(int)
        df['collateral_present_No'] = (df['collateral_present'] == 'No').astype(int)
    
    # Ensure all feature_names are present
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    # Select only the features the model expects
    X = df[feature_names].copy()
    
    # Handle any missing values
    X = X.fillna(0)
    
    # Scale if scaler is provided
    if scaler is not None:
        try:
            X_scaled = scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=feature_names)
        except Exception as e:
            print(f"Scaling failed: {e}, using unscaled data")
    
    return X

# -------------------- Risk Assessment --------------------
def assess_risk(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform risk assessment on input data.
    """
    # Load model artifacts
    model, scaler, feature_names, threshold = load_model_artifacts()
    
    if model is None:
        raise RuntimeError("Model not loaded. Please train the model first.")
    
    if not feature_names:
        # If feature names not available, use all numeric columns from input
        print("Feature names not found, using default processing")
        feature_names = list(input_data.keys())
    
    # Preprocess data
    X = preprocess_input_data(input_data, feature_names, scaler)
    
    # Make prediction
    try:
        # Try predict_proba first
        probability = float(model.predict_proba(X)[0, 1])
    except AttributeError:
        # If no predict_proba, use decision_function or predict
        try:
            probability = float(model.decision_function(X)[0])
            # Normalize to 0-1 if needed
            probability = 1 / (1 + np.exp(-probability))
        except:
            probability = float(model.predict(X)[0])
    except Exception as e:
        # Final fallback
        try:
            probability = float(model.predict(X)[0])
        except:
            raise RuntimeError(f"Model prediction failed: {e}")
    
    # Calculate risk score (0-1000, higher is safer)
    risk_score = int(round((1 - probability) * 1000))
    risk_score = max(0, min(1000, risk_score))  # Clamp to 0-1000
    
    # Determine risk category
    if probability < 0.1:
        category = "Very Low Risk"
    elif probability < 0.2:
        category = "Low Risk"
    elif probability < 0.4:
        category = "Medium Risk"
    elif probability < 0.6:
        category = "High Risk"
    else:
        category = "Very High Risk"
    
    return {
        "default_probability": probability,
        "risk_score": risk_score,
        "risk_category": category,
        "threshold_used": threshold
    }

# -------------------- Save Assessment --------------------
def save_assessment(applicant_data: Dict[str, Any], assessment_result: Dict[str, Any], data_hash: str):
    """Save assessment results to database."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    cur.execute("""
        INSERT INTO assessments
        (applicant_id, applicant_name, applicant_email, assessment_date,
         risk_score, default_probability, risk_category, data_hash,
         annual_income, loan_amount, employment_status, credit_score,
         age, loan_term, num_previous_loans, num_defaults, payment_delay)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        applicant_data.get('applicant_id', ''),
        applicant_data.get('applicant_name', ''),
        applicant_data.get('applicant_email', ''),
        datetime.now().isoformat(),
        assessment_result['risk_score'],
        assessment_result['default_probability'],
        assessment_result['risk_category'],
        data_hash,
        applicant_data.get('annual_income', 0),
        applicant_data.get('loan_amount', 0),
        applicant_data.get('employment_status', ''),
        applicant_data.get('current_credit_score', 0),
        applicant_data.get('age', 0),
        applicant_data.get('loan_term_months', 0),
        applicant_data.get('num_previous_loans', 0),
        applicant_data.get('num_defaults', 0),
        applicant_data.get('avg_payment_delay_days', 0)
    ))
    
    conn.commit()
    conn.close()

# -------------------- Load Assessment History --------------------
@st.cache_data(ttl=60)
def load_assessment_history():
    """Load assessment history from database."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("""
            SELECT 
                applicant_id,
                applicant_name,
                applicant_email,
                assessment_date,
                risk_score,
                default_probability,
                risk_category,
                annual_income,
                loan_amount,
                credit_score,
                age,
                employment_status,
                data_hash
            FROM assessments 
            ORDER BY assessment_date DESC
        """, conn)
    finally:
        conn.close()
    return df

# -------------------- Delete Assessment --------------------
def delete_assessment(applicant_id: str, assessment_date: str):
    """Delete a specific assessment from database."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        DELETE FROM assessments 
        WHERE applicant_id = ? AND assessment_date = ?
    """, (applicant_id, assessment_date))
    conn.commit()
    conn.close()
    st.cache_data.clear()

# -------------------- Clear All History --------------------
def clear_all_history():
    """Delete all assessments from database."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM assessments")
    conn.commit()
    conn.close()
    st.cache_data.clear()

# -------------------- Session State --------------------
if 'assessment_completed' not in st.session_state:
    st.session_state.assessment_completed = False
    st.session_state.current_result = None
    st.session_state.current_data = None
    st.session_state.current_hash = None
    st.session_state.page = "New Assessment"

# -------------------- Sidebar --------------------
with st.sidebar:
    st.title("Navigation")
    
    # Simple radio buttons for navigation
    selected_page = st.radio(
        "Go to",
        ["New Assessment", "Assessment History"],
        index=0 if st.session_state.page == "New Assessment" else 1,
        key="navigation"
    )
    
    # Update session state when navigation changes
    if selected_page != st.session_state.page:
        st.session_state.page = selected_page
        st.rerun()
    
    st.markdown("---")
    
    # Model status indicator
    model, scaler, feature_names, threshold = load_model_artifacts()
    if model is not None:
        st.success(f"✓ Model loaded successfully")
        if feature_names:
            st.caption(f"Features: {len(feature_names)}")
    else:
        st.error("✗ Model not loaded")
        st.caption("Please train the model first")

# -------------------- New Assessment Page --------------------
if st.session_state.page == "New Assessment":
    st.title("New Credit Risk Assessment")
    st.markdown("Enter applicant information to assess credit risk.")
    
    # Check if model is loaded
    model, _, _, _ = load_model_artifacts()
    if model is None:
        st.warning("⚠️ Model not loaded. Please check the models directory.")
        st.info(f"Looking for model files in: {MODELS_DIR}")
        st.stop()
    
    # Create form
    with st.form("assessment_form", clear_on_submit=False):
        st.subheader("Personal Information")
        col1, col2 = st.columns(2)
        
        with col1:
            applicant_id = st.text_input("Applicant ID *", placeholder="e.g., APP001")
            applicant_name = st.text_input("Full Name *", placeholder="John Doe")
            applicant_email = st.text_input("Email Address *", placeholder="john@example.com")
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
        
        with col2:
            employment_status = st.selectbox(
                "Employment Status",
                ["Employed", "Self-Employed", "Unemployed", "Retired", "Student"]
            )
            education_level = st.selectbox(
                "Education Level",
                ["High School", "Diploma", "Bachelor", "Master", "PhD"]
            )
        
        st.markdown("---")
        st.subheader("Financial Information")
        col3, col4 = st.columns(2)
        
        with col3:
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
        
        with col4:
            loan_term_months = st.selectbox(
                "Loan Term (months)",
                [12, 24, 36, 48, 60, 72]
            )
            collateral_present = st.radio(
                "Collateral Present",
                ["Yes", "No"],
                horizontal=True
            )
        
        st.markdown("---")
        st.subheader("Credit History")
        col5, col6 = st.columns(2)
        
        with col5:
            credit_history_length = st.number_input(
                "Credit History (years)",
                min_value=0,
                max_value=50,
                value=5
            )
            num_previous_loans = st.number_input(
                "Number of Previous Loans",
                min_value=0,
                max_value=50,
                value=2
            )
            num_defaults = st.number_input(
                "Number of Defaults",
                min_value=0,
                max_value=20,
                value=0
            )
        
        with col6:
            current_credit_score = st.slider(
                "Current Credit Score",
                min_value=300,
                max_value=850,
                value=650
            )
            avg_payment_delay_days = st.number_input(
                "Average Payment Delay (days)",
                min_value=0,
                max_value=120,
                value=5
            )
        
        st.markdown("---")
        
        # Submit button
        col_submit1, col_submit2, col_submit3 = st.columns([1, 2, 1])
        with col_submit2:
            submitted = st.form_submit_button(
                "Run Risk Assessment",
                type="primary",
                use_container_width=True
            )
    
    # Process form submission
    if submitted:
        # Validate required fields
        errors = []
        if not applicant_id:
            errors.append("Applicant ID")
        if not applicant_name:
            errors.append("Full Name")
        if not applicant_email:
            errors.append("Email Address")
        if annual_income <= 0:
            errors.append("Annual Income must be greater than 0")
        if loan_amount <= 0:
            errors.append("Loan Amount must be greater than 0")
        
        if errors:
            st.error(f"Please fill in all required fields: {', '.join(errors)}")
        else:
            with st.spinner("Processing assessment..."):
                # Prepare data
                applicant_data = {
                    "applicant_id": applicant_id,
                    "applicant_name": applicant_name,
                    "applicant_email": applicant_email,
                    "age": age,
                    "employment_status": employment_status,
                    "education_level": education_level,
                    "annual_income": annual_income,
                    "loan_amount": loan_amount,
                    "loan_purpose": loan_purpose,
                    "loan_term_months": loan_term_months,
                    "collateral_present": collateral_present,
                    "credit_history_length": credit_history_length,
                    "num_previous_loans": num_previous_loans,
                    "num_defaults": num_defaults,
                    "current_credit_score": current_credit_score,
                    "avg_payment_delay_days": avg_payment_delay_days
                }
                
                # Generate data hash
                data_hash = generate_data_hash(applicant_data)
                
                try:
                    # Run assessment
                    result = assess_risk(applicant_data)
                    
                    # Save to session state
                    st.session_state.assessment_completed = True
                    st.session_state.current_result = result
                    st.session_state.current_data = applicant_data
                    st.session_state.current_hash = data_hash
                    
                    # Save to database
                    save_assessment(applicant_data, result, data_hash)
                    
                    # Force a rerun to show results
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Assessment failed: {str(e)}")
    
    # Display results if assessment completed
    if st.session_state.assessment_completed and st.session_state.current_result:
        result = st.session_state.current_result
        data = st.session_state.current_data
        data_hash = st.session_state.current_hash
        
        st.markdown("---")
        st.header("Assessment Results")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Risk Score",
                f"{result['risk_score']}/1000"
            )
        
        with col2:
            st.metric(
                "Default Probability",
                f"{result['default_probability']:.1%}"
            )
        
        with col3:
            st.metric(
                "Risk Category",
                result['risk_category']
            )
        
        with col4:
            st.metric(
                "Decision Threshold",
                f"{result['threshold_used']:.2f}"
            )
        
        # Risk gauge
        st.subheader("Risk Assessment Visualization")
        
        # Create gauge chart
        fig, ax = plt.subplots(figsize=(10, 2))
        
        # Create color gradient
        colors = ['#28a745', '#5cb85c', '#f0ad4e', '#d9534f', '#c9302c']
        
        # Horizontal bar
        risk_score = result['risk_score']
        color_idx = min(4, risk_score // 200)
        ax.barh([0], [risk_score], color=colors[color_idx])
        ax.barh([0], [1000 - risk_score], left=[risk_score], color='#e0e0e0')
        
        ax.set_xlim(0, 1000)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xlabel('Risk Score (Higher is Safer)')
        
        # Add markers
        for i, label in enumerate(['Very Low', 'Low', 'Medium', 'High', 'Very High']):
            pos = i * 200 + 100
            ax.axvline(x=pos, color='black', linestyle='--', alpha=0.3, ymin=0.4, ymax=0.6)
            ax.text(pos, -0.3, label, ha='center', va='center', fontsize=8)
        
        st.pyplot(fig)
        plt.close()
        
        # Financial Summary
        st.subheader("Financial Summary")
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            st.info(f"**Annual Income:** GHS {data['annual_income']:,.0f}")
        
        with col_f2:
            st.info(f"**Loan Amount:** GHS {data['loan_amount']:,.0f}")
        
        with col_f3:
            if data['annual_income'] > 0:
                ltv = (data['loan_amount'] / data['annual_income']) * 100
                st.info(f"**Loan-to-Income Ratio:** {ltv:.1f}%")
        
        # Data Integrity
        with st.expander("Data Integrity Information"):
            st.text("Data Hash:")
            st.code(data_hash, language="text")
            
            if st.button("Verify Data Integrity", key="verify_btn"):
                if verify_data_hash(data, data_hash):
                    st.success("✓ Data integrity verified - Hash matches")
                else:
                    st.error("✗ Data integrity check failed - Hash mismatch")
        
        # Action buttons
        col_b1, col_b2, col_b3 = st.columns([1, 1, 2])
        
        with col_b1:
            if st.button("Start New Assessment", type="primary", use_container_width=True):
                st.session_state.assessment_completed = False
                st.session_state.current_result = None
                st.session_state.current_data = None
                st.session_state.current_hash = None
                st.rerun()
        
        with col_b2:
            if st.button("View History", use_container_width=True):
                st.session_state.page = "Assessment History"
                st.rerun()

# -------------------- Assessment History Page --------------------
elif st.session_state.page == "Assessment History":
    st.title("Assessment History")
    
    # Load data
    df = load_assessment_history()
    
    if df.empty:
        st.info("No assessment records found. Create a new assessment to get started.")
        
        # Button to go to new assessment
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Create New Assessment", type="primary", use_container_width=True):
                st.session_state.page = "New Assessment"
                st.rerun()
    else:
        # Summary statistics
        st.subheader("Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Assessments", len(df))
        
        with col2:
            avg_score = df['risk_score'].mean()
            st.metric("Average Risk Score", f"{avg_score:.0f}")
        
        with col3:
            high_risk_count = len(df[df['risk_category'].str.contains('High', na=False)])
            st.metric("High Risk Cases", high_risk_count)
        
        with col4:
            avg_prob = df['default_probability'].mean()
            st.metric("Avg Default Probability", f"{avg_prob:.1%}")
        
        # Distribution charts
        st.subheader("Risk Distribution")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.hist(df['risk_score'], bins=20, color='steelblue', edgecolor='white', alpha=0.7)
            ax1.set_xlabel('Risk Score')
            ax1.set_ylabel('Count')
            ax1.set_title('Risk Score Distribution')
            ax1.axvline(x=df['risk_score'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df["risk_score"].mean():.0f}')
            ax1.legend()
            st.pyplot(fig1)
            plt.close()
        
        with col_chart2:
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            category_counts = df['risk_category'].value_counts()
            colors = ['#28a745', '#5cb85c', '#f0ad4e', '#d9534f', '#c9302c']
            ax2.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
                    colors=colors[:len(category_counts)], startangle=90)
            ax2.set_title('Risk Category Distribution')
            st.pyplot(fig2)
            plt.close()
        
        # Assessment records table
        st.subheader("Assessment Records")
        
        # Format for display
        display_df = df.copy()
        display_df['default_probability'] = display_df['default_probability'].apply(
            lambda x: f"{x:.1%}"
        )
        display_df['assessment_date'] = pd.to_datetime(display_df['assessment_date']).dt.strftime('%Y-%m-%d %H:%M')
        display_df['loan_amount'] = display_df['loan_amount'].apply(
            lambda x: f"GHS {x:,.0f}" if pd.notnull(x) else "N/A"
        )
        
        # Add selection column
        display_df['Select'] = False
        
        # Create data editor for selection
        edited_df = st.data_editor(
            display_df[['Select', 'applicant_id', 'applicant_name', 'risk_score', 
                       'risk_category', 'default_probability', 'assessment_date', 'loan_amount']],
            use_container_width=True,
            column_config={
                "Select": st.column_config.CheckboxColumn("Select", default=False),
                "applicant_id": "ID",
                "applicant_name": "Name",
                "risk_score": "Risk Score",
                "risk_category": "Category",
                "default_probability": "Default Prob",
                "assessment_date": "Date",
                "loan_amount": "Loan Amount"
            },
            hide_index=True,
            key="history_editor"
        )
        
        # Get selected rows
        selected_rows = edited_df[edited_df['Select']].index.tolist()
        
        # Action buttons
        col_act1, col_act2, col_act3, col_act4 = st.columns([1, 1, 1, 2])
        
        with col_act1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Export CSV",
                data=csv,
                file_name=f"assessments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_btn"
            )
        
        with col_act2:
            if selected_rows and st.button("Delete Selected", use_container_width=True):
                for idx in selected_rows:
                    row = df.iloc[idx]
                    delete_assessment(row['applicant_id'], row['assessment_date'])
                st.success(f"Deleted {len(selected_rows)} record(s)")
                st.rerun()
        
        with col_act3:
            if st.button("Clear All", use_container_width=True):
                # Use session state for confirmation
                if 'confirm_clear' not in st.session_state:
                    st.session_state.confirm_clear = True
                    st.rerun()
        
        with col_act4:
            if st.button("Refresh", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        # Confirmation dialog for clear all
        if 'confirm_clear' in st.session_state and st.session_state.confirm_clear:
            st.warning("Are you sure you want to delete ALL records?")
            col_conf1, col_conf2 = st.columns(2)
            with col_conf1:
                if st.button("Yes, Delete All", use_container_width=True):
                    clear_all_history()
                    st.session_state.confirm_clear = False
                    st.success("All records deleted")
                    st.rerun()
            with col_conf2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.confirm_clear = False
                    st.rerun()
        
        # Detailed view for selected record
        if selected_rows and len(selected_rows) == 1:
            idx = selected_rows[0]
            record = df.iloc[idx]
            
            st.markdown("---")
            st.subheader(f"Details for {record['applicant_name']}")
            
            col_det1, col_det2 = st.columns(2)
            
            with col_det1:
                st.text(f"Applicant ID: {record['applicant_id']}")
                st.text(f"Email: {record['applicant_email']}")
                st.text(f"Assessment Date: {record['assessment_date']}")
                st.text(f"Credit Score: {record['credit_score']}")
                st.text(f"Age: {record['age']}")
            
            with col_det2:
                st.text(f"Risk Score: {record['risk_score']}/1000")
                st.text(f"Default Probability: {record['default_probability']}")
                st.text(f"Risk Category: {record['risk_category']}")
                st.text(f"Loan Amount: {record['loan_amount']}")
                st.text(f"Employment: {record['employment_status']}")
            
            # Data hash
            with st.expander("Data Integrity Hash"):
                st.code(record['data_hash'], language="text")
        
        # New assessment button
        st.markdown("---")
        if st.button("+ Create New Assessment", type="primary", use_container_width=True):
            st.session_state.page = "New Assessment"
            st.rerun()

# -------------------- Footer --------------------
st.markdown("---")
st.caption("Credit Risk Assessment System")
