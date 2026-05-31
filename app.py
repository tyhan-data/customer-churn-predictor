"""
Customer Churn Prediction App
A production-ready Streamlit application that predicts customer churn
using a pre-trained Random Forest model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""

<style>

    /* Main background */

    .stApp { background-color: #36454F; }



    /* Sidebar */

    [data-testid="stSidebar"] {

        background: linear-gradient(180deg, #1a1f2e 0%, #16213e 100%);

        border-right: 1px solid #2d3748;

    }



    /* Cards */

    .metric-card {

        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);

        border: 1px solid #2d3748;

        border-radius: 12px;

        padding: 1.5rem;

        margin: 0.5rem 0;

    }



    /* Result boxes */

    .churn-box {

        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);

        border: 1px solid #ef4444;

        border-radius: 12px;

        padding: 2rem;

        text-align: center;

    }

    .safe-box {

        background: linear-gradient(135deg, #14532d 0%, #166534 100%);

        border: 1px solid #22c55e;

        border-radius: 12px;

        padding: 2rem;

        text-align: center;

    }

    .result-text {

        font-size: 2rem;

        font-weight: 800;

        color: #ffffff;

        margin: 0;

    }

    .result-sub {

        font-size: 1rem;

        color: rgba(255,255,255,0.75);

        margin-top: 0.5rem;

    }



    /* Section headers */

    .section-header {

        font-size: 1rem;

        font-weight: 700;

        color: #7c3aed;

        text-transform: uppercase;

        letter-spacing: 0.1em;

        border-bottom: 2px solid #7c3aed;

        padding-bottom: 0.4rem;

        margin: 1.5rem 0 1rem 0;

    }



    /* Predict button */

    div[data-testid="stButton"] > button {

        width: 100%;

        background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%);

        color: white;

        border: none;

        border-radius: 10px;

        padding: 0.85rem 2rem;

        font-size: 1.1rem;

        font-weight: 700;

        letter-spacing: 0.05em;

        cursor: pointer;

        transition: all 0.2s;

    }

    div[data-testid="stButton"] > button:hover {

        transform: translateY(-2px);

        box-shadow: 0 8px 20px rgba(124,58,237,0.4);

    }



    /* Input styling */

    .stSelectbox > div, .stNumberInput > div { border-radius: 8px; }

    /* Hide Streamlit branding */

    #MainMenu {visibility: hidden;}

    footer {visibility: hidden;}

</style>

""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────
MODEL_DIR = Path(".")

@st.cache_resource(show_spinner="Loading prediction model...")
def load_artifacts():
    """Load model, encoder, and feature list from disk."""
    model   = joblib.load(MODEL_DIR / "model.pkl")
    encoder = joblib.load(MODEL_DIR / "encoder.pkl")
    features = joblib.load(MODEL_DIR / "features.pkl")
    return model, encoder, features

try:
    model, encoder, feature_cols = load_artifacts()
except FileNotFoundError as e:
    st.error(f"⚠️ Could not load model files: {e}\n\nMake sure model.pkl, encoder.pkl, and features.pkl are in the app directory.")
    st.stop()

# All columns the encoder was fit on (includes 'Churn' from training data)
# We pass a dummy 'No' for Churn at inference; its OHE output is dropped
# by reindex() since 'Churn_Yes' is not in feature_cols.
ALL_ENCODER_COLS = list(encoder.feature_names_in_)          # includes 'Churn'
CATEGORICAL_COLS = [c for c in ALL_ENCODER_COLS if c != "Churn"]   # user-input cols
NUMERICAL_COLS   = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📉 Churn Predictor")
    st.markdown("---")
    st.markdown("""
    **What is customer churn?**  
    Churn occurs when a customer stops doing business with a company.  
    Predicting churn helps businesses take *proactive* retention actions.

    ---
    **How it works**
    1. Fill in the customer profile on the right
    2. Click **Predict Churn**
    3. See the prediction + probability score

    ---
    **Model details**
    - 🌲 Random Forest Classifier
    - 🔢 30 engineered features
    - 📊 Trained on Telco Customer Churn dataset

    ---
    **Feature categories**
    | Category | Features |
    |---|---|
    | Demographics | Gender, Senior, Partner, Dependents |
    | Services | Phone, Internet, Security, Backup… |
    | Account | Contract, Billing, Payment, Charges |

    ---
    """)
    st.caption("Built with Streamlit · scikit-learn · Plotly")

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("# 📉 Customer Churn Prediction")
    st.markdown("Fill in the customer profile below and click **Predict Churn** to see results.")

st.markdown("---")

# ─────────────────────────────────────────────
# Input Form
# ─────────────────────────────────────────────
with st.form("churn_form"):

    # ── Demographics ──────────────────────────
    st.markdown('<div class="section-header">👤 Demographics</div>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with d2:
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    with d3:
        partner = st.selectbox("Partner", ["Yes", "No"])
    with d4:
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    # ── Account Info ──────────────────────────
    st.markdown('<div class="section-header">📋 Account Information</div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    with a1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12, step=1)
    with a2:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    with a3:
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    a4, a5, a6 = st.columns(3)
    with a4:
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
    with a5:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0,
                                          value=65.0, step=0.5, format="%.2f")
    with a6:
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0,
                                        value=float(tenure * 65), step=1.0, format="%.2f")

    # ── Phone Services ────────────────────────
    st.markdown('<div class="section-header">📱 Phone Services</div>', unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    with p1:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    with p2:
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

    # ── Internet Services ─────────────────────
    st.markdown('<div class="section-header">🌐 Internet Services</div>', unsafe_allow_html=True)
    i1, i2, i3, i4 = st.columns(4)
    with i1:
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    with i2:
        online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    with i3:
        online_bkp = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    with i4:
        device_prot = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

    i5, i6, i7 = st.columns(3)
    with i5:
        tech_sup = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    with i6:
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    with i7:
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍 Predict Churn")

# ─────────────────────────────────────────────
# Prediction Logic
# ─────────────────────────────────────────────
def preprocess_input(raw: dict) -> pd.DataFrame:
    """
    Replicates the training pipeline:
    1. Builds a DataFrame with all encoder columns (adds dummy 'Churn'='No')
    2. One-hot encodes using the saved encoder (drop='first', handle_unknown='ignore')
    3. Combines with numerical features
    4. Reindexes to exact feature order expected by the model
    The 'Churn' column was present during training; its OHE output is
    automatically excluded by the reindex step since it doesn't appear in feature_cols.
    """
    # Build categorical DataFrame — include dummy 'Churn' for encoder compatibility
    cat_input = {col: raw[col] for col in CATEGORICAL_COLS}
    cat_input["Churn"] = "No"   # dummy placeholder; its OHE columns are excluded by reindex
    cat_df = pd.DataFrame([cat_input])[ALL_ENCODER_COLS]  # preserve encoder column order

    # Apply OHE
    ohe_array = encoder.transform(cat_df)
    ohe_cols   = encoder.get_feature_names_out(ALL_ENCODER_COLS)
    ohe_df     = pd.DataFrame(ohe_array, columns=ohe_cols)

    # Numerical columns
    num_df = pd.DataFrame([{col: raw[col] for col in NUMERICAL_COLS}])

    # Combine and reindex to exact training feature order (also drops Churn OHE cols)
    full_df = pd.concat([num_df, ohe_df], axis=1)
    full_df = full_df.reindex(columns=feature_cols, fill_value=0)
    return full_df


if submitted:
    raw_input = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_bkp,
        "DeviceProtection": device_prot,
        "TechSupport": tech_sup,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    with st.spinner("Analyzing customer profile…"):
        X = preprocess_input(raw_input)
        proba = model.predict_proba(X)[0]          # [prob_no_churn, prob_churn]
        churn_prob  = float(proba[1])
        retain_prob = float(proba[0])
        prediction  = int(model.predict(X)[0])     # 0 = No Churn, 1 = Churn

    st.markdown("---")
    st.markdown("## 🎯 Prediction Results")

    # ── Verdict ───────────────────────────────
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        if prediction == 1:
            st.markdown(f"""
            <div class="churn-box">
                <p class="result-text">⚠️ Will Churn</p>
                <p class="result-sub">This customer is likely to leave</p>
                <p class="result-sub" style="font-size:1.5rem;font-weight:800;color:#fca5a5;">
                    {churn_prob:.1%} probability
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="safe-box">
                <p class="result-text">✅ Will NOT Churn</p>
                <p class="result-sub">This customer is likely to stay</p>
                <p class="result-sub" style="font-size:1.5rem;font-weight:800;color:#86efac;">
                    {retain_prob:.1%} probability
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ── Probability Gauge ─────────────────────
    with res_col2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=churn_prob * 100,
            number={"suffix": "%", "font": {"size": 32, "color": "#ffffff"}},
            title={"text": "Churn Probability", "font": {"size": 16, "color": "#a0aec0"}},
            delta={"reference": 50, "increasing": {"color": "#ef4444"},
                   "decreasing": {"color": "#22c55e"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#4a5568",
                         "tickfont": {"color": "#a0aec0"}},
                "bar": {"color": "#ef4444" if churn_prob > 0.5 else "#22c55e",
                        "thickness": 0.3},
                "bgcolor": "#1a1f2e",
                "bordercolor": "#2d3748",
                "steps": [
                    {"range": [0,  30], "color": "#14532d"},
                    {"range": [30, 60], "color": "#713f12"},
                    {"range": [60, 100], "color": "#7f1d1d"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 3},
                    "thickness": 0.85,
                    "value": 50,
                },
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
            height=260,
            margin=dict(t=40, b=20, l=30, r=30),
            font={"color": "#ffffff"},
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Probability Bar ───────────────────────
    st.markdown("#### Probability Breakdown")
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=["Will NOT Churn", "Will Churn"],
        y=[retain_prob * 100, churn_prob * 100],
        marker_color=["#22c55e", "#ef4444"],
        text=[f"{retain_prob:.1%}", f"{churn_prob:.1%}"],
        textposition="outside",
        textfont={"color": "#ffffff", "size": 14},
        width=0.4,
    ))
    fig_bar.update_layout(
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1a1f2e",
        height=280,
        margin=dict(t=20, b=20, l=30, r=30),
        yaxis=dict(range=[0, 115], ticksuffix="%", gridcolor="#2d3748",
                   tickfont={"color": "#a0aec0"}),
        xaxis=dict(tickfont={"color": "#ffffff", "size": 14}),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Key Risk Factors ──────────────────────
    st.markdown("#### 📌 Key Risk Indicators")
    risk_factors = []

    if contract == "Month-to-month":
        risk_factors.append(("📅 Month-to-month contract", "High churn risk — no long-term commitment", "high"))
    if internet == "Fiber optic":
        risk_factors.append(("🔌 Fiber optic service", "Fiber customers churn more than DSL", "medium"))
    if tenure < 12:
        risk_factors.append(("⏳ Low tenure (< 12 months)", "New customers are more likely to churn", "high"))
    if payment == "Electronic check":
        risk_factors.append(("💳 Electronic check payment", "Associated with higher churn rates", "medium"))
    if online_sec == "No" and internet != "No":
        risk_factors.append(("🔒 No online security", "Customers without security features churn more", "low"))
    if tech_sup == "No" and internet != "No":
        risk_factors.append(("🛠 No tech support", "Lack of support drives dissatisfaction", "low"))

    if risk_factors:
        cols = st.columns(min(len(risk_factors), 3))
        color_map = {"high": "#ef4444", "medium": "#f59e0b", "low": "#3b82f6"}
        for i, (title, desc, level) in enumerate(risk_factors):
            with cols[i % 3]:
                c = color_map[level]
                st.markdown(f"""
                <div style="background:#1a1f2e;border:1px solid {c};border-radius:10px;
                            padding:1rem;margin:0.3rem 0;">
                    <b style="color:{c}">{title}</b>
                    <p style="color:#a0aec0;font-size:0.85rem;margin:0.3rem 0 0 0">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.success("✅ No major risk indicators detected for this customer profile.")

    # ── Processed Feature Preview (collapsed) ─
    with st.expander("🔬 View processed feature vector"):
        st.dataframe(X.T.rename(columns={0: "Value"}), use_container_width=True)
