"""
Smart CreditWise — Loan Risk Assessment Dashboard
Streamlit frontend only. Loads a pre-trained scikit-learn pipeline
(preprocessing + classifier) and never retrains it.

Run:
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart CreditWise",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Constants — exact feature names/order the trained pipeline expects.
# Taken from ml/preprocessing.py (CATEGORICAL_FEATURES + NUMERIC_FEATURES).
# Do not rename these; the ColumnTransformer selects columns by name.
# --------------------------------------------------------------------------
CATEGORICAL_FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]
NUMERIC_FEATURES = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# The training script strips "+" from Dependents ("3+" -> "3") before
# fitting, so the encoder only knows "0", "1", "2", "3".
DEPENDENTS_DISPLAY_TO_MODEL = {"0": "0", "1": "1", "2": "2", "3+": "3"}

# Candidate file locations — checks the names given in the spec first,
# then falls back to the actual paths shipped in this project's zip.
MODEL_PATH_CANDIDATES = [
    "models/loan_risk_pipeline.pkl",
    "ml/model/model.joblib",
    "model/model.joblib",
    "model.joblib",
]
METRICS_PATH_CANDIDATES = [
    "models/metrics.json",
    "ml/model/metrics.json",
    "model/metrics.json",
    "metrics.json",
]


def _first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None


@st.cache_resource(show_spinner="Loading trained pipeline...")
def load_pipeline():
    path = _first_existing(MODEL_PATH_CANDIDATES)
    if path is None:
        return None, None
    return joblib.load(path), path


@st.cache_data(show_spinner=False)
def load_metrics():
    path = _first_existing(METRICS_PATH_CANDIDATES)
    if path is None:
        return None
    with open(path, "r") as f:
        return json.load(f)


pipeline, model_path = load_pipeline()
metrics = load_metrics()

# --------------------------------------------------------------------------
# Light custom styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main .block-container { padding-top: 2rem; }
    .cw-hero {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 2.2rem 2rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .cw-hero h1 { margin-bottom: 0.2rem; font-size: 2.1rem; }
    .cw-hero p { opacity: 0.9; font-size: 1.02rem; margin-bottom: 0; }
    .cw-card {
        background: #ffffff10;
        border: 1px solid #ffffff22;
        border-radius: 12px;
        padding: 1rem 1.2rem;
    }
    .risk-badge {
        display: inline-block;
        padding: 0.45rem 1.1rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.05rem;
    }
    .risk-low { background: #d1f7dc; color: #0a6b2d; }
    .risk-medium { background: #fff3cd; color: #8a6100; }
    .risk-high { background: #fddede; color: #a11212; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------
st.sidebar.markdown("## 💳 Smart CreditWise")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Risk Assessment", "Model Insights", "Security & Privacy", "About"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
if pipeline is not None:
    st.sidebar.success(f"Model loaded from:\n`{model_path}`")
else:
    st.sidebar.error(
        "Model file not found. Expected one of:\n"
        + "\n".join(f"- `{p}`" for p in MODEL_PATH_CANDIDATES)
    )
if metrics is not None:
    st.sidebar.caption(f"Selected model: **{metrics.get('selected_model', 'n/a')}**")


def risk_from_probability(p_approve: float):
    """Higher approval probability -> lower risk (documented app behavior)."""
    if p_approve >= 0.70:
        return "Low", "risk-low"
    elif p_approve >= 0.40:
        return "Medium", "risk-medium"
    else:
        return "High", "risk-high"


def get_feature_importance(pipe):
    """Return (dataframe, label) of importance/coefficient magnitude, or (None, None)."""
    try:
        preprocessor = pipe.named_steps["preprocess"]
        model = pipe.named_steps["model"]
        feature_names = preprocessor.get_feature_names_out()
        if hasattr(model, "feature_importances_"):
            values = model.feature_importances_
            label = "Feature Importance"
        elif hasattr(model, "coef_"):
            values = np.abs(model.coef_[0])
            label = "Coefficient Magnitude (|coef|)"
        else:
            return None, None
        df = pd.DataFrame({"Feature": feature_names, "Score": values})
        df = df.sort_values("Score", ascending=False).reset_index(drop=True)
        return df, label
    except Exception:
        return None, None


# ==========================================================================
# HOME
# ==========================================================================
if page == "Home":
    st.markdown(
        """
        <div class="cw-hero">
            <h1>💳 Smart CreditWise</h1>
            <p>An interactive loan risk assessment dashboard powered by a trained
            machine learning pipeline. Enter applicant details and get an
            instant approval-probability–based risk read.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    if metrics:
        best = metrics["all_models"][metrics["selected_model"]]
        col1.metric("Selected Model", metrics["selected_model"].replace("_", " ").title())
        col2.metric("Test Accuracy", f"{best['accuracy']*100:.1f}%")
        col3.metric("ROC-AUC", f"{best['roc_auc']:.3f}")
        col4.metric("Test Set Size", metrics.get("test_set_size", "n/a"))
    else:
        col1.metric("Selected Model", "—")
        col2.metric("Test Accuracy", "—")
        col3.metric("ROC-AUC", "—")
        col4.metric("Test Set Size", "—")

    st.markdown("### How it works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 1. Enter details")
        st.write(
            "Fill in applicant, co-applicant, and loan information on the "
            "**Risk Assessment** page."
        )
    with c2:
        st.markdown("#### 2. Model scores it")
        st.write(
            "The saved pipeline preprocesses your inputs exactly as during "
            "training and outputs an approval probability."
        )
    with c3:
        st.markdown("#### 3. Get a risk read")
        st.write(
            "A higher approval probability maps to a lower risk category — "
            "Low, Medium, or High."
        )

    st.info(
        "This is a portfolio/educational project. The model predicts the "
        "probability a loan would be **approved** based on historical "
        "patterns in a public dataset; it is not a real underwriting or "
        "credit-risk system.",
        icon="ℹ️",
    )

# ==========================================================================
# RISK ASSESSMENT
# ==========================================================================
elif page == "Risk Assessment":
    st.title("🔍 Risk Assessment")
    st.write(
        "Enter the applicant's details below. Fields match exactly what the "
        "trained pipeline was built on."
    )

    if pipeline is None:
        st.error(
            "Can't run an assessment — the model file wasn't found. Check the "
            "sidebar for the paths that were searched."
        )
    else:
        with st.form("risk_form"):
            st.subheader("Applicant Information")
            c1, c2, c3 = st.columns(3)
            with c1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                married = st.selectbox("Married", ["Yes", "No"])
            with c2:
                dependents_display = st.selectbox(
                    "Dependents", ["0", "1", "2", "3+"]
                )
                education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            with c3:
                self_employed = st.selectbox("Self Employed", ["No", "Yes"])
                property_area = st.selectbox(
                    "Property Area", ["Urban", "Semiurban", "Rural"]
                )

            st.subheader("Financial Information")
            c4, c5, c6 = st.columns(3)
            with c4:
                applicant_income = st.number_input(
                    "Applicant Income (monthly)", min_value=0, value=5000, step=100
                )
                coapplicant_income = st.number_input(
                    "Coapplicant Income (monthly)", min_value=0, value=0, step=100
                )
            with c5:
                loan_amount = st.number_input(
                    "Loan Amount (in thousands)", min_value=0, value=128, step=1
                )
                loan_term = st.selectbox(
                    "Loan Amount Term (days)",
                    [12, 36, 60, 84, 120, 180, 240, 300, 360, 480],
                    index=8,
                )
            with c6:
                credit_history_display = st.selectbox(
                    "Credit History Meets Guidelines?", ["Yes", "No"]
                )

            submitted = st.form_submit_button(
                "Assess Loan Risk", use_container_width=True, type="primary"
            )

        if submitted:
            input_row = pd.DataFrame(
                [
                    {
                        "Gender": gender,
                        "Married": married,
                        "Dependents": DEPENDENTS_DISPLAY_TO_MODEL[dependents_display],
                        "Education": education,
                        "Self_Employed": self_employed,
                        "Property_Area": property_area,
                        "ApplicantIncome": applicant_income,
                        "CoapplicantIncome": coapplicant_income,
                        "LoanAmount": loan_amount,
                        "Loan_Amount_Term": loan_term,
                        "Credit_History": 1 if credit_history_display == "Yes" else 0,
                    }
                ]
            )[ALL_FEATURES]

            try:
                prediction = pipeline.predict(input_row)[0]
                proba = None
                if hasattr(pipeline, "predict_proba"):
                    proba = pipeline.predict_proba(input_row)[0][1]
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

            st.markdown("---")
            st.subheader("Result")

            r1, r2 = st.columns([1, 1])
            with r1:
                if prediction == 1:
                    st.success("✅ Predicted outcome: **Loan Approved**")
                else:
                    st.error("❌ Predicted outcome: **Loan Not Approved**")

                if proba is not None:
                    st.metric("Approval Probability", f"{proba*100:.1f}%")
                    st.progress(float(proba))

            with r2:
                if proba is not None:
                    risk_label, risk_class = risk_from_probability(proba)
                    st.markdown("**Risk Category**")
                    st.markdown(
                        f'<span class="risk-badge {risk_class}">{risk_label} Risk</span>',
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "Risk is derived from the model's predicted approval "
                        "probability: higher approval probability → lower risk. "
                        "This is a portfolio proxy for credit risk, not a real "
                        "underwriting risk score."
                    )
                else:
                    st.warning("This model does not expose probability estimates.")

            with st.expander("View submitted input"):
                st.dataframe(input_row, use_container_width=True)

# ==========================================================================
# MODEL INSIGHTS
# ==========================================================================
elif page == "Model Insights":
    st.title("📊 Model Insights")

    if metrics is None:
        st.error(
            "Metrics file not found. Expected one of:\n"
            + "\n".join(f"- `{p}`" for p in METRICS_PATH_CANDIDATES)
        )
    else:
        selected = metrics["selected_model"]
        st.markdown(f"**Selected model:** `{selected}`  (chosen by highest ROC-AUC on the held-out test set)")
        st.caption(
            f"Train rows: {metrics.get('train_set_size', 'n/a')} · "
            f"Test rows: {metrics.get('test_set_size', 'n/a')}"
        )

        st.subheader("Model comparison (test set)")
        rows = []
        for name, m in metrics["all_models"].items():
            row = {"Model": name.replace("_", " ").title(), **m}
            rows.append(row)
        comp_df = pd.DataFrame(rows).set_index("Model")
        st.dataframe(
            comp_df.style.highlight_max(axis=0, color="#d1f7dc"),
            use_container_width=True,
        )

        best = metrics["all_models"][selected]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{best['accuracy']*100:.1f}%")
        c2.metric("Precision", f"{best['precision']*100:.1f}%")
        c3.metric("Recall", f"{best['recall']*100:.1f}%")
        c4.metric("F1 Score", f"{best['f1']*100:.1f}%")
        c5.metric("ROC-AUC", f"{best['roc_auc']:.3f}")

        st.subheader("About the selected model")
        model_descriptions = {
            "logistic_regression": (
                "Logistic Regression models the log-odds of loan approval as a "
                "linear combination of the input features after preprocessing "
                "(imputation, one-hot encoding, scaling). It was selected here "
                "because it had the highest ROC-AUC among the candidates tried."
            ),
            "random_forest": (
                "Random Forest is an ensemble of decision trees trained on "
                "bootstrapped samples, averaging their votes to reduce "
                "overfitting compared to a single tree."
            ),
            "gradient_boosting": (
                "Gradient Boosting builds trees sequentially, each one "
                "correcting the errors of the previous ensemble."
            ),
        }
        st.write(
            model_descriptions.get(
                selected, "Model description not available for this model type."
            )
        )

        if pipeline is not None:
            st.subheader("Feature importance")
            importance_df, label = get_feature_importance(pipeline)
            if importance_df is not None:
                st.caption(
                    f"{label} — for Logistic Regression this reflects each "
                    "encoded feature's linear influence, not a causal effect."
                )
                st.bar_chart(importance_df.set_index("Feature")["Score"].head(15))
                with st.expander("View full table"):
                    st.dataframe(importance_df, use_container_width=True)
            else:
                st.info("Feature importance is not available for this model type.")

        st.markdown("---")
        st.subheader("What \"risk\" means in this app")
        st.write(
            "The model predicts the probability a loan would be **approved**, "
            "based on patterns in a public historical dataset. The app maps "
            "that probability to a Low / Medium / High risk label — a higher "
            "approval probability means lower risk. This is an educational "
            "proxy for credit risk, not a real underwriting model."
        )

# ==========================================================================
# SECURITY & PRIVACY
# ==========================================================================
elif page == "Security & Privacy":
    st.title("🔒 Security & Privacy")

    st.write(
        "This page describes, plainly and accurately, how this demo "
        "application handles the information you enter."
    )

    st.subheader("How your inputs are used")
    st.markdown(
        """
        - The values you enter on the **Risk Assessment** page are used only
          to build a single row of data that is passed to the loaded model
          pipeline for a prediction.
        - Inputs are processed **in-memory, during your session**, to
          produce the prediction shown on screen.
        - This application, as built, does **not** write your inputs to a
          database, log file, or any external service.
        """
    )

    st.subheader("What this app does not claim")
    st.markdown(
        """
        - It does **not** implement authentication, encryption at rest,
          audit logging, or other enterprise security controls — those are
          not part of this codebase.
        - It does **not** perform identity verification or connect to any
          credit bureau or financial institution.
        - Data you enter is example/demo input only and should not include
          real personally identifying financial information.
        """
    )

    st.warning(
        "This is a portfolio/demo project, not a production system. Treat "
        "any data you enter here as non-sensitive test data.",
        icon="⚠️",
    )

# ==========================================================================
# ABOUT
# ==========================================================================
elif page == "About":
    st.title("ℹ️ About Smart CreditWise")

    st.write(
        "Smart CreditWise is a portfolio project that demonstrates an "
        "end-to-end, if intentionally small-scale, ML workflow: a trained "
        "scikit-learn pipeline served through an interactive Streamlit "
        "dashboard."
    )

    st.subheader("What the model does")
    st.write(
        "The pipeline predicts the probability that a loan application "
        "would be approved, based on a public historical loan-approval "
        "dataset. The app translates that probability into a Low / Medium "
        "/ High risk category as an educational proxy for credit risk."
    )

    st.subheader("Architecture")
    st.markdown(
        """
        - **Data & training:** pandas + scikit-learn (`ColumnTransformer` for
          preprocessing, several classifiers compared, best one selected by
          ROC-AUC on a held-out test set).
        - **Serving:** a single serialized `Pipeline` (preprocessing +
          model) loaded with `joblib` — no retraining happens in this app.
        - **Frontend:** this Streamlit application — runs entirely locally
          with `streamlit run app.py`, no cloud dependency required.
        - **Azure-ready:** the app has no hard-coded local-only assumptions
          (e.g., no local file writes required, config via relative paths),
          so it can be containerized and deployed to a service like Azure
          App Service or Azure Container Apps later. No Azure integration
          is implemented in this version.
        """
    )

    st.subheader("Tech stack")
    st.markdown("`Python` · `scikit-learn` · `pandas` · `joblib` · `Streamlit`")

    st.subheader("Limitations")
    st.markdown(
        """
        - Small dataset (a few hundred rows) from a single historical
          source — not representative of any real lender's criteria.
        - No fairness or bias audit has been performed.
        - Intended for a portfolio/demo, not real lending decisions.
        """
    )
