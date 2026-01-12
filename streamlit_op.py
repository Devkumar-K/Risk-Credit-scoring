import streamlit as st
import numpy as np
import pandas as pd
import json
from PIL import Image
import os

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="Unified Credit Risk System",
    layout="wide"
)

st.title("💳 Unified Credit Risk Prediction System")
st.write(
    """
    **Linear SVM implemented from scratch**  
    - Card Approval  
    - Credit Default  
    """
)

# =====================================================
# TASK SELECTION
# =====================================================
task = st.sidebar.selectbox(
    "Select Prediction Task",
    ["Card Approval", "Credit Default"]
)

# =====================================================
# PATH CONFIG
# =====================================================
if task == "Card Approval":
    model_path = "model/card_approval_model.json"
    prefix = "card"
    positive_label = "APPROVED ✅"
    negative_label = "REJECTED ❌"
else:
    model_path = "model/credit_default_model.json"
    prefix = "credit"
    positive_label = "DEFAULT ❌"
    negative_label = "NO DEFAULT ✅"

# =====================================================
# LOAD MODEL ARTIFACTS
# =====================================================
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()

with open(model_path, "r") as f:
    artifacts = json.load(f)

weights = np.array(artifacts["weights"])
features = artifacts["features"]
metrics = artifacts["metrics"]

# =====================================================
# USER INPUT
# =====================================================
st.sidebar.header("Manual Feature Input")

user_vals = []
for feat in features:
    val = st.sidebar.number_input(
        label=feat,
        value=0.0
    )
    user_vals.append(val)

X_input = np.array(user_vals).reshape(1, -1)

# =====================================================
# PREDICTION
# =====================================================
Xb = np.c_[np.ones(1), X_input]
decision_score = float(np.dot(Xb, weights))
prediction = 1 if decision_score >= 0 else -1

st.subheader("📌 Prediction Result")

if prediction == 1:
    st.success(positive_label)
else:
    st.error(negative_label)

st.write(f"**Decision Score:** `{decision_score:.4f}`")

st.info(
    """
    **Score interpretation**
    - Positive score → Positive class
    - Negative score → Negative class
    - Larger |score| → Higher confidence
    """
)

# =====================================================
# METRICS
# =====================================================
st.subheader("📊 Model Performance Metrics")
st.json(metrics)

# =====================================================
# VISUALIZATIONS
# =====================================================
st.subheader("📈 Model Visualizations")

def show_image(title, path):
    if os.path.exists(path):
        st.markdown(f"**{title}**")
        st.image(
            path,
            use_container_width=True  # ✅ FIXED
        )
    else:
        st.warning(f"Missing file: {path}")


col1, col2 = st.columns(2)

with col1:
    show_image(
        "Confusion Matrix",
        f"figures/{prefix}_confusion_matrix.png"
    )

with col2:
    show_image(
        "ROC Curve",
        f"figures/{prefix}_roc_curve.png"
    )

col3, col4 = st.columns(2)

with col3:
    show_image(
        "Feature Importance",
        f"figures/{prefix}_feature_importance.png"
    )

with col4:
    show_image(
        "Decision Score Distribution",
        f"figures/{prefix}_score_distribution.png"
    )

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption(
    "✔ Dataset-driven features  "
    "✔ No data leakage  "
    "✔ SVM from scratch  "
    "✔ Production-safe inference"
)
