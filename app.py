import streamlit as st

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="APS Failure Detection",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# SESSION STATE INITIALIZATION
# --------------------------------------------------
defaults = {
    "uploaded_df": None,          # uploaded feature dataframe
    "true_labels": None,          # numeric labels (0/1) if available
    "evaluation_results": {},     # unified prediction/evaluation results
    "selected_model": None,       # currently selected model
    "training_metrics": None      # optional benchmark metrics from notebook
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        color: white;
    }

    .main-header h1 {
        font-size: 2.4rem;
        margin: 0;
    }

    .main-header p {
        font-size: 1.1rem;
        opacity: 0.85;
        margin-top: 0.4rem;
    }

    .info-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-left: 4px solid #0f3460;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        min-height: 140px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🚛 APS Failure Detection System</h1>
    <p>Scania Truck Predictive Maintenance · Menoufia University AI Project</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# STATUS BAR
# --------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state.uploaded_df is not None:
        df = st.session_state.uploaded_df
        st.success(
            f"✅ Dataset loaded · {df.shape[0]:,} rows · {df.shape[1]} features"
        )
    else:
        st.warning("⚠️ No dataset uploaded yet")

with col2:
    eval_results = st.session_state.evaluation_results

    if eval_results:
        model_names = ", ".join(eval_results.keys())
        st.success(
            f"✅ {len(eval_results)} model(s) evaluated: {model_names}"
        )
    else:
        st.info("🤖 No predictions/evaluations run yet")

with col3:
    if st.session_state.true_labels is not None:
        st.success("🏷️ Ground truth labels available")
    else:
        st.info("📋 Unlabeled dataset (prediction only)")

st.markdown("---")

# --------------------------------------------------
# WORKFLOW
# --------------------------------------------------
st.markdown("### Workflow")

steps = [
    (
        "📂",
        "Upload Dataset",
        "Upload APS CSV data. Labels are detected automatically if available."
    ),
    (
        "🔍",
        "Explore Data",
        "Inspect distributions, missing values, and dataset characteristics."
    ),
    (
        "🤖",
        "Predict & Evaluate",
        "Run one or all models. If labels exist, metrics are computed automatically."
    ),
    (
        "📊",
        "Model Analysis",
        "Compare benchmark training metrics with live uploaded dataset performance."
    ),
]

cols = st.columns(4)

for col, (icon, title, desc) in zip(cols, steps):
    with col:
        st.markdown(f"""
        <div class="info-card">
            <h3>{icon} {title}</h3>
            <p style="color:#555z; font-size:0.92rem;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# --------------------------------------------------
# SUPPORTED MODELS
# --------------------------------------------------
st.markdown("### Supported Models")

mc1, mc2, mc3 = st.columns(3)

with mc1:
    st.markdown("""
    **🔵 Logistic Regression**  
    Linear baseline · Fast inference · Interpretable
    """)

with mc2:
    st.markdown("""
    **🟢 Random Forest**  
    Ensemble trees · Robust to noise · Feature importance
    """)

with mc3:
    st.markdown("""
    **🟠 LightGBM**  
    Gradient boosting · Strong APS performance · Efficient inference
    """)

st.markdown("---")