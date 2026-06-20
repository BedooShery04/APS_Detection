import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path

st.title("📊 Model Information & Performance")

# --------------------------------------------------
# PATH CONFIG
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_PATH = BASE_DIR / "models" / "training_metrics.json"


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def load_training_metrics():
    try:
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning("Training benchmark metrics file not found.")
        return None
    except Exception as e:
        st.error(f"Failed to load training metrics: {e}")
        return None


def plot_confusion_matrix_plotly(cm, title):
    """Create interactive confusion matrix using Plotly"""
    cm = np.array(cm)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Other Faliuers (0)", "APS Failure (1)"],
        y=["Other Faliuers (0)", "APS Failure (1)"],
        text=cm,
        texttemplate="%{text:,}",
        textfont={"size": 20, "weight": "bold"},
        colorscale="Blues",
        showscale=True,
        zmin=0,
        zmax=cm.max(),
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, weight="bold")
        ),
        xaxis=dict(
            title=dict(
                text="Predicted",
                font=dict(size=18, weight="bold")
            ),
            tickfont=dict(size=18)
        ),
        yaxis=dict(
            title=dict(
                text="Actual",
                font=dict(size=18, weight="bold")
            ),
            tickfont=dict(size=18)
        ),
        width=600,
        height=500,
        autosize=True
    )
    
    st.plotly_chart(fig, use_container_width=True)


def calculate_cost(cm):
    tn, fp = cm[0]
    fn, tp = cm[1]
    return (fn * 500) + (fp * 10)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
training_metrics = load_training_metrics()
live_results = st.session_state.evaluation_results


# ==================================================
# SECTION A — TRAINING BENCHMARK METRICS
# ==================================================
st.markdown("## 🏆 Training Benchmark Performance")

if training_metrics:
    rows = []

    for model_name, metrics in training_metrics.items():
        cm = metrics["cm"]
        cost = calculate_cost(cm)

        rows.append({
            "Model": model_name,
            "Accuracy": round(metrics["accuracy"], 4),
            "Recall": round(metrics["recall"], 4),
            "F1": round(metrics["f1"], 4),
            "Cost": cost
        })

    benchmark_df = pd.DataFrame(rows)

    st.dataframe(
        benchmark_df.set_index("Model"),
        use_container_width=True
    )

    # --------------------------------------------------
    # BENCHMARK BAR CHART - Plotly Version
    # --------------------------------------------------
    st.markdown("### 📈 Benchmark Metric Comparison")

    # Create grouped bar chart with Plotly
    fig_bar = go.Figure()
    
    metrics_to_plot = ["Accuracy", "Recall", "F1"]
    colors = ["#636EFA", "#EF553B", "#00CC96"]
    
    for i, metric in enumerate(metrics_to_plot):
        fig_bar.add_trace(go.Bar(
            name=metric,
            x=benchmark_df["Model"],
            y=benchmark_df[metric],
            text=benchmark_df[metric].apply(lambda x: f"{x:.3f}"),
            textposition="outside",
            marker_color=colors[i],
            opacity=0.85
        ))
    
    fig_bar.update_layout(
        title=dict(
            text="Training Benchmark Metrics",
            font=dict(size=16, weight="bold")
        ),
        xaxis=dict(
            title=dict(
                text="Model",
                font=dict(size=12, weight="bold")
            ),
            tickfont=dict(size=12, weight="bold")
        ),
        yaxis=dict(
            title=dict(
                text="Score",
                font=dict(size=12, weight="bold")
            ),
            range=[0, 1.1],
            gridcolor="lightgray",
            tickformat=".3f"
        ),
        barmode="group",
        bargap=0.15,
        bargroupgap=0.1,
        height=500,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Additional radar chart for metrics comparison
    st.markdown("### 📊 Metrics Radar Chart")
    
    fig_radar = go.Figure()
    
    for idx, row in benchmark_df.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row["Accuracy"], row["Recall"], row["F1"]],
            theta=["Accuracy", "Recall", "F1"],
            fill="toself",
            name=row["Model"],
            line=dict(width=2),
            opacity=0.7
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat=".2f"
            )
        ),
        title=dict(
            text="Model Performance Comparison",
            font=dict(size=16, weight="bold")
        ),
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)

    # --------------------------------------------------
    # BENCHMARK CONFUSION MATRICES
    # --------------------------------------------------
    st.markdown("### 🔲 Training Confusion Matrices")

    tabs = st.tabs(list(training_metrics.keys()))

    for tab, (model_name, metrics) in zip(tabs, training_metrics.items()):
        with tab:
            cm = metrics["cm"]

            plot_confusion_matrix_plotly(
                cm,
                f"{model_name} — Training Benchmark"
            )

            tn, fp = cm[0]
            fn, tp = cm[1]
            
            # Display metrics in a nice format
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("True Negatives (TN)", f"{tn:,}")
                st.metric("False Positives (FP)", f"{fp:,}")
            
            with col2:
                st.metric("False Negatives (FN)", f"{fn:,}")
                st.metric("True Positives (TP)", f"{tp:,}")
            
            st.metric("💰 APS Cost", f"{calculate_cost(cm):,}", delta=None)

else:
    st.info("Training benchmark metrics not available.")

# ==================================================
# SECTION B — LIVE EVALUATION
# ==================================================
st.markdown("---")
st.markdown("## 🧪 Live Uploaded Dataset Evaluation")

if live_results:
    live_rows = []

    for model_name, result in live_results.items():
        if "metrics" not in result:
            continue

        metrics = result["metrics"]

        live_rows.append({
            "Model": model_name,
            "Accuracy": round(metrics["accuracy"], 4),
            "Precision": round(metrics["precision"], 4),
            "Recall": round(metrics["recall"], 4),
            "F1": round(metrics["f1"], 4),
            "Cost": metrics["cost"]
        })

    if live_rows:
        live_df = pd.DataFrame(live_rows)

        st.dataframe(
            live_df.set_index("Model"),
            use_container_width=True
        )
        
        # Live evaluation comparison chart
        st.markdown("### 📊 Live Evaluation Metrics Comparison")
        
        fig_live = go.Figure()
        
        live_metrics = ["Accuracy", "Precision", "Recall", "F1"]
        colors_live = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
        
        for i, metric in enumerate(live_metrics):
            fig_live.add_trace(go.Bar(
                name=metric,
                x=live_df["Model"],
                y=live_df[metric],
                text=live_df[metric].apply(lambda x: f"{x:.4f}"),
                textposition="outside",
                marker_color=colors_live[i],
                opacity=0.85
            ))
        
        fig_live.update_layout(
            title=dict(
                text="Live Dataset Performance Metrics",
                font=dict(size=16, weight="bold")
            ),
            xaxis=dict(
                title=dict(
                    text="Model",
                    font=dict(size=12, weight="bold")
                ),
                tickfont=dict(size=12, weight="bold")
            ),
            yaxis=dict(
                title=dict(
                    text="Score",
                    font=dict(size=12, weight="bold")
                ),
                range=[0, 1.1],
                gridcolor="lightgray",
                tickformat=".4f"
            ),
            barmode="group",
            bargap=0.15,
            bargroupgap=0.1,
            height=500,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_live, use_container_width=True)
        
        # Cost comparison chart
        st.markdown("### 💰 Cost Comparison")
        
        fig_cost = go.Figure()
        
        fig_cost.add_trace(go.Bar(
            x=live_df["Model"],
            y=live_df["Cost"],
            text=live_df["Cost"].apply(lambda x: f"{x:,.0f}"),
            textposition="outside",
            marker_color="#FF6B6B",
            opacity=0.85
        ))
        
        fig_cost.update_layout(
            title=dict(
                text="APS Cost per Model (Live Evaluation)",
                font=dict(size=16, weight="bold")
            ),
            xaxis=dict(
                title=dict(
                    text="Model",
                    font=dict(size=12, weight="bold")
                ),
                tickfont=dict(size=12, weight="bold")
            ),
            yaxis=dict(
                title=dict(
                    text="Cost",
                    font=dict(size=12, weight="bold")
                ),
                gridcolor="lightgray"
            ),
            height=400,
            hovermode="x unified"
        )
        
        st.plotly_chart(fig_cost, use_container_width=True)

        st.markdown("### 🔲 Live Confusion Matrices")

        live_tabs = st.tabs([row["Model"] for row in live_rows])

        for tab, row in zip(live_tabs, live_rows):
            with tab:
                metrics = live_results[row["Model"]]["metrics"]

                plot_confusion_matrix_plotly(
                    metrics["cm"],
                    f"{row['Model']} — Live Evaluation"
                )
                
                # Add performance gauge for each model
                st.markdown("#### Performance Gauges")

                def create_gauge(value, title, color):
                    """Create a consistent gauge chart"""
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=value,
                        title={"text": title},
                        gauge={
                            "axis": {"range": [0, 1], "tickformat": ".2f"},
                            "bar": {"color": color},
                            "steps": [
                                {"range": [0, 0.6], "color": "lightgray"},
                                {"range": [0.6, 0.8], "color": "gray"},
                                {"range": [0.8, 1], "color": "darkgray"}
                            ],
                            "threshold": {
                                "line": {"color": "red", "width": 4},
                                "thickness": 0.75
                            }
                        }
                    ))
                    fig.update_layout(height=250)
                    return fig

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.plotly_chart(create_gauge(metrics["accuracy"], "Accuracy", "#636EFA"), use_container_width=True)

                with col2:
                    st.plotly_chart(create_gauge(metrics["recall"], "Recall", "#EF553B"), use_container_width=True)

                with col3:
                    st.plotly_chart(create_gauge(metrics["f1"], "F1 Score", "#00CC96"), use_container_width=True)

    else:
        st.info("Uploaded dataset has no labels. Live evaluation unavailable.")

else:
    st.info("Run predictions first to see live evaluation metrics.")

# ==================================================
# SECTION C — MODEL DETAILS
# ==================================================
st.markdown("---")
st.markdown("## 🏗️ Model Architecture Details")

arch_tabs = st.tabs([
    "Logistic Regression",
    "Random Forest",
    "LightGBM"
])

with arch_tabs[0]:
    st.markdown("""
    **Logistic Regression**
    
    - Linear baseline classifier
    - Solver: `lbfgs`
    - Max iterations: 1000
    - Trained on SMOTE-balanced APS data
    - Fast and interpretable
    - Less effective for complex nonlinear patterns
    """)

with arch_tabs[1]:
    st.markdown("""
    **Random Forest**
    
    - Ensemble of decision trees
    - Trained on SMOTE-balanced APS data
    - Robust to noise and outliers
    - Good feature importance analysis
    - Strong general-purpose tabular model
    """)

with arch_tabs[2]:
    st.markdown("""
    **LightGBM**
    
    - Gradient boosting decision trees
    - `n_estimators=300`
    - `learning_rate=0.05`
    - `num_leaves=63`
    - Excellent for high-dimensional tabular data
    - Fast inference
    """)

# ==================================================
# SECTION D — PREPROCESSING PIPELINE
# ==================================================
st.markdown("---")
st.markdown("## ⚙️ Preprocessing Pipeline")

# Create a funnel/flow visualization for preprocessing steps
steps = [
    "Drop high-missing columns (>60%)",
    "Median imputation",
    "VarianceThreshold",
    "StandardScaler",
    "Correlation pruning",
    "Feature selection (top 50)",
    "SMOTE (training only)"
]

# Create a timeline/flow chart
fig_flow = go.Figure()

fig_flow.add_trace(go.Scatter(
    x=list(range(len(steps))),
    y=[1] * len(steps),
    mode="markers+text",
    marker=dict(
        size=30,
        color=list(range(len(steps))),
        colorscale="Viridis",
        showscale=True,
        symbol="circle"
    ),
    text=steps,
    textposition="top center",
    textfont=dict(size=10),
    hovertemplate="Step: %{text}<extra></extra>"
))

fig_flow.update_layout(
    title=dict(
        text="Data Preprocessing Pipeline Flow",
        font=dict(size=16, weight="bold")
    ),
    xaxis=dict(
        title=dict(
            text="Processing Steps",
            font=dict(size=12, weight="bold")
        ),
        showticklabels=False,
        gridcolor="white"
    ),
    yaxis=dict(
        showticklabels=False,
        gridcolor="white",
        range=[0.8, 1.2]
    ),
    height=300,
    hovermode="closest"
)

st.plotly_chart(fig_flow, use_container_width=True)

# Detailed description in columns
cols = st.columns(4)

with cols[0]:
    st.markdown("**1. Column Filtering**")
    st.caption("Remove columns with >60% missing values")
    
    st.markdown("**2. Imputation**")
    st.caption("Median imputation for missing values")

with cols[1]:
    st.markdown("**3. Variance Filter**")
    st.caption("Remove near-constant features")
    
    st.markdown("**4. Normalization**")
    st.caption("StandardScaler normalization")

with cols[2]:
    st.markdown("**5. Correlation Pruning**")
    st.caption("Remove highly correlated features")
    
    st.markdown("**6. Feature Selection**")
    st.caption("Keep top 50 informative features")

with cols[3]:
    st.markdown("**7. SMOTE Balancing**")
    st.caption("Balance APS failure class (training only)")