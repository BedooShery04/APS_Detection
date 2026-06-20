import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from utils.model_loader import list_model_labels
from utils.predictor import predict, predict_all

st.title("🤖 Prediction & Model Comparison")

# --------------------------------------------------
# DATA CHECK
# --------------------------------------------------
if st.session_state.uploaded_df is None:
    st.warning("⚠️ Please upload a dataset first (Page 1).")
    st.stop()

df = st.session_state.uploaded_df
true_labels = st.session_state.true_labels


def display_prediction_labels(pred_df):
    """
    Convert numeric predictions to UI-friendly labels.
    Backend remains numeric.
    """
    df_display = pred_df.copy()
    df_display["Prediction"] = df_display["Prediction"].map({
        0: "Other",
        1: "APS Failure"
    })
    return df_display


# --------------------------------------------------
# MODE SELECTION
# --------------------------------------------------
mode = st.radio(
    "Prediction mode:",
    ["Compare All 3 Models", "Single Model"],
    horizontal=True
)

# ==================================================
# MODE A — COMPARE ALL
# ==================================================
if mode == "Compare All 3 Models":
    st.markdown(
        "Run **all available models** on the uploaded dataset "
        "and compare predictions side-by-side."
    )

    if st.button("🚀 Run All Models", type="primary"):
        with st.spinner("Running all models..."):
            try:
                all_results = predict_all(df, true_labels)
                st.session_state.evaluation_results = all_results
                st.session_state.selected_model = None

                st.success(
                    f"✅ Completed predictions for: {', '.join(all_results.keys())}"
                )

            except FileNotFoundError as e:
                st.error(f"Model or artifact file not found: {e}")
                st.stop()

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

    all_results = st.session_state.evaluation_results

    if all_results:
        model_names = list(all_results.keys())

        # --------------------------------------------------
        # SUMMARY TABLE
        # --------------------------------------------------
        st.markdown("### 📊 Summary Comparison")

        summary_rows = []

        for model_name, result in all_results.items():
            y_pred = result["y_pred"]

            failures = int((y_pred == 1).sum())
            onthers = int((y_pred == 0).sum())
            fail_pct = (failures / len(y_pred)) * 100

            row = {
                "Model": model_name,
                "Total Samples": len(y_pred),
                "APS Failures": failures,
                "Other Failures": onthers,
                "Failure Rate %": round(fail_pct, 2),
            }

            if "metrics" in result:
                row["Accuracy"] = round(result["metrics"]["accuracy"], 4)
                row["Recall"] = round(result["metrics"]["recall"], 4)
                row["F1"] = round(result["metrics"]["f1"], 4)
                row["Cost"] = result["metrics"]["cost"]

            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df.set_index("Model"), use_container_width=True)

        # --------------------------------------------------
        # BAR CHART - Plotly Version
        # --------------------------------------------------
        st.markdown("### 📈 Failure Count per Model")

        # Create grouped bar chart
        fig_bar = go.Figure(data=[
            go.Bar(
                name="APS Failure",
                x=model_names,
                y=[row["APS Failures"] for row in summary_rows],
                text=[row["APS Failures"] for row in summary_rows],
                textposition='auto',
                marker_color='#EF553B',
                opacity=0.85
            ),
            go.Bar(
                name="Other Failures",
                x=model_names,
                y=[row["Other Failures"] for row in summary_rows],
                text=[row["Other Failures"] for row in summary_rows],
                textposition='auto',
                marker_color='#636EFA',
                opacity=0.85
            )
        ])

        fig_bar.update_layout(
            title=dict(
                text="Prediction Counts per Model",
                font=dict(size=16, weight='bold')
            ),
            xaxis=dict(
                title="Model",
                tickfont=dict(size=12, weight='bold')
            ),
            yaxis=dict(
                title="Sample Count",
                gridcolor='lightgray'
            ),
            barmode='group',
            bargap=0.15,
            bargroupgap=0.1,
            height=500,
            hovermode='x unified'
        )

        st.plotly_chart(fig_bar, use_container_width=True)

        # --------------------------------------------------
        # AGREEMENT ANALYSIS
        # --------------------------------------------------
        st.markdown("### 🤝 Model Agreement")

        pred_matrix = pd.DataFrame({
            model_name: result["y_pred"]
            for model_name, result in all_results.items()
        })

        all_failure = int((pred_matrix.sum(axis=1) == len(model_names)).sum())
        all_Other = int((pred_matrix.sum(axis=1) == 0).sum())
        mixed = len(pred_matrix) - all_failure - all_Other

        c1, c2, c3 = st.columns(3)

        c1.metric("All agree: ABS Failure", f"{all_failure:,}")
        c2.metric("All agree: Other Failure", f"{all_Other:,}")
        c3.metric("Models disagree", f"{mixed:,}")

        # Add agreement pie chart
        fig_agreement = go.Figure(data=[go.Pie(
            labels=['All Agree: ABS Failure', 'All Agree: Other Faliure', 'Models Disagree'],
            values=[all_failure, all_Other, mixed],
            hole=0.3,
            marker_colors=['#EF553B', '#636EFA', '#FFA15A'],
            textinfo='label+percent',
            textposition='auto'
        )])

        fig_agreement.update_layout(
            title=dict(
                text="Model Agreement Distribution",
                font=dict(size=16, weight='bold')
            ),
            height=450
        )

        st.plotly_chart(fig_agreement, use_container_width=True)

        # --------------------------------------------------
        # PER MODEL TABS
        # --------------------------------------------------
        st.markdown("### 🔎 Per-Model Results")

        tabs = st.tabs(model_names)

        for tab, (model_name, result) in zip(tabs, all_results.items()):
            with tab:
                y_pred = result["y_pred"]

                failures = int((y_pred == 1).sum())
                onthers = int((y_pred == 0).sum())

                c1, c2, c3 = st.columns(3)
                c1.metric("Total", len(y_pred))
                c2.metric("ABS Failures 🔴", failures)
                c3.metric("Other Failures 🟢", onthers)

                if "metrics" in result:
                    m = result["metrics"]

                    st.markdown("#### Live Evaluation Metrics")
                    mc1, mc2, mc3, mc4 = st.columns(4)

                    mc1.metric("Accuracy", f"{m['accuracy']:.4f}")
                    mc2.metric("Recall", f"{m['recall']:.4f}")
                    mc3.metric("F1", f"{m['f1']:.4f}")
                    mc4.metric("Cost", f"{m['cost']:,}")

                # Mini donut chart for this model
                fig_mini = go.Figure(data=[go.Pie(
                    labels=['APS Failure', 'Other'],
                    values=[failures, onthers],
                    hole=0.5,
                    marker_colors=['#EF553B', '#636EFA'],
                    textinfo='label+percent',
                    textposition='auto'
                )])

                fig_mini.update_layout(
                    title=f"{model_name} - Prediction Distribution",
                    height=400,
                    showlegend=True
                )

                st.plotly_chart(fig_mini, use_container_width=True)

                display_df = display_prediction_labels(result["prediction_df"])
                st.dataframe(display_df, use_container_width=True)

                csv = display_df.to_csv(index=False)

                st.download_button(
                    f"⬇️ Download {model_name} Results",
                    csv,
                    file_name=f"aps_{model_name.lower().replace(' ', '_')}.csv",
                    mime="text/csv"
                )

# ==================================================
# MODE B — SINGLE MODEL
# ==================================================
else:
    available_models = list_model_labels()

    if not available_models:
        st.error("No trained models found.")
        st.stop()

    selected_model = st.selectbox("Choose model:", available_models)

    if st.button("▶️ Run Prediction", type="primary"):
        with st.spinner(f"Running {selected_model}..."):
            try:
                result = predict(df, selected_model, true_labels)

                st.session_state.evaluation_results = {
                    selected_model: result
                }

                st.session_state.selected_model = selected_model

                st.success("✅ Prediction complete.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

    eval_results = st.session_state.evaluation_results

    if (
        eval_results
        and selected_model in eval_results
        and st.session_state.selected_model == selected_model
    ):
        result = eval_results[selected_model]
        y_pred = result["y_pred"]

        failures = int((y_pred == 1).sum())
        onthers = int((y_pred == 0).sum())

        c1, c2, c3 = st.columns(3)

        c1.metric("Total Samples", len(y_pred))
        c2.metric("APS Failures 🔴", failures)
        c3.metric("Other Failures 🟢", onthers)

        if "metrics" in result:
            st.markdown("### Live Evaluation Metrics")

            m = result["metrics"]

            mc1, mc2, mc3, mc4 = st.columns(4)

            mc1.metric("Accuracy", f"{m['accuracy']:.4f}")
            mc2.metric("Recall", f"{m['recall']:.4f}")
            mc3.metric("F1", f"{m['f1']:.4f}")
            mc4.metric("Cost", f"{m['cost']:,}")

        # Donut chart - Plotly version
        fig_donut = go.Figure(data=[go.Pie(
            labels=["APS Failure", "Other Failures"],
            values=[failures, onthers],
            hole=0.6,
            marker_colors=['#EF553B', '#636EFA'],
            textinfo='label+percent',
            textposition='auto',
            pull=[0.05, 0]  # Slightly pull the failure slice
        )])

        fig_donut.update_layout(
            title=dict(
                text=f"{selected_model} — Prediction Distribution",
                font=dict(size=16, weight='bold')
            ),
            height=450,
            annotations=[
                dict(
                    text=f"Total: {len(y_pred):,}",
                    x=0.5,
                    y=0.5,
                    font_size=14,
                    showarrow=False
                )
            ]
        )

        st.plotly_chart(fig_donut, use_container_width=True)

        # Additional metric gauge chart
        if "metrics" in result:
            st.markdown("### 📊 Performance Metrics Visualization")
            
            metrics_data = {
                'Metric': ['Accuracy', 'Recall', 'F1 Score'],
                'Value': [m['accuracy'], m['recall'], m['f1']]
            }
            metrics_df = pd.DataFrame(metrics_data)
            
            fig_gauge = go.Figure()
            
            # Add bar chart for metrics
            fig_gauge.add_trace(go.Bar(
                x=metrics_df['Metric'],
                y=metrics_df['Value'],
                text=metrics_df['Value'].apply(lambda x: f'{x:.3f}'),
                textposition='auto',
                marker_color=['#00B4D8', '#48CAE4', '#90E0EF'],
                opacity=0.8
            ))
            
            fig_gauge.update_layout(
                title="Model Performance Metrics",
                yaxis=dict(
                    title="Score",
                    range=[0, 1],
                    gridcolor='lightgray'
                ),
                xaxis=dict(
                    title="Metric",
                    tickfont=dict(size=12)
                ),
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_gauge, use_container_width=True)

        display_df = display_prediction_labels(result["prediction_df"])
        st.dataframe(display_df, use_container_width=True)

        csv = display_df.to_csv(index=False)

        st.download_button(
            "⬇️ Download Results CSV",
            csv,
            file_name=f"aps_{selected_model.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )