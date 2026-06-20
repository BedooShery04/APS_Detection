import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.title("🔍 Exploratory Data Analysis")

# --------------------------------------------------
# DATA CHECK
# --------------------------------------------------
if st.session_state.uploaded_df is None:
    st.warning("⚠️ Please upload a dataset first (Page 1).")
    st.stop()

df = st.session_state.uploaded_df.copy()
true_labels = st.session_state.get("true_labels")

# --------------------------------------------------
# TABS
# --------------------------------------------------
tabs = st.tabs([
    "📋 Overview",
    "📊 Class Distribution",
    "❓ Missing Values",
    "📈 Feature Distributions",
    "🔗 Correlation Heatmap",
])

# ==================================================
# TAB 1 — OVERVIEW
# ==================================================
with tabs[0]:
    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    total_rows = df.shape[0]
    total_cols = df.shape[1]
    total_missing = df.isnull().sum().sum()

    c1.metric("Total Rows", f"{total_rows:,}")
    c2.metric("Feature Columns", total_cols)
    c3.metric("Total Missing Cells", f"{total_missing:,}")

    if total_rows > 0 and total_cols > 0:
        miss_pct = round((total_missing / (total_rows * total_cols)) * 100, 2)
    else:
        miss_pct = 0

    c4.metric("Overall Missing %", f"{miss_pct}%")

    st.markdown("#### Data Types")
    dtype_counts = df.dtypes.astype(str).value_counts().reset_index()
    dtype_counts.columns = ["dtype", "count"]
    st.dataframe(dtype_counts, use_container_width=True)

    st.markdown("#### Statistical Summary (numeric columns)")

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.empty:
        st.info("No numeric columns available.")
    else:
        st.dataframe(
            numeric_df.describe().T.style.format("{:.4f}"),
            use_container_width=True
        )

# ==================================================
# TAB 2 — CLASS DISTRIBUTION
# ==================================================
with tabs[1]:
    st.subheader("Class Distribution")

    if true_labels is None:
        st.info(
            "No labels available. This uploaded dataset is prediction-only."
        )

    else:
        vc = true_labels.value_counts()
        vc_pct = true_labels.value_counts(normalize=True) * 100

        onther_count = int(vc.get(0, 0))
        failure_count = int(vc.get(1, 0))

        onther_pct = vc_pct.get(0, 0)
        failure_pct = vc_pct.get(1, 0)

        c1, c2 = st.columns(2)

        c1.metric(
            "Onther (0)",
            f"{onther_count:,}",
            f"{onther_pct:.1f}%"
        )

        c2.metric(
            "APS Failure (1)",
            f"{failure_count:,}",
            f"{failure_pct:.1f}%"
        )

        total_labels = onther_count + failure_count

        if total_labels > 0:
            # Create subplot with bar chart and pie chart
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Class Counts", "Class Proportions"),
                specs=[[{"type": "bar"}, {"type": "pie"}]]
            )

            # Add bar chart
            fig.add_trace(
                go.Bar(
                    x=["Onther", "APS Failure"],
                    y=[onther_count, failure_count],
                    text=[f"{onther_count:,}", f"{failure_count:,}"],
                    textposition="outside",
                    marker_color=["#2196F3", "#F44336"],
                    name="Count",
                    showlegend=False
                ),
                row=1, col=1
            )

            # Add pie chart
            fig.add_trace(
                go.Pie(
                    labels=["Onther", "APS Failure"],
                    values=[onther_count, failure_count],
                    marker_colors=["#2196F3", "#F44336"],
                    hole=0.3,
                    rotation=90,
                    textinfo="percent+label",
                    showlegend=False
                ),
                row=1, col=2
            )

            fig.update_layout(
                height=500,
                showlegend=False,
                title_text="Class Distribution Analysis",
                title_font_size=16
            )

            fig.update_xaxes(title_text="Class Type", row=1, col=1)
            fig.update_yaxes(title_text="Count", row=1, col=1)

            st.plotly_chart(fig, use_container_width=True)

            ratio = onther_count / max(failure_count, 1)

            st.info(
                f"⚖️ Class imbalance ratio = **{ratio:.1f}:1** "
                "(Onther : APS Failure)\n\n"
                "SMOTE was used during training to address imbalance."
            )

        else:
            st.warning("Labels exist but contain no valid values.")

# ==================================================
# TAB 3 — MISSING VALUES
# ==================================================
with tabs[2]:
    st.subheader("Missing Values Analysis")

    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)

    miss_df = pd.DataFrame({
        "Missing Count": missing_counts,
        "Missing %": missing_pct
    }).sort_values("Missing %", ascending=False)

    cols_with_missing = miss_df[miss_df["Missing Count"] > 0]
    cols_no_missing = miss_df[miss_df["Missing Count"] == 0]

    c1, c2 = st.columns(2)

    c1.metric("Columns WITH missing values", len(cols_with_missing))
    c2.metric("Complete columns", len(cols_no_missing))

    if not cols_with_missing.empty:
        st.markdown("#### Top 30 columns by missing %")

        top30 = cols_with_missing.head(30)

        # Create color mapping based on missing percentage
        colors = []
        for v in top30["Missing %"]:
            if v > 60:
                colors.append("#F44336")  # Red
            elif v > 20:
                colors.append("#FF9800")  # Orange
            else:
                colors.append("#2196F3")  # Blue

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=top30["Missing %"][::-1],
            y=top30.index[::-1],
            orientation='h',
            marker_color=colors[::-1],
            text=top30["Missing %"][::-1].round(1),
            textposition="outside",
            texttemplate="%{text}%",
            name="Missing %"
        ))

        # Add threshold line
        fig.add_vline(
            x=60, line_dash="dash", line_color="red",
            annotation_text="60% threshold (drop)",
            annotation_position="top right"
        )

        fig.update_layout(
            title="Missing Values per Column (Top 30)",
            xaxis_title="Missing Percentage (%)",
            yaxis_title="Features",
            height=600,
            showlegend=False,
            hovermode="y"
        )

        st.plotly_chart(fig, use_container_width=True)

        high_miss = cols_with_missing[
            cols_with_missing["Missing %"] > 60
        ]

        if not high_miss.empty:
            st.warning(
                f"🗑️ {len(high_miss)} columns exceed 60% missing "
                "and were dropped during preprocessing."
            )

        st.markdown("#### Full Missing Value Table")
        
        # Create heatmap style table
        fig_table = go.Figure(data=[go.Table(
            header=dict(
                values=list(cols_with_missing.columns),
                fill_color="#0f3460",
                font=dict(color="white", size=12),
                align="center"
            ),
            cells=dict(
                values=[
                    cols_with_missing.index,
                    cols_with_missing["Missing Count"],
                    cols_with_missing["Missing %"]
                ],
                fill_color=[
                    ["#F44336" if x > 60 else "#FF9800" if x > 20 else "#2196F3" 
                     for x in cols_with_missing["Missing %"]]
                ],
                align="center",
                font=dict(size=11),
                format=[None, ",d", ".1f"]
            )
        )])

        fig_table.update_layout(height=400)
        st.plotly_chart(fig_table, use_container_width=True)

    else:
        st.success("✅ No missing values found.")

# ==================================================
# TAB 4 — FEATURE DISTRIBUTIONS
# ==================================================
with tabs[3]:
    st.subheader("Feature Distributions")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        st.info("No numeric columns found.")

    else:
        selected_cols = st.multiselect(
            "Select features to visualize (max 12):",
            options=numeric_cols,
            default=numeric_cols[:min(6, len(numeric_cols))],
            max_selections=12
        )

        if selected_cols:
            n = len(selected_cols)
            ncols = 3
            nrows = (n + ncols - 1) // ncols

            # Create subplot
            fig = make_subplots(
                rows=nrows, cols=ncols,
                subplot_titles=[f"<b>{col}</b>" for col in selected_cols],
                horizontal_spacing=0.08,
                vertical_spacing=0.12
            )

            for i, col in enumerate(selected_cols):
                row = i // ncols + 1
                col_pos = i % ncols + 1
                
                data = df[col].dropna()
                
                # Add histogram
                fig.add_trace(
                    go.Histogram(
                        x=data,
                        nbinsx=40,
                        marker_color="#0f3460",
                        marker_line_color="white",
                        marker_line_width=0.5,
                        opacity=0.85,
                        name=col,
                        showlegend=False
                    ),
                    row=row, col=col_pos
                )
                
                # Add mean line
                mean_val = data.mean()
                fig.add_vline(
                    x=mean_val,
                    line_dash="dash",
                    line_color="#F44336",
                    line_width=1.5,
                    row=row, col=col_pos,
                    annotation_text=f"μ={mean_val:.2f}",
                    annotation_position="top"
                )
                
                # Update axes labels
                fig.update_xaxes(title_text="Value", row=row, col=col_pos)
                fig.update_yaxes(title_text="Frequency", row=row, col=col_pos)

            # Update layout
            fig.update_layout(
                height=350 * nrows,
                showlegend=False,
                title_text="Feature Distribution Analysis",
                title_font_size=16,
                bargap=0.05
            )

            st.plotly_chart(fig, use_container_width=True)

# ==================================================
# TAB 5 — CORRELATION HEATMAP (Lower Triangle Only)
# ==================================================
with tabs[4]:
    st.subheader("Correlation Heatmap")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) < 2:
        st.info("Not enough numeric columns for correlation analysis.")

    else:
        max_cols = st.slider(
            "Number of columns:",
            min_value=5,
            max_value=min(60, len(numeric_cols)),
            value=min(15, len(numeric_cols)),
            step=5
        )

        top_var_cols = (
            df[numeric_cols]
            .var()
            .sort_values(ascending=False)
            .head(max_cols)
            .index.tolist()
        )

        corr = df[top_var_cols].corr()
        
        # Create a mask for upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool), k=0)
        
        # Create a copy for the heatmap
        corr_masked = corr.copy()
        corr_masked.values[mask] = None
        
        # Create text matrix with rounded values (only for upper triangle)
        text_matrix = corr_masked.copy()
        for i in range(len(text_matrix)):
            for j in range(len(text_matrix)):
                if text_matrix.iloc[i, j] is not None:
                    text_matrix.iloc[i, j] = f"{text_matrix.iloc[i, j]:.2f}"
                else:
                    text_matrix.iloc[i, j] = ""

        # Create interactive heatmap with masked upper triangle
        fig = go.Figure(data=go.Heatmap(
            z=corr_masked.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmid=0,
            text=text_matrix.values,
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            hoverinfo="z+x+y",
            showscale=True,
            colorbar=dict(
                title="Correlation",
                thickness=20,
                len=0.8
            )
        ))

        # Update layout for better readability
        fig.update_layout(
            title=f"Correlation Matrix - Upper Triangle (Top {max_cols} Features)",
            xaxis_title="Features",
            yaxis_title="Features",
            xaxis=dict(
                tickangle=45, 
                tickfont=dict(size=10),
                side="bottom",
                ticks="outside"
            ),
            yaxis=dict(
                tickfont=dict(size=10),
                ticks="outside"
            ),
            height=800,
            width=1000,
            hovermode="closest",
            margin=dict(l=100, r=50, t=100, b=100)
        )

        st.plotly_chart(fig, use_container_width=True)
        
        # Add correlation insights
        st.markdown("#### 💡 Correlation Insights")
        
        # Find highest correlations (excluding diagonal)
        corr_values = corr.values
        np.fill_diagonal(corr_values, 0)
        
        # Get upper triangle indices for finding max correlation
        upper_tri_indices = np.triu_indices_from(corr_values, k=1)
        upper_tri_values = corr_values[upper_tri_indices]
        
        if len(upper_tri_values) > 0:
            max_corr_idx_flat = np.argmax(np.abs(upper_tri_values))
            max_corr_pos = (upper_tri_indices[0][max_corr_idx_flat], 
                          upper_tri_indices[1][max_corr_idx_flat])
            
            if corr.values[max_corr_pos] > 0:
                st.info(
                    f"🔗 **Strongest positive correlation**: between "
                    f"**{corr.columns[max_corr_pos[0]]}** and "
                    f"**{corr.columns[max_corr_pos[1]]}** "
                    f"({corr.values[max_corr_pos]:.3f})"
                )
            
            # Find strongest negative correlation
            min_corr_idx_flat = np.argmin(upper_tri_values)
            min_corr_pos = (upper_tri_indices[0][min_corr_idx_flat], 
                          upper_tri_indices[1][min_corr_idx_flat])
            
            if corr.values[min_corr_pos] < 0:
                st.info(
                    f"🔗 **Strongest negative correlation**: between "
                    f"**{corr.columns[min_corr_pos[0]]}** and "
                    f"**{corr.columns[min_corr_pos[1]]}** "
                    f"({corr.values[min_corr_pos]:.3f})"
                )