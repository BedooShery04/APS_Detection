import streamlit as st
import pandas as pd

st.title("📂 Upload Dataset")
st.markdown(
    "Upload the APS sensor CSV file (training or test set from the Scania dataset)."
)

uploaded_file = st.file_uploader(
    "Choose an APS CSV file",
    type=["csv"],
    help="The file should match the APS Failure at Scania Trucks dataset format (with 20 header rows)."
)

# --------------------------------------------------
# FILE UPLOAD HANDLING
# --------------------------------------------------
if uploaded_file is not None:
    with st.spinner("Reading file..."):
        try:
            # APS dataset parsing
            df = pd.read_csv(
                uploaded_file,
                skiprows=20,
                na_values="na"
            )

            # --------------------------
            # LABEL HANDLING
            # --------------------------
            if "class" in df.columns:
                raw_labels = df["class"].copy()

                valid_labels = {"neg", "pos"}
                unique_labels = set(raw_labels.dropna().unique())

                if not unique_labels.issubset(valid_labels):
                    st.error(
                        f"Invalid label values detected: {unique_labels}. "
                        "Expected APS labels: {'neg', 'pos'}"
                    )
                    st.stop()

                # Numeric mapping
                label_mapping = {
                    "neg": 0,   # Other Failure 
                    "pos": 1    # APS Failure
                }

                st.session_state.true_labels = raw_labels.map(label_mapping)
                df_features = df.drop(columns=["class"])

            else:
                st.session_state.true_labels = None
                df_features = df

            # --------------------------
            # STORE DATA
            # --------------------------
            st.session_state.uploaded_df = df_features

            # Reset evaluation state
            st.session_state.evaluation_results = {}
            st.session_state.selected_model = None

        except Exception as e:
            st.error(f"Failed to read file: {e}")
            st.stop()

    st.success("✅ Dataset uploaded successfully!")

    # --------------------------------------------------
    # DATASET SUMMARY
    # --------------------------------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", f"{df_features.shape[0]:,}")
    col2.metric("Feature Columns", df_features.shape[1])

    has_labels = st.session_state.true_labels is not None
    col3.metric("Labels found", "Yes ✅" if has_labels else "No ❌")

    if has_labels:
        st.info(
            "🏷️ APS labels detected and converted to numeric format "
            "(0 = Normal, 1 = APS Failure) for live evaluation."
        )

    # --------------------------------------------------
    # DATA PREVIEW
    # --------------------------------------------------
    st.markdown("### Preview (first 10 rows)")
    st.dataframe(df_features.head(10), use_container_width=True)

    # --------------------------------------------------
    # MISSING VALUES SUMMARY
    # --------------------------------------------------
    st.markdown("### Missing Values per Column")

    missing = df_features.isnull().sum()
    missing_pct = (missing / len(df_features) * 100).round(2)

    miss_df = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_pct
    })

    miss_df = miss_df.query("`Missing Count` > 0")
    miss_df = miss_df.sort_values("Missing %", ascending=False)

    if miss_df.empty:
        st.success("No missing values found.")
    else:
        st.warning(f"{len(miss_df)} columns contain missing values.")
        st.dataframe(miss_df.head(20), use_container_width=True)

# --------------------------------------------------
# EXISTING DATASET ALREADY LOADED
# --------------------------------------------------
elif st.session_state.uploaded_df is not None:
    st.info("A dataset is already loaded. Upload a new file to replace it.")

    df_features = st.session_state.uploaded_df

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", f"{df_features.shape[0]:,}")
    col2.metric("Feature Columns", df_features.shape[1])

    has_labels = st.session_state.true_labels is not None
    col3.metric("Labels found", "Yes ✅" if has_labels else "No ❌")

    st.dataframe(df_features.head(5), use_container_width=True)

# --------------------------------------------------
# NO DATASET
# --------------------------------------------------
else:
    st.info("👆 Please upload a CSV file to get started.")