"""Streamlit interface for the Smart Data Analyzer demo."""
import pandas as pd
import streamlit as st
from sklearn.datasets import load_wine

from model import train_and_evaluate

st.set_page_config(page_title="Smart Data Analyzer", page_icon="📊", layout="wide")
st.title("📊 Smart Data Analyzer")
st.write(
    "Explore a built-in dataset and see how a machine-learning classifier "
    "performs on held-out data."
)
st.info("Educational demo only — not intended for high-stakes decisions.")

with st.sidebar:
    st.header("Model settings")
    n_estimators = st.slider(
        "Number of trees", min_value=10, max_value=300, value=100, step=10
    )
    test_size = st.slider(
        "Test set size", min_value=0.10, max_value=0.50, value=0.25, step=0.05
    )

data = load_wine(as_frame=True)
df = data.frame
left, right = st.columns(2)

with left:
    st.subheader("Dataset preview")
    st.dataframe(df.head(10), use_container_width=True)

with right:
    st.subheader("Class distribution")
    st.bar_chart(df["target"].value_counts().sort_index())

if st.button("Train and evaluate model", type="primary"):
    with st.spinner("Training model..."):
        result = train_and_evaluate(
            n_estimators=n_estimators,
            test_size=test_size,
        )

    st.subheader("Evaluation on held-out test data")
    metric_col, details_col = st.columns([1, 2])

    with metric_col:
        st.metric("Accuracy", f"{result['accuracy']:.1%}")

    with details_col:
        report = pd.DataFrame(result["report"]).T
        st.dataframe(report.round(3), use_container_width=True)

    st.subheader("Top feature importances")
    importance = (
        pd.Series(
            result["model"].feature_importances_,
            index=result["feature_names"],
        )
        .sort_values(ascending=False)
        .head(10)
    )
    st.bar_chart(importance)
else:
    st.caption("Choose settings in the sidebar, then click **Train and evaluate model**.")
