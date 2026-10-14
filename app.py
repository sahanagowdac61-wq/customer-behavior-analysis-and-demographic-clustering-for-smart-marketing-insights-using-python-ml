import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="wide")

st.title("🛍️ Customer Segmentation Dashboard")
st.write("Explore customer groups using purchase behavior and demographics. Upload your own CSV or try the included sample dataset.")

@st.cache_data
def load_sample():
    return pd.read_csv("data/customers.csv")

with st.sidebar:
    st.header("Project settings")
    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    st.caption("Recommended columns: Age, Annual Income (k$), Spending Score (1-100).")
    use_demo = st.checkbox("Use included sample data", value=True if uploaded is None else False)

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read this CSV: {exc}")
        st.stop()
elif use_demo:
    df = load_sample()
else:
    st.info("Upload a CSV or select 'Use included sample data' in the sidebar.")
    st.stop()

st.subheader("1. Preview the data")
st.dataframe(df.head(10), use_container_width=True)
st.write(f"**Rows:** {len(df)} · **Columns:** {len(df.columns)}")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols) < 2:
    st.error("Your CSV needs at least two numeric columns to perform clustering.")
    st.stop()

default_features = [c for c in ["Age", "Annual Income (k$)", "Spending Score (1-100)"] if c in numeric_cols]
if len(default_features) < 2:
    default_features = numeric_cols[:min(3, len(numeric_cols))]

st.subheader("2. Choose features and number of groups")
left, right = st.columns([2, 1])
with left:
    features = st.multiselect("Numeric features used to group customers", numeric_cols, default=default_features)
with right:
    k = st.slider("Number of customer segments (K)", min_value=2, max_value=8, value=5)

if len(features) < 2:
    st.warning("Select at least two numeric features.")
    st.stop()

working = df[features].copy()
for col in features:
    working[col] = pd.to_numeric(working[col], errors="coerce")
valid_mask = working.notna().all(axis=1)
clean = working.loc[valid_mask].copy()

if len(clean) <= k:
    st.error(f"At least {k + 1} complete rows are needed for {k} clusters.")
    st.stop()

scaler = StandardScaler()
scaled = scaler.fit_transform(clean)
model = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = model.fit_predict(scaled)

result = df.loc[clean.index].copy()
result["Segment"] = labels + 1
centers = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=features)
centers["Segment"] = np.arange(1, k + 1)

st.subheader("3. Results")
c1, c2, c3 = st.columns(3)
c1.metric("Customers analyzed", f"{len(result):,}")
c2.metric("Segments found", k)
try:
    score = silhouette_score(scaled, labels) if len(set(labels)) > 1 and len(clean) > k else float("nan")
    c3.metric("Silhouette score", f"{score:.2f}" if np.isfinite(score) else "N/A")
except Exception:
    c3.metric("Silhouette score", "N/A")

counts = result["Segment"].value_counts().sort_index().rename_axis("Segment").reset_index(name="Customers")
fig_counts = px.bar(counts, x="Segment", y="Customers", color="Segment",
                    title="Number of customers in each segment", text="Customers")
st.plotly_chart(fig_counts, use_container_width=True)

if len(features) >= 2:
    x_feature = st.selectbox("X-axis feature", features, index=0)
    y_feature = st.selectbox("Y-axis feature", features, index=1)
    plot_df = result.copy()
    plot_df["Segment"] = plot_df["Segment"].astype(str)
    fig_scatter = px.scatter(plot_df, x=x_feature, y=y_feature, color="Segment",
                             hover_data=[c for c in result.columns if c not in [x_feature, y_feature, "Segment"]],
                             title=f"Customer groups: {x_feature} vs {y_feature}")
    st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("4. Understand each segment")
profile = result.groupby("Segment")[features].mean().round(2)
profile.insert(0, "Customer count", result.groupby("Segment").size())
st.dataframe(profile, use_container_width=True)

st.markdown("""
**How to interpret:** compare the average feature values for each segment. For example, a segment with high income and high spending score may be a group worth exploring for premium offers. Treat these patterns as clues, not as automatic business decisions.
""")

st.subheader("5. Download results")
csv = result.to_csv(index=False).encode("utf-8")
st.download_button("Download customers with segment labels (CSV)", data=csv,
                   file_name="customer_segments.csv", mime="text/csv")
st.caption("Learning project: K-Means is an exploratory method. Review data quality and business context before using segments in real campaigns.")
