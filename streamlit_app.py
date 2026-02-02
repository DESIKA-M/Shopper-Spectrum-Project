# optimized_streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Retail Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛒 Retail Analytics Dashboard")
st.markdown(
    """
    This dashboard provides:
    - **Product Recommendations** using Collaborative Filtering
    - **Customer Segmentation** using RFM Analysis and KMeans Clustering
    - **Customer Purchase Analysis** with Visualizations
    """
)

# ---------------------------
# Load Models & Data (Cached)
# ---------------------------
@st.cache_resource
def load_models_data():
    kmeans_model = joblib.load("rfm_kmeans_model.pkl")
    scaler = joblib.load("rfm_scaler.pkl")
    df = pd.read_csv("fin_online_retail_cleaned.csv")
    return kmeans_model, scaler, df

kmeans_model, scaler, df = load_models_data()

# ---------------------------
# Product Similarity Matrix (Cached)
# ---------------------------
@st.cache_resource
def get_product_similarity(df):
    customer_product_matrix = pd.pivot_table(
        df,
        index="CustomerID",
        columns="Description",
        values="Quantity",
        aggfunc="sum",
        fill_value=0
    )

    product_customer_matrix = customer_product_matrix.T
    similarity = cosine_similarity(product_customer_matrix)

    similarity_df = pd.DataFrame(
        similarity,
        index=product_customer_matrix.index,
        columns=product_customer_matrix.index
    )

    product_index_map = {p.lower(): p for p in similarity_df.index}
    return similarity_df, product_index_map

product_similarity_df, product_index_map = get_product_similarity(df)

# ---------------------------
# Product Recommendation Logic
# ---------------------------
def recommend_products(product_name, top_n=5):
    name = product_name.strip().lower()
    if name not in product_index_map:
        return None

    actual_name = product_index_map[name]
    scores = product_similarity_df[actual_name]

    return (
        scores.sort_values(ascending=False)
        .iloc[1:top_n + 1]
        .index
        .tolist()
    )

# ---------------------------
# Sidebar Navigation
# ---------------------------
menu = ["Product Recommendation", "Customer Segmentation","Customer Purchase Analysis"]
choice = st.sidebar.selectbox("Select Module", menu)

# =========================================================
# PRODUCT RECOMMENDATION MODULE
# =========================================================
if choice == "Product Recommendation":
    st.header("🔹 Product Recommendation")

    product_name_input = st.text_input("Enter Product Name")

    if st.button("Get Recommendations"):
        recommendations = recommend_products(product_name_input)

        if recommendations is None:
            st.warning("Product not found.")
        else:
            st.success("Top 5 Similar Products")
            cols = st.columns(5)
            for i, prod in enumerate(recommendations):
                with cols[i]:
                    st.markdown(f"**{prod}**")

# =========================================================
# CUSTOMER SEGMENTATION MODULE
# =========================================================
# =========================================================
# CUSTOMER PURCHASE ANALYSIS MODULE
# =========================================================
if choice == "Customer Purchase Analysis":
    st.header("🔹 Customer Purchase Analysis")
    st.markdown("Enter **Customer ID** to see their frequent purchases and visualization.")

    customer_id_input = st.text_input("Enter Customer ID")

    if st.button("Show Purchases"):
        if customer_id_input.isdigit():
            customer_id = int(customer_id_input)

            # Filter data for this customer
            customer_data = df[df["CustomerID"] == customer_id]

            if customer_data.empty:
                st.warning("Customer ID not found.")
            else:
                # Group by product and sum quantity
                purchase_summary = (
                    customer_data.groupby("Description")["Quantity"]
                    .sum()
                    .sort_values(ascending=False)
                )

                st.subheader(f"📦 Products Bought by Customer {customer_id}")
                st.dataframe(purchase_summary.reset_index().rename(columns={"Quantity":"Total Quantity"}), use_container_width=True)

                # ---------------------------
                # Visualization
                # ---------------------------
                st.subheader("📊 Top Products Purchased")
                top_n = st.slider("Select number of top products to display", min_value=3, max_value=20, value=10)
                st.bar_chart(purchase_summary.head(top_n))

        else:
            st.error("Please enter a valid numeric Customer ID.")

if choice == "Customer Segmentation":
    st.header("🔹 Customer Segmentation (RFM + KMeans)")
    st.markdown("Enter **RFM values** to understand cluster assignment.")

    with st.form("rfm_form"):
        recency = st.number_input("Recency (days)", min_value=0)
        frequency = st.number_input("Frequency", min_value=0)
        monetary = st.number_input("Monetary Value", min_value=0.0, format="%.2f")
        submit = st.form_submit_button("Predict Cluster")

    if submit:
        # ---------------------------
        # Scale Input
        # ---------------------------
        rfm_input = np.array([[recency, frequency, monetary]])
        rfm_scaled = scaler.transform(rfm_input)

        # ---------------------------
        # Predict Cluster
        # ---------------------------
        cluster_label = kmeans_model.predict(rfm_scaled)[0]

        cluster_mapping = {
            0: "Occasional",
            1: "At-Risk",
            2: "High-Value",
            3: "Regular"
        }

        segment = cluster_mapping.get(cluster_label, "Unknown")

        st.success(f"Predicted Customer Segment: **{segment}**")

        # ---------------------------
        # Distance to Each Cluster
        # ---------------------------
        centroids = kmeans_model.cluster_centers_

        distances = np.linalg.norm(
            rfm_scaled - centroids,
            axis=1
        )

        distance_df = pd.DataFrame({
            "Cluster ID": range(len(distances)),
            "Distance": distances
        }).sort_values("Distance")

        st.subheader("📏 Distance from Each Cluster")
        st.dataframe(distance_df, use_container_width=True)

        # ---------------------------
        # Confidence Score
        # ---------------------------
        closest = distance_df.iloc[0]["Distance"]
        second_closest = distance_df.iloc[1]["Distance"]

        confidence = round((1 - closest / second_closest) * 100, 2)
        st.metric("Prediction Confidence", f"{confidence}%")

        # ---------------------------
        # Distance Visualization
        # ---------------------------
        st.subheader("📊 Cluster Distance Visualization")
        st.bar_chart(distance_df.set_index("Cluster ID"))

        # ---------------------------
        # Cluster Centroid Table
        # ---------------------------
        st.subheader("📍 Cluster Centroids (Scaled RFM)")
        centroid_df = pd.DataFrame(
            centroids,
            columns=["Recency", "Frequency", "Monetary"]
        )
        st.dataframe(centroid_df, use_container_width=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("Developed by **Desika**")
