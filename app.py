# ecommerce_eda.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="🛒 E-commerce EDA", layout="wide")
st.title("🛒 E-commerce Exploratory Data Analysis")

# Upload file
uploaded_file = st.file_uploader("Upload your E-commerce CSV", type=["csv"])

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)

    # Convert date
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

    # Feature engineering
    df['revenue'] = df['quantity'] * df['price'] * (1 - df['discount'])
    df['day'] = df['order_date'].dt.date
    df['month'] = df['order_date'].dt.to_period("M").astype(str)
    df['hour'] = df['order_date'].dt.hour
    df['dayofweek'] = df['order_date'].dt.day_name()

    # -------------------------
    # 📌 Key Metrics
    # -------------------------
    st.header("📌 Business KPIs")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"${df['revenue'].sum():,.0f}")
    col2.metric("Total Orders", df['order_id'].nunique())
    col3.metric("Unique Customers", df['customer_id'].nunique())
    col4.metric("Avg Order Value", f"${df['revenue'].mean():.2f}")

    # -------------------------
    # ⏳ Sales Trends
    # -------------------------
    st.header("⏳ Sales Trends")
    daily = df.groupby('day')['revenue'].sum()
    monthly = df.groupby('month')['revenue'].sum()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Daily Revenue Trend")
        st.line_chart(daily)
    with col2:
        st.subheader("Monthly Revenue Trend")
        st.line_chart(monthly)

    # -------------------------
    # 🛍️ Product Insights
    # -------------------------
   st.header("🛍️ Product Insights")

st.subheader("Top 10 Products by Revenue")
top_products = (
    df.groupby("product_id")['revenue']
    .sum()
    .nlargest(10)                   # get top 10 products
    .sort_values(ascending=True)    # sort ascending so bar chart shows nicely
)
st.subheader("Category Revenue Share (%)")
category_revenue = df.groupby("category")['revenue'].sum()
share = (category_revenue / category_revenue.sum() * 100).sort_values(ascending=False)

fig, ax = plt.subplots()
sns.barplot(x=share.index, y=share.values, ax=ax, palette="Blues_d")
ax.set_ylabel("Revenue Share (%)")
for i, val in enumerate(share.values):
    ax.text(i, val + 0.3, f"{val:.1f}%", ha="center")
st.pyplot(fig)



    # -------------------------
    # 👤 Customer Insights
    # -------------------------
    st.header("👤 Customer Insights")
    orders_per_customer = df.groupby("customer_id")['order_id'].nunique()
    revenue_per_customer = df.groupby("customer_id")['revenue'].sum()

    st.subheader("New vs Repeat Customers")
    repeat = (orders_per_customer > 1).sum()
    new = (orders_per_customer == 1).sum()
    st.bar_chart({"New Customers": new, "Repeat Customers": repeat})

    st.subheader("Customer Lifetime Revenue Distribution")
    fig, ax = plt.subplots()
    sns.histplot(revenue_per_customer, bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Top 10 Customers by Revenue")
    st.dataframe(revenue_per_customer.sort_values(ascending=False).head(10))

    # -------------------------
    # 📊 Pricing & Discounts
    # -------------------------
    st.header("📊 Pricing & Discounts")
    st.subheader("Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['price'], bins=40, kde=True, ax=ax)
    st.pyplot(fig)

    
    st.subheader("Discount Impact on Revenue")
    fig, ax = plt.subplots()
    sns.boxplot(x=pd.qcut(df['discount'], 5, duplicates='drop'), y=df['revenue'], ax=ax)
    ax.set_xlabel("Discount Quintiles")
    st.pyplot(fig)

    # -------------------------
    # 🌍 Regional & Payment Analysis
    # -------------------------
    st.header("🌍 Regional & Payment Insights")
    st.subheader("Revenue by Region")
    st.bar_chart(df.groupby("region")['revenue'].sum())

    st.subheader("Orders by Payment Method")
    st.bar_chart(df['payment_method'].value_counts())

    # -------------------------
    # 📅 Seasonality
    # -------------------------
    st.header("📅 Seasonality Patterns")
    st.subheader("Sales by Day of Week")
    st.bar_chart(df['dayofweek'].value_counts())

    st.subheader("Hourly Sales Pattern")
    hourly_sales = df.groupby('hour')['revenue'].sum()
    st.bar_chart(hourly_sales)

    st.subheader("Weekend vs Weekday Sales")
    df['is_weekend'] = df['dayofweek'].isin(["Saturday", "Sunday"])
    weekend_sales = df.groupby('is_weekend')['revenue'].sum()
    st.bar_chart(weekend_sales)

    # -------------------------
    # 🔗 Correlations
    # -------------------------
    st.header("🔗 Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df[['price','discount','quantity','revenue']].corr(),
                annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
