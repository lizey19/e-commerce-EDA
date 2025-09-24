import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🛒 E-commerce Data Cleaning & EDA")

# Step 1: Upload CSV
uploaded_file = st.file_uploader("Upload your e-commerce CSV", type=["csv"])

if uploaded_file:
    # Step 2: Load dataset
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # Step 3: Convert data types
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    df['time'] = pd.to_datetime(df['time'], errors='coerce').dt.time

    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['discount'] = pd.to_numeric(df['discount'], errors='coerce')
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')

    # Step 4: Handle missing values
    df = df.dropna(subset=['order_id', 'product_id', 'customer_id'])
    df['quantity'] = df['quantity'].fillna(df['quantity'].median())

    # Step 5: Remove duplicates
    df = df.drop_duplicates(subset=['order_id', 'product_id', 'customer_id'])

    # Step 6: Handle invalid entries
    df = df[(df['price'] > 0) & (df['quantity'] > 0)]
    df = df[df['discount'] <= df['price']]

    # Step 7: Create combined datetime
    df['order_datetime'] = pd.to_datetime(
        df['order_date'].astype(str) + " " + df['time'].astype(str),
        errors='coerce'
    )

    # -------------------------------
    # ⚙️ Feature Engineering
    # -------------------------------
    st.header("⚙️ Feature Engineering")

    df['revenue'] = df['quantity'] * (df['price'] - df['discount'])

    if not df['order_datetime'].isnull().all():
        df['day'] = df['order_datetime'].dt.date
        df['week'] = df['order_datetime'].dt.isocalendar().week
        df['month'] = df['order_datetime'].dt.to_period("M").astype(str)
        df['hour'] = df['order_datetime'].dt.hour
        df['dayofweek'] = df['order_datetime'].dt.day_name()

    st.subheader("Engineered Features Preview")
    st.dataframe(df[['order_id', 'revenue', 'day', 'week', 'month', 'hour', 'dayofweek']].head())

    # -------------------------------
    # 📊 Univariate Analysis
    # -------------------------------
    st.header("📊 Univariate Analysis")

    st.subheader("Total Orders")
    st.write("Total Orders:", df['order_id'].nunique())

    st.subheader("Customers Overview")
    unique_customers = df['customer_id'].nunique()
    avg_orders = df['order_id'].nunique() / unique_customers
    st.write("Unique Customers:", unique_customers)
    st.write("Average Orders per Customer:", round(avg_orders, 2))

    st.subheader("Products Overview")
    st.write("Total Products Sold:", df['product_id'].nunique())
    st.write("Bestselling Products:")
    st.dataframe(df['product_id'].value_counts().head(10))

    st.subheader("Popular Categories")
    st.bar_chart(df['category'].value_counts())

    st.subheader("Quantity Distribution")
    fig, ax = plt.subplots()
    sns.boxplot(x=df['quantity'], ax=ax)
    st.pyplot(fig)

    st.subheader("Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['price'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Discount Levels")
    st.bar_chart(df['discount'].value_counts().head(10))

    st.subheader("Orders by Region")
    st.bar_chart(df['region'].value_counts())

    st.subheader("Payment Methods")
    st.bar_chart(df['payment_method'].value_counts())

    st.subheader("Orders Over Time")
    if not df['order_datetime'].isnull().all():
        st.line_chart(df['day'].value_counts().sort_index())
        st.bar_chart(df['week'].value_counts().sort_index())
        st.bar_chart(df['hour'].value_counts().sort_index())

    # -------------------------------
    # 🔗 Bivariate Analysis
    # -------------------------------
    st.header("🔗 Bivariate Analysis")

    st.subheader("Revenue by Category")
    fig, ax = plt.subplots()
    sns.barplot(x="category", y="revenue", data=df, estimator=sum, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

    st.subheader("Revenue by Region")
    fig, ax = plt.subplots()
    sns.barplot(x="region", y="revenue", data=df, estimator=sum, ax=ax)
    st.pyplot(fig)

    st.subheader("Payment Method vs Category")
    cross_tab = pd.crosstab(df['payment_method'], df['category'])
    st.dataframe(cross_tab)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(cross_tab, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.subheader("Quantity vs Price")
    fig, ax = plt.subplots()
    sns.scatterplot(x="price", y="quantity", data=df, ax=ax, alpha=0.5)
    st.pyplot(fig)

    if 'order_datetime' in df.columns:
        st.subheader("Orders by Day of Week")
        fig, ax = plt.subplots()
        sns.countplot(x="dayofweek", data=df,
                      order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], ax=ax)
        st.pyplot(fig)

    # -------------------------------
    # ⏳ Time Series Analysis
    # -------------------------------
    st.header("⏳ Time Series Analysis")

    if 'order_datetime' in df.columns:
        st.subheader("Daily Sales Trend")
        daily_sales = df.groupby('day')['revenue'].sum()
        st.line_chart(daily_sales)

        st.subheader("Weekly Sales Trend")
        weekly_sales = df.groupby('week')['revenue'].sum()
        st.line_chart(weekly_sales)

        st.subheader("Monthly Sales Trend")
        monthly_sales = df.groupby('month')['revenue'].sum()
        st.line_chart(monthly_sales)

        st.subheader("Weekend vs Weekday Sales")
        df['is_weekend'] = df['dayofweek'].isin(["Saturday", "Sunday"])
        weekend_sales = df.groupby('is_weekend')['revenue'].sum()
        st.bar_chart(weekend_sales)

        st.subheader("Peak Hours of Purchase")
        hourly_sales = df.groupby('hour')['revenue'].sum()
        st.bar_chart(hourly_sales)

    # -------------------------------
    # 👤 Customer Analysis
    # -------------------------------
    st.header("👤 Customer Analysis")

    st.subheader("Orders per Customer")
    orders_per_customer = df.groupby("customer_id")['order_id'].nunique()
    st.write("Average Orders per Customer:", round(orders_per_customer.mean(), 2))
    st.bar_chart(orders_per_customer.value_counts().sort_index())

    st.subheader("Repeat vs New Customers")
    repeat_customers = (orders_per_customer > 1).sum()
    new_customers = (orders_per_customer == 1).sum()
    st.write("New Customers:", new_customers)
    st.write("Repeat Customers:", repeat_customers)
    st.bar_chart({"New Customers": new_customers, "Repeat Customers": repeat_customers})

    st.subheader("Average Order Value per Customer")
    aov_per_customer = df.groupby("customer_id")['revenue'].sum() / orders_per_customer
    st.write("Overall Average Order Value:", round(aov_per_customer.mean(), 2))
    fig, ax = plt.subplots()
    sns.histplot(aov_per_customer, bins=30, kde=True, ax=ax)
    ax.set_title("Distribution of Average Order Value per Customer")
    st.pyplot(fig)

    # -------------------------------
    # 🛍️ Product Analysis
    # -------------------------------
    st.header("🛍️ Product Analysis")

    st.subheader("Bestselling Products by Revenue & Quantity")
    top_products_revenue = df.groupby("product_id")['revenue'].sum().sort_values(ascending=False).head(10)
    top_products_quantity = df.groupby("product_id")['quantity'].sum().sort_values(ascending=False).head(10)

    col1, col2 = st.columns(2)
    with col1:
        st.write("🔝 Top 10 Products by Revenue")
        st.bar_chart(top_products_revenue)
    with col2:
        st.write("📦 Top 10 Products by Quantity")
        st.bar_chart(top_products_quantity)

    st.subheader("Most Discounted Products")
    most_discounted = df.groupby("product_id")['discount'].mean().sort_values(ascending=False).head(10)
    st.bar_chart(most_discounted)

    st.subheader("Category Contribution to Revenue (%)")
    category_revenue = df.groupby("category")['revenue'].sum()
    category_share = (category_revenue / category_revenue.sum()) * 100
    st.write(category_share)
    st.bar_chart(category_share)

    # -------------------------------
    # 🌍 Regional Analysis
    # -------------------------------
    st.header("🌍 Regional Analysis")

    st.subheader("Revenue & Orders by Region")
    region_revenue = df.groupby("region")['revenue'].sum()
    region_orders = df.groupby("region")['order_id'].nunique()

    col1, col2 = st.columns(2)
    with col1:
        st.write("Revenue by Region")
        st.bar_chart(region_revenue)
    with col2:
        st.write("Orders by Region")
        st.bar_chart(region_orders)

    st.subheader("Top vs Low Performing Regions")
    st.write("Top Regions by Revenue:")
    st.dataframe(region_revenue.sort_values(ascending=False).head(5))
    st.write("Lowest Regions by Revenue:")
    st.dataframe(region_revenue.sort_values().head(5))

    # -------------------------------
    # 💳 Payment Insights
    # -------------------------------
    st.header("💳 Payment Insights")

    st.subheader("Popular Payment Methods")
    payment_counts = df['payment_method'].value_counts()
    st.bar_chart(payment_counts)
    st.write(payment_counts)

    st.subheader("Average Order Value (AOV) by Payment Method")
    aov_payment = df.groupby("payment_method")['revenue'].mean()
    st.bar_chart(aov_payment)
    st.write(aov_payment)

    # -------------------------------
    # 📈 Insights & Visualizations
    # -------------------------------
    st.header("📈 Insights & Visualizations")

    st.subheader("Heatmap: Price, Discount, Quantity, Revenue")
    corr = df[['price', 'discount', 'quantity', 'revenue']].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Sales Over Time")
    daily_sales = df.groupby("day")['revenue'].sum()
    st.line_chart(daily_sales)

    st.subheader("Revenue by Category")
    st.bar_chart(category_revenue)

    st.subheader("Boxplots: Order Values & Discounts")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(y=df['revenue'], ax=ax[0])
    ax[0].set_title("Order Value Distribution")
    sns.boxplot(y=df['discount'], ax=ax[1])
    ax[1].set_title("Discount Distribution")
    st.pyplot(fig)

    st.subheader("Payment Method Share")
    fig, ax = plt.subplots()
    ax.pie(payment_counts.values, labels=payment_counts.index, autopct='%1.1f%%', startangle=90)
    ax.set_title("Payment Method Distribution")
    st.pyplot(fig)
