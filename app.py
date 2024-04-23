import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
import numpy as np
from millify import millify

st.markdown(
    """
<style>
[data-testid="stMetricValue"] {
    font-size: 40px;
#    color: rgb(0, 170, 60);
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def read_data(path):
    sales_df = pd.read_parquet(path)
    sales_df = sales_df.loc[(sales_df.UnitPrice > 0) & (
        sales_df.Quantity > 0)].reset_index(drop=True)
    sales_df["InvoiceDate"] = pd.to_datetime(sales_df["InvoiceDate"])
    return sales_df


@st.cache_data
def mergeIds(sales_df):
    bar = st.progress(0, "Vectorizing product names ...")
    # for each product we try to find similar products based on the decription
    prices = sales_df.groupby("Description")["UnitPrice"].mean()
    descriptions = sales_df.Description.unique()
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(descriptions)

    bar.progress(40, "Clustering product names ...")
    # cluster products by names
    clust = DBSCAN(metric="cosine")
    labels = clust.fit_predict(X)

    bar.progress(60, "Assigning new ids ...")
    # assign the same id to product with similar names and price
    Product_ids = {}
    for label in np.unique(labels):
        desc = descriptions[labels == label]
    # give products that are unique a unique id
        if label == -1:
            for i, k in enumerate(desc):
                Product_ids[k] = f"{i}"
        else:
            price_clust = MeanShift(bandwidth=1).fit(
                prices.loc[descriptions[labels == label]].to_numpy().reshape(-1, 1))
            for i, price_label in enumerate(price_clust.labels_):
                Product_ids[desc[i]] = f"{label}_{price_label}"
    bar.empty()
    return sales_df["Description"].map(Product_ids)


def first_day_of_interval(dates, interval_in_days):
    intervals = dates.apply(lambda x: x.dayofyear//interval_in_days+1)
    year = dates.apply(lambda x: x.year)
    days = pd.DataFrame({"intervals": intervals,
                         "date": dates.dt.date,
                         "year": year})
    return days.groupby(["intervals", "year"])["date"].transform(min)


df = read_data("test_data.parquet")
df["new_ids"] = mergeIds(df)

# get user input (sidebar)
interval_ = st.sidebar.slider("Interval size (days)", 1, 180,
                              help="This changes the time granularity of the plots and metrics")
product_id = st.sidebar.selectbox("Product id", options=df.new_ids.unique())

#product info
product_name = df[df.new_ids == product_id]["Description"].reset_index(drop=True)[
    0]
product_stockCode = df[df.new_ids == product_id]["StockCode"].reset_index(drop=True)[
    0]
st.markdown(f"# Product Info:")
st.markdown(f"#### :green[Name :] {product_name}")
st.markdown(f"#### :green[StockId :] {product_stockCode}")

# Interactive plots
st.divider()
st.markdown("# Sales, Price and EBIT during 2011")
col1, col2 = st.columns(spec=[8, 2])
# compute quantity(total) and price(averaged) over intervals
df[f"Date"] = first_day_of_interval(df.InvoiceDate, interval_)
sales = df[df.new_ids == product_id].sort_values("InvoiceDate").groupby(
    f"Date").agg({"Quantity": "sum", "UnitPrice": "mean"})
sales["EBIT"] = sales["Quantity"] * sales["UnitPrice"]
sales = sales.reset_index()


if interval_ == 1:
    period = "1 day"
else:
    period = f"{interval_} days"
col1.markdown(
    f"#### :green[Product Sales, Price and Earnings Before Interest and Taxes (EBIT), computed over {period}]")


# plot price - quantity - Profit
col1.line_chart(sales, x="Date", y="Quantity",
                height=160, use_container_width=True)
col1.line_chart(sales, x="Date", y="UnitPrice",
                height=160, use_container_width=True)
col1.line_chart(sales, x="Date", y="EBIT",
                height=160, use_container_width=True)

# plot metrics
col2.markdown("#")
col2.markdown("#")
col2.metric("Avg sales", value=f"{millify(sales.Quantity.mean(),precision=1)}")
col2.markdown("#")
col2.markdown("#")
col2.markdown("###")
col2.metric(
    "Avg price", value=f"{millify(sales.UnitPrice.mean(),precision=1)}$")
col2.markdown("###")
col2.markdown("###")
col2.markdown("#")
col2.metric("Avg EBIT", value=f"{millify(sales.EBIT.mean(),precision=1)}$")
