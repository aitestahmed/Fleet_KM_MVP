import pandas as pd

def analyze_sales(df):

    if "product" not in df.columns:
        return pd.DataFrame({"Error":["Column product not found"]})

    sales = df.groupby("product").agg(

        revenue=("revenue","sum"),
        quantity=("quantity","sum")

    ).reset_index()

    sales["avg_price"] = sales["revenue"] / sales["quantity"]

    return sales