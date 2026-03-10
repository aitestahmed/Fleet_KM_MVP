import pandas as pd

def analyze_inventory(df):

    if "quantity" not in df.columns:
        return pd.DataFrame({"Error":["Column quantity not found"]})

    if "unit_cost" not in df.columns:
        return pd.DataFrame({"Error":["Column unit_cost not found"]})

    df["stock_value"] = df["quantity"] * df["unit_cost"]

    return df