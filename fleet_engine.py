import pandas as pd

def analyze_fleet(df):

    if "kilometers" not in df.columns:
        return pd.DataFrame({"Error":["Column kilometers not found"]})

    if "expense_amount" not in df.columns:
        df["expense_amount"] = 0

    summary = pd.DataFrame({

        "Total KM":[df["kilometers"].sum()],
        "Total Cost":[df["expense_amount"].sum()]

    })

    summary["Cost per KM"] = summary["Total Cost"] / summary["Total KM"]

    return summary