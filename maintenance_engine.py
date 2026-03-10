import pandas as pd

def analyze_maintenance(df):

    if "vehicle" not in df.columns:
        return pd.DataFrame({"Error":["Column vehicle not found"]})

    failures = df.groupby("vehicle").size().reset_index(name="failure_count")

    return failures