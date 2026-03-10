import pandas as pd

def apply_ai_schema(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    rename_map = {}

    for canonical_name, source_name in schema.items():
        if source_name and source_name in df.columns:
            rename_map[source_name] = canonical_name

    mapped_df = df.rename(columns=rename_map).copy()

    return mapped_df
