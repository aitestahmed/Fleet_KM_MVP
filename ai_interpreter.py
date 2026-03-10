import json
import pandas as pd
import streamlit as st
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

ALLOWED_DATASET_TYPES = ["fleet", "sales", "inventory", "maintenance", "other"]

def detect_schema_kpis_charts_with_ai(df: pd.DataFrame) -> dict:
    sample_df = df.head(12).copy()

    # تحويل القيم الصعبة إلى نص
    for col in sample_df.columns:
        sample_df[col] = sample_df[col].astype(str)

    sample_rows = sample_df.to_dict(orient="records")
    columns = [str(c) for c in df.columns.tolist()]

    prompt = f"""
You are a senior BI analyst.

Analyze this Excel dataset from:
1) column names
2) sample rows

Return STRICT JSON only.

Dataset columns:
{columns}

Sample rows:
{json.dumps(sample_rows, ensure_ascii=False)}

You must return JSON with this exact structure:

{{
  "dataset_type": "fleet or sales or inventory or maintenance or other",
  "schema": {{
    "date": "source column name or null",
    "vehicle_id": "source column name or null",
    "location": "source column name or null",
    "vehicle_type": "source column name or null",
    "account_type": "source column name or null",
    "expense_amount": "source column name or null",
    "revenue": "source column name or null",
    "kilometers": "source column name or null",

    "product": "source column name or null",
    "quantity": "source column name or null",
    "unit_price": "source column name or null",
    "customer": "source column name or null",

    "item": "source column name or null",
    "unit_cost": "source column name or null",
    "warehouse": "source column name or null",
    "category": "source column name or null",

    "asset_id": "source column name or null",
    "fault_type": "source column name or null",
    "maintenance_cost": "source column name or null",
    "downtime_hours": "source column name or null",
    "status": "source column name or null"
  }},
  "kpis": [
    "only choose from supported KPI names"
  ],
  "charts": [
    {{
      "type": "bar or line or pie",
      "x": "canonical field name",
      "y": "canonical field name",
      "aggregation": "sum or mean or count",
      "title": "short title"
    }}
  ],
  "notes": "short explanation"
}}

Supported KPI names by dataset_type:

fleet:
["total_km","total_cost","total_revenue","total_profit","cost_per_km","profit_margin_pct"]

sales:
["total_revenue","total_quantity","avg_unit_price","unique_products","unique_customers"]

inventory:
["total_stock_qty","total_stock_value","unique_items","unique_warehouses"]

maintenance:
["total_maintenance_cost","fault_count","avg_downtime_hours","unique_assets"]

Rules:
- dataset_type must be one of: fleet, sales, inventory, maintenance, other
- schema values must be original source column names exactly, or null
- choose only KPIs that make sense for this dataset
- choose 1 to 3 charts only
- do not include markdown
- return JSON only
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are an expert data analyst and BI architect."},
            {"role": "user", "content": prompt}
        ]
    )

    content = response.choices[0].message.content.strip()

    try:
        result = json.loads(content)
    except Exception:
        result = {
            "dataset_type": "other",
            "schema": {},
            "kpis": [],
            "charts": [],
            "notes": "AI response could not be parsed as JSON."
        }

    if result.get("dataset_type") not in ALLOWED_DATASET_TYPES:
        result["dataset_type"] = "other"

    return result
