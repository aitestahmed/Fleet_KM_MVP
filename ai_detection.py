from openai import OpenAI
import streamlit as st

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def detect_dataset(columns):

    prompt = f"""
    Identify dataset type.

    Columns:
    {columns}

    Possible types:

    fleet
    sales
    inventory
    maintenance

    Return only one word.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )

    return response.choices[0].message.content.strip().lower()