from openai import OpenAI
import streamlit as st

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def generate_report(data):

    prompt = f"""
    Analyze this data and write management insights:

    {data}
    """

    response = client.chat.completions.create(

        model="gpt-4o-mini",

        messages=[

            {"role":"system","content":"You are a business data analyst."},

            {"role":"user","content":prompt}

        ]
    )

    return response.choices[0].message.content