import pandas as pd

def load_excel(file):

    df = pd.read_excel(file)

    df.columns = df.columns.str.strip().str.lower()

    return df