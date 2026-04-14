import pandas as pd
import numpy as np

def load_data(path):

    df = pd.read_csv(path)

    df.columns = df.columns.str.strip()

    try:
        df["DATETIME"] = pd.to_datetime(
            df["DATETIME"].str.strip(),
            format ="%d/%m/%y %H"
        )
    except ValueError:
        df["DATETIME"] = pd.to_datetime(
            df["DATETIME"].str.strip(), 
            format="%d/%m/%y %H:%M"
        )

    df = df.set_index("DATETIME").sort_index()

    df = df.interpolate(method="linear", limit=3)

    df = df.fillna(df.median())

    print(f"Loaded {path} with {df.shape[0]} rows.")
    return df

