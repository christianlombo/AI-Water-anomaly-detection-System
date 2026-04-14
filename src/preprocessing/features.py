import pandas as pd

def add_water_loss_feats(df):
    df = df.copy()

    df["hour"] = df.index.hour

    df["is_night"] = ((df["hour"] >= 1) & (df["hour"] <= 4)).astype(int)

    flow_cols = [c for c in df.columns if c.startswith('F_')]
    for c in flow_cols:
        df[f"{c}_mnf"] = (
            df[c].where(df["is_night"] == 1)
            .rolling(24)
            .min()
            .ffill()
            .bfill()
        )

    for c in flow_cols:
        df[f"{c}_delta"] = df[c].diff().fillna(0)

    return df.fillna(0) 
