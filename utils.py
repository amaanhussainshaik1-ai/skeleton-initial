# Reusable helper utilities shared across the project.

import numpy as np
import pandas as pd
from Config import Config


def build_chain_labels(df: pd.DataFrame) -> pd.DataFrame:
    # Design Choice 1 — Chained Multi-outputs.
    
    df = df.copy()
    df[Config.CHAIN_Y2]       = df['y2'].astype(str)
    df[Config.CHAIN_Y2_Y3]    = df['y2'].astype(str) + ' + ' + df['y3'].astype(str)
    df[Config.CHAIN_Y2_Y3_Y4] = (df['y2'].astype(str) + ' + ' +
                                  df['y3'].astype(str) + ' + ' +
                                  df['y4'].astype(str))
    return df
