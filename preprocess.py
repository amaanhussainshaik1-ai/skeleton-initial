# All data-loading and preprocessing steps live here.
# The controller (main.py) calls these functions; it never touches raw CSV logic.

import os
import re
import numpy as np
import pandas as pd
from Config import Config


# 1. Data Loading

def get_input_data() -> pd.DataFrame:
    """Load and concatenate all CSV files from the data/ folder."""
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    frames = []
    for fname in os.listdir(data_dir):
        if fname.endswith('.csv'):
            path = os.path.join(data_dir, fname)
            df = pd.read_csv(path, encoding='utf-8', encoding_errors='replace')
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    # Rename raw Type columns to internal names
    combined.rename(columns={
        Config.TYPE1_COL: Config.GROUPED,
        Config.TYPE2_COL: 'y2',
        Config.TYPE3_COL: 'y3',
        Config.TYPE4_COL: 'y4',
    }, inplace=True)

    return combined


# 2. Deduplication

def de_duplication(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    print(f"[preprocess] Deduplication: {before} -> {len(df)} rows")
    return df.reset_index(drop=True)


# 3. Noise Removal

def noise_remover(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Drop rows where key label columns are NaN.
    - Strip whitespace from text and label columns.
    - Remove classes that appear fewer than MIN_CLASS_SAMPLES times.
    """
    df = df.dropna(subset=["y2"]).copy()
    df['y3'] = df['y3'].fillna('Unknown')
    df['y4'] = df['y4'].fillna('Unknown')

    for col in [Config.TICKET_SUMMARY, Config.INTERACTION_CONTENT]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    for col in ['y2', 'y3', 'y4']:
        df[col] = df[col].astype(str).str.strip()

    counts = df['y2'].value_counts()
    valid  = counts[counts >= Config.MIN_CLASS_SAMPLES].index
    before = len(df)
    df = df[df["y2"].isin(valid)].copy()
    print(f"[preprocess] Rare-class removal: {before} -> {len(df)} rows")

    return df.reset_index(drop=True)


# 4. Translation (stub)

def translate_to_en(texts: list) -> list:
    """Stub: returns texts unchanged. Replace with real translator if needed."""
    return texts
