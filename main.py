# Orchestrates the full pipeline. Never contains preprocessing or
# model-specific logic; it only calls functions from other modules.

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import random
import numpy as np
import pandas as pd

from preprocess import get_input_data, de_duplication, noise_remover, translate_to_en
from embeddings import get_tfidf_embd
from modelling.modelling import model_predict, print_comparison_table
from modelling.data_model import Data
from utils import build_chain_labels
from Config import Config

seed = Config.RANDOM_SEED
random.seed(seed)
np.random.seed(seed)


# Pipeline steps

def load_data() -> pd.DataFrame:
    return get_input_data()


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = de_duplication(df)
    df = noise_remover(df)
    df[Config.TICKET_SUMMARY]      = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY]      = df[Config.TICKET_SUMMARY].astype('U')
    return df


def get_embeddings(df: pd.DataFrame):
    X = get_tfidf_embd(df)
    return X, df


def get_data_object(X: np.ndarray, df: pd.DataFrame, target_col: str = None) -> Data:
    return Data(X, df, target_col=target_col)


# Design Choice 1 — Chained Multi-outputs across ALL models

def run_chained_multioutput(X: np.ndarray, df: pd.DataFrame) -> dict:
    """
    Run every model in MODEL_REGISTRY across three chain levels.
    Returns a dict of {level_label: [(model_name, acc, f1), ...]}
    """
    print("\n" + "#"*60)
    print("  DESIGN CHOICE 1 — CHAINED MULTI-OUTPUT CLASSIFICATION")
    print("#"*60)

    df_chained = build_chain_labels(df)

    chain_levels = [
        (Config.CHAIN_Y2,       "Level 1: Type2 only"),
        (Config.CHAIN_Y2_Y3,    "Level 2: Type2+Type3"),
        (Config.CHAIN_Y2_Y3_Y4, "Level 3: Type2+Type3+Type4"),
    ]

    all_results = {}
    for col, label in chain_levels:
        data = get_data_object(X, df_chained, target_col=col)
        results = model_predict(data, df_chained, label)
        all_results[label] = results

    return all_results


# Entry point

if __name__ == '__main__':

    # 1. Load & preprocess
    df = load_data()
    df = preprocess_data(df)

    # 2. TF-IDF embeddings
    X, df = get_embeddings(df)

    # 3. Run chained multi-output across all models (includes Type2-only as Level 1)
    all_results = run_chained_multioutput(X, df)

    # 4. Print the full comparison table
    print_comparison_table(all_results)
