# Encapsulates all data needed by ML models into one object.
# Separating this from the models means the input format stays
# consistent regardless of which model is being used (Feature 2).

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from Config import Config
import random

seed = Config.RANDOM_SEED
random.seed(seed)
np.random.seed(seed)


class Data:
    # Encapsulates training and testing splits for one classification target.

    def __init__(self,
                 X: np.ndarray,
                 df: pd.DataFrame,
                 target_col: str = None) -> None:

        if target_col is None:
            target_col = Config.CLASS_COL

        self.target_col = target_col

        # Keep only rows whose target label is not NaN / empty
        mask = df[target_col].notna() & (df[target_col].astype(str).str.strip() != '')
        X_clean  = X[mask]
        df_clean = df[mask].reset_index(drop=True)

        self.y          = df_clean[target_col].values
        self.embeddings = X_clean

        # Stratified split when possible; fall back to random split for
        # chained labels that may have singleton classes.
        counts = pd.Series(self.y).value_counts()
        use_stratify = (counts >= 2).all()
        (self.X_train, self.X_test,
         self.y_train, self.y_test,
         self.train_df, self.test_df) = train_test_split(
            X_clean,
            self.y,
            df_clean,
            test_size=Config.TEST_SIZE,
            random_state=seed,
            stratify=self.y if use_stratify else None,
        )

        print(f"[data_model] target='{target_col}' | "
              f"train={len(self.y_train)}, test={len(self.y_test)} | "
              f"classes={len(np.unique(self.y))}")

    # Accessor methods (used by models and controller)

    def get_type(self):
        return self.y

    def get_X_train(self):
        return self.X_train

    def get_X_test(self):
        return self.X_test

    def get_type_y_train(self):
        return self.y_train

    def get_type_y_test(self):
        return self.y_test

    def get_train_df(self):
        return self.train_df

    def get_embeddings(self):
        return self.embeddings

    def get_type_test_df(self):
        return self.test_df
