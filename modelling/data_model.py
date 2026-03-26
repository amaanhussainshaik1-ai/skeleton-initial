import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import *
from utils import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)

class Data():
    def __init__(self,
                 X: np.ndarray,
                 df: pd.DataFrame,
                 target_col: str = None) -> None:
                 # This method will create the model for data
                 #This will be performed in second activity

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


    def get_type(self):
        return  self.y
    def get_X_train(self):
        return  self.X_train
    def get_X_test(self):
        return  self.X_test
    def get_type_y_train(self):
        return  self.y_train
    def get_type_y_test(self):
        return  self.y_test
    def get_train_df(self):
        return  self.train_df
    def get_embeddings(self):
        return  self.embeddings
    def get_type_test_df(self):
        return  self.test_df


