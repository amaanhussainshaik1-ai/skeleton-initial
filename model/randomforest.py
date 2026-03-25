# Concrete implementation of BaseModel using scikit-learn's RandomForest.
# All model-specific code (train, predict, evaluate) stays here so the
# controller only ever calls the three abstract methods defined in base.py.

import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import random

seed = 0
np.random.seed(seed)
random.seed(seed)


class RandomForest(BaseModel):
    # Random Forest classifier that conforms to the BaseModel interface.

    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        super().__init__()
        self.model_name  = model_name
        self.embeddings  = embeddings
        self.y           = y
        self.predictions = None

        self.mdl = RandomForestClassifier(
            n_estimators=500,
            random_state=seed,
            class_weight='balanced_subsample',
        )

        self.data_transform()

    # Abstract method implementations (required by BaseModel)

    def train(self, data) -> None:
        # Fit the RandomForest on training data.
        self.mdl.fit(data.get_X_train(), data.get_type_y_train())

    def predict(self, X_test: np.ndarray) -> None:
        # Run inference and store predictions.
        self.predictions = self.mdl.predict(X_test)

    def print_results(self, data) -> None:
        # Print a full classification report.
        print(f"\n--- Results for: {self.model_name} ---")
        print(classification_report(
            data.get_type_y_test(),
            self.predictions,
            zero_division=0,
        ))

    def data_transform(self) -> None:
        # Placeholder for any feature engineering specific to this model.
        # Currently a no-op because TF-IDF embeddings are already prepared
        # in embeddings.py before this model is instantiated.
        pass
