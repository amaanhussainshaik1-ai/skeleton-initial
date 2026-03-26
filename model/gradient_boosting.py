from model.base import BaseModel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import numpy as np
import random
 
seed = 0
np.random.seed(seed)
random.seed(seed)
 
 
class GradientBoosting(BaseModel):
    def __init__(self, model_name: str, embeddings: np.ndarray, y: np.ndarray) -> None:
        super().__init__()
        self.model_name  = model_name
        self.embeddings  = embeddings
        self.y           = y
        self.predictions = None
        self.mdl = GradientBoostingClassifier(n_estimators=200, random_state=seed)
        self.data_transform()
 
    def train(self, data) -> None:
        self.mdl.fit(data.get_X_train(), data.get_type_y_train())
 
    def predict(self, X_test: np.ndarray) -> None:
        self.predictions = self.mdl.predict(X_test)
 
    def print_results(self, data) -> None:
        print(f"\n--- Results for: {self.model_name} ---")
        print(classification_report(data.get_type_y_test(), self.predictions, zero_division=0))
 
    def data_transform(self) -> None:
        pass