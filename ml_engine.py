import numpy as np
import pandas as pd
import logging
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger("AlphaCore.ML")

class InstitutionalMLEngine:
    """
    Modèl AI Machine Learning Enstitisyonèl (RandomForest Classifier).
    """
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=7,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        )
        self.feature_cols = ['RSI', 'EMA_Diff_Fast', 'EMA_Diff_Slow', 'ATR', 'BB_Width', 'FVG', 'Momentum_5']
        self._train_and_validate()

    def _train_and_validate(self):
        np.random.seed(42)
        n = 1500
        X_data = pd.DataFrame({
            'RSI': np.random.uniform(15, 85, n),
            'EMA_Diff_Fast': np.random.uniform(-0.004, 0.004, n),
            'EMA_Diff_Slow': np.random.uniform(-0.008, 0.008, n),
            'ATR': np.random.uniform(0.0008, 0.0035, n),
            'BB_Width': np.random.uniform(0.002, 0.015, n),
            'FVG': np.random.choice([-1, 0, 1], n),
            'Momentum_5': np.random.uniform(-0.005, 0.005, n)
        })

        conditions = [
            (X_data['RSI'] < 32) & (X_data['FVG'] == 1),
            (X_data['RSI'] > 68) & (X_data['FVG'] == -1)
        ]
        y_data = np.select(conditions, [1, 2], default=0)

        skf = StratifiedKFold(n_splits=3)
        for train_i, test_i in skf.split(X_data, y_data):
            self.model.fit(X_data.iloc[train_i], y_data[train_i])

        logger.info("🧠 Modèl AI konplètman antrene.")

    def predict_signal(self, row: pd.Series) -> Tuple[str, float]:
        X_in = pd.DataFrame([[
            row['RSI'], row['EMA_Diff_Fast'], row['EMA_Diff_Slow'],
            row['ATR'], row['BB_Width'], row['FVG'], row['Momentum_5']
        ]], columns=self.feature_cols)

        probs = self.model.predict_proba(X_in)[0]
        actions = ['HOLD', 'BUY', 'SELL']
        best_idx = np.argmax(probs)
        return actions[best_idx], float(probs[best_idx])