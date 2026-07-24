import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("AlphaCore.FeatureEngine")

class QuantitativeFeatureEngine:
    """
    Sistèm Kalkil Endikatè Quantitatifs ak Smart Money Concepts (FVG, Order Blocks, Volatilité).
    """
    @staticmethod
    def compute_institutional_features(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) < 50:
            return pd.DataFrame()

        df = df.copy()

        # 1. EMAs Filter
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
        df['EMA_Diff_Fast'] = df['EMA_20'] - df['EMA_50']
        df['EMA_Diff_Slow'] = df['EMA_50'] - df['EMA_200']

        # 2. RSI 14
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))

        # 3. ATR 14
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['Close'].shift(1)).abs()
        tr3 = (df['Low'] - df['Close'].shift(1)).abs()
        df['ATR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(window=14).mean()

        # 4. Bollinger Bands
        df['BB_Mid'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']

        # 5. Smart Money Concepts: Fair Value Gap (FVG)
        fvg_bull = (df['Low'] > df['High'].shift(2)).astype(int)
        fvg_bear = (df['High'] < df['Low'].shift(2)).astype(int)
        df['FVG'] = fvg_bull - fvg_bear

        # Momentum Vector
        df['Momentum_5'] = df['Close'] - df['Close'].shift(5)

        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df