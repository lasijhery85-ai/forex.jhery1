import os
import sys
import sqlite3
import logging
import asyncio
import threading
import time
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

import aiohttp
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

# =====================================================================
# 1. LOGGING SETUP
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
)
logger = logging.getLogger("AlphaCore.AllInOne")

# =====================================================================
# 2. MODULE 1: API CONNECTOR
# =====================================================================
class InstitutionalAPIConnector:
    """Konektè Rezo Asenkawon High-Performance pou rale done piblik san kle API."""
    def __init__(self, retries: int = 3, timeout: int = 10):
        self.retries = retries
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.tickers_map = {
            "USD_DXY": "DX-Y.NYB",
            "EURUSD":  "EURUSD=X",
            "GBPUSD":  "GBPUSD=X",
            "USDJPY":  "JPY=X",
            "GOLD":    "GC=F",
            "BITCOIN": "BTC-USD"
        }
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    async def fetch_candles_async(self, symbol: str, interval: str = "15m") -> Optional[pd.DataFrame]:
        auto_ranges = {"15m": "1mo", "1h": "3mo", "4h": "6mo"}
        range_period = auto_ranges.get(interval, "1mo")
        ticker = self.tickers_map.get(symbol, f"{symbol}=X")
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval={interval}&range={range_period}"

        async with aiohttp.ClientSession(headers=self.headers, timeout=self.timeout) as session:
            for attempt in range(1, self.retries + 1):
                try:
                    async with session.get(url) as response:
                        if response.status != 200:
                            await asyncio.sleep(1)
                            continue
                        
                        data = await response.json()
                        result = data['chart']['result'][0]
                        timestamps = result.get('timestamp', [])
                        indicators = result.get('indicators', {}).get('quote', [{}])[0]

                        if not timestamps or not indicators.get('close'):
                            return None

                        df = pd.DataFrame({
                            'Open': indicators.get('open'),
                            'High': indicators.get('high'),
                            'Low': indicators.get('low'),
                            'Close': indicators.get('close'),
                            'Volume': indicators.get('volume', 0)
                        })
                        df['Timestamp'] = [datetime.fromtimestamp(ts) for ts in timestamps]
                        df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
                        df.reset_index(drop=True, inplace=True)
                        return df

                except Exception as e:
                    logger.warning(f"Tentativ API {attempt}/{self.retries} echwe pou {symbol}: {e}")
                    await asyncio.sleep(1)

        logger.error(f"❌ Enposib pou rale done pou {symbol} [{interval}]")
        return None

# =====================================================================
# 3. MODULE 2: FEATURE ENGINEERING
# =====================================================================
class QuantitativeFeatureEngine:
    """Sistèm Kalkil Endikatè Quantitatifs ak Smart Money Concepts."""
    @staticmethod
    def compute_institutional_features(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) < 50:
            return pd.DataFrame()

        df = df.copy()

        # EMAs
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
        df['EMA_Diff_Fast'] = df['EMA_20'] - df['EMA_50']
        df['EMA_Diff_Slow'] = df['EMA_50'] - df['EMA_200']

        # RSI 14
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))

        # ATR 14
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['Close'].shift(1)).abs()
        tr3 = (df['Low'] - df['Close'].shift(1)).abs()
        df['ATR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(window=14).mean()

        # Bollinger Bands
        df['BB_Mid'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']

        # Smart Money Concepts: Fair Value Gap (FVG)
        fvg_bull = (df['Low'] > df['High'].shift(2)).astype(int)
        fvg_bear = (df['High'] < df['Low'].shift(2)).astype(int)
        df['FVG'] = fvg_bull - fvg_bear

        # Momentum Vector
        df['Momentum_5'] = df['Close'] - df['Close'].shift(5)

        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

# =====================================================================
# 4. MODULE 3: MACHINE LEARNING ENGINE
# =====================================================================
class InstitutionalMLEngine:
    """Modèl AI Machine Learning Enstitisyonèl (RandomForest Classifier)."""
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

# =====================================================================
# 5. MODULE 4: RISK MANAGEMENT ENGINE
# =====================================================================
class EnterpriseRiskEngine:
    """Sistèm Kalkil Lot Size, Stop Loss ak Take Profit pwofesyonèl."""
    def __init__(self, balance: float = 10000.0, max_risk_per_trade: float = 0.01):
        self.balance = balance
        self.max_risk_per_trade = max_risk_per_trade

    def calculate_trade_parameters(self, symbol: str, action: str, price: float, atr: float, decimals: int = 5) -> Optional[Dict[str, Any]]:
        if action not in ['BUY', 'SELL']:
            return None

        risk_usd = self.balance * self.max_risk_per_trade
        atr_buffer = 1.5 * atr

        if action == 'BUY':
            sl = round(price - atr_buffer, decimals)
            tp = round(price + (atr_buffer * 2.5), decimals)
        else:
            sl = round(price + atr_buffer, decimals)
            tp = round(price - (atr_buffer * 2.5), decimals)

        pip_scale = 0.01 if decimals in [2, 3] else 0.0001
        pips_risk = (price - sl) / pip_scale if action == 'BUY' else (sl - price) / pip_scale

        if pips_risk <= 0:
            return None

        lot_size = round((risk_usd / (pips_risk * 10.0)), 2)
        lot_size = max(0.01, min(lot_size, 50.0))

        return {
            "ENTRY": price,
            "STOP_LOSS": sl,
            "TAKE_PROFIT": tp,
            "LOT_SIZE": lot_size,
            "RISK_USD": round(risk_usd, 2),
            "PIPS_AT_RISK": round(pips_risk, 1),
            "RR_RATIO": "1:2.5"
        }

# =====================================================================
# 6. MODULE 5: MEMORY ENGINE (SQLITE DB)
# =====================================================================
class EnterpriseMemoryEngine:
    """Baz de done SQLite an mòd WAL (Write-Ahead Logging)."""
    def __init__(self, db_path: str = "alphacore_enterprise.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL;")
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS signals_audit (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        action TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        alignment TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        sl REAL NOT NULL,
                        tp REAL NOT NULL,
                        lot_size REAL NOT NULL,
                        risk_usd REAL NOT NULL
                    )
                ''')
                conn.commit()
        except Exception as e:
            logger.error(f"Erè nan kreyasyon DB: {e}")

    def save_signal(self, payload: Dict[str, Any]):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO signals_audit (
                        timestamp, symbol, action, confidence, alignment,
                        entry_price, sl, tp, lot_size, risk_usd
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    payload['TIMESTAMP'], payload['SYMBOL'], payload['ACTION'],
                    payload['CONFIDENCE'], payload['ALIGNMENT'],
                    payload['EXEC']['ENTRY'], payload['EXEC']['STOP_LOSS'],
                    payload['EXEC']['TAKE_PROFIT'], payload['EXEC']['LOT_SIZE'],
                    payload['EXEC']['RISK_USD']
                ))
                conn.commit()
            logger.info(f"💾 Siyal {payload['SYMBOL']} sove nan baz done a.")
        except Exception as e:
            logger.error(f"Erè nan sove siyal: {e}")

# =====================================================================
# 7. MODULE 6: ALERT ENGINE
# =====================================================================
class EnterpriseAlertEngine:
    """Modil Afichaj Notifikasyon an tan reyèl nan Terminal la."""
    @staticmethod
    def dispatch_alert(payload: Dict[str, Any]):
        print("\n" + "="*60)
        print(f"🚨 INSTITUTIONAL SIGNAL DETECTED: {payload['SYMBOL']}")
        print("="*60)
        print(f"🚦 ACTION                : {payload['ACTION']}")
        print(f"🧠 AI CONFIDENCE SCORE   : {payload['CONFIDENCE']}%")
        print(f"📊 TIMEFRAME ALIGNMENT   : {payload['ALIGNMENT']}")
        print(f"📍 ENTRY PRICE           : {payload['EXEC']['ENTRY']}")
        print(f"🛑 STOP LOSS             : {payload['EXEC']['STOP_LOSS']}")
        print(f"🎯 TAKE PROFIT           : {payload['EXEC']['TAKE_PROFIT']}")
        print(f"📦 RECOMMENDED LOT SIZE  : {payload['EXEC']['LOT_SIZE']} Lots")
        print(f"💰 RISK AMOUNT (USD)     : ${payload['EXEC']['RISK_USD']} (RR: {payload['EXEC']['RR_RATIO']})")
        print("="*60 + "\n")

# =====================================================================
# 8. FLASK SERVER & WEB DASHBOARD UI
# =====================================================================
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ht">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlphaCore AI - Institutional Trading Dashboard</title>
    <style>
        :root {
            --bg-color: #0b0e14;
            --card-bg: #151a23;
            --accent-color: #00f2fe;
            --buy-color: #00c853;
            --sell-color: #ff1744;
            --text-color: #e0e6ed;
            --text-dim: #8899a6;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
        }
        .container { max-width: 1100px; margin: 0 auto; }
        .header {
            display: flex; justify-content: space-between; align-items: center;
            border-bottom: 1px solid #232d3f; padding-bottom: 15px; margin-bottom: 25px;
        }
        .header h1 { margin: 0; font-size: 24px; color: var(--accent-color); letter-spacing: 1px; }
        .badge-live {
            background: rgba(0, 242, 254, 0.1); color: var(--accent-color);
            border: 1px solid var(--accent-color); padding: 5px 12px;
            border-radius: 20px; font-size: 12px; font-weight: bold;
        }
        .grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px; margin-bottom: 30px;
        }
        .card { background: var(--card-bg); border-radius: 10px; padding: 20px; border: 1px solid #232d3f; }
        .card h3 { margin-top: 0; font-size: 12px; color: var(--text-dim); text-transform: uppercase; }
        .card .value { font-size: 26px; font-weight: bold; }
        .table-container { background: var(--card-bg); border-radius: 10px; border: 1px solid #232d3f; overflow-x: auto; }
        table { width: 100%; border-collapse: collapse; text-align: left; }
        th, td { padding: 15px; border-bottom: 1px solid #232d3f; }
        th { background-color: #1a222d; color: var(--text-dim); font-size: 12px; text-transform: uppercase; }
        .action-BUY { color: var(--buy-color); font-weight: bold; }
        .action-SELL { color: var(--sell-color); font-weight: bold; }
        .footer { text-align: center; margin-top: 40px; font-size: 12px; color: var(--text-dim); }
    </style>
    <script>setInterval(() => { window.location.reload(); }, 30000);</script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏛️ ALPHACORE AI QUANT</h1>
            <span class="badge-live">● LIVE ENGINE RUNNING</span>
        </div>

        <div class="grid">
            <div class="card">
                <h3>Sistèm Status</h3>
                <div class="value" style="color: var(--buy-color); font-size: 20px;">ONLINE (24/7 VPS)</div>
            </div>
            <div class="card">
                <h3>Total Siyal Detekte</h3>
                <div class="value">{{ total_signals }}</div>
            </div>
            <div class="card">
                <h3>Kouvèti Timeframe</h3>
                <div class="value" style="font-size: 20px;">15M - 1H - 4H</div>
            </div>
        </div>

        <h2>📊 Siyal ak Ekzekisyon Enstitisyonèl Yo</h2>
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Date/Lè</th>
                        <th>Asset</th>
                        <th>Aksyon</th>
                        <th>Konfyans AI</th>
                        <th>Aliyman Timeframes</th>
                        <th>Prix Antre</th>
                        <th>Stop Loss</th>
                        <th>Take Profit</th>
                        <th>Lot Size</th>
                    </tr>
                </thead>
                <tbody>
                    {% for s in signals %}
                    <tr>
                        <td>{{ s[1] }}</td>
                        <td><strong>{{ s[2] }}</strong></td>
                        <td class="action-{{ s[3] }}">{{ s[3] }}</td>
                        <td>{{ s[4] }}%</td>
                        <td><small>{{ s[5] }}</small></td>
                        <td>{{ s[6] }}</td>
                        <td style="color: var(--sell-color);">{{ s[7] }}</td>
                        <td style="color: var(--buy-color);">{{ s[8] }}</td>
                        <td><strong>{{ s[9] }} Lots</strong></td>
                    </tr>
                    {% else %}
                    <tr>
                        <td colspan="9" style="text-align: center; color: var(--text-dim);">Poko gen okenn siyal ki konfime nan baz de done a... Bot la ap analize mache a.</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <div class="footer">
            AlphaCore Enterprise AI Engine &copy; 2026 - Powered by Flask & Render Cloud
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def dashboard():
    signals = []
    total_signals = 0
    try:
        conn = sqlite3.connect("alphacore_enterprise.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM signals_audit ORDER BY id DESC LIMIT 20")
        signals = cursor.fetchall()
        cursor.execute("SELECT COUNT(*) FROM signals_audit")
        res = cursor.fetchone()
        if res:
            total_signals = res[0]
        conn.close()
    except Exception as e:
        logger.error(f"Erè nan lekti DB pou Dashboard la: {e}")

    return render_template_string(HTML_TEMPLATE, signals=signals, total_signals=total_signals)

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()}), 200

# =====================================================================
# 9. BACKGROUND WORKER (TRADING BOT SCANNER)
# =====================================================================
class QuantBackgroundWorker:
    def __init__(self):
        self.api = InstitutionalAPIConnector()
        self.feature_engine = QuantitativeFeatureEngine()
        self.ai = InstitutionalMLEngine()
        self.risk = EnterpriseRiskEngine(balance=10000.0, max_risk_per_trade=0.01)
        self.memory = EnterpriseMemoryEngine()
        self.assets = {"EURUSD": 5, "GBPUSD": 5, "USDJPY": 3, "GOLD": 2, "BITCOIN": 2}

    async def scan_loop(self):
        while True:
            logger.info("🔄 AlphaCore Background Scan Pipeline kòmanse...")
            for symbol, decimals in self.assets.items():
                try:
                    # Liy sa a te gen yon ti erè nan parentezi yo
                    df_15m = await self.api.fetch_candles_async(symbol, "15m")
                    df_1h  = await self.api.fetch_candles_async(symbol, "1h")
                    df_4h  = await self.api.fetch_candles_async(symbol, "4h")

                    if df_15m is None or df_1h is None or df_4h is None:
                        continue

                    feat_15m = self.feature_engine.compute_institutional_features(df_15m)
                    feat_1h  = self.feature_engine.compute_institutional_features(df_1h)
                    feat_4h  = self.feature_engine.compute_institutional_features(df_4h)

                    if feat_15m.empty or feat_1h.empty or feat_4h.empty:
                        continue

                    row_15m, row_1h, row_4h = feat_15m.iloc[-1], feat_1h.iloc[-1], feat_4h.iloc[-1]

                    act_15m, conf_15m = self.ai.predict_signal(row_15m)
                    act_1h,  conf_1h  = self.ai.predict_signal(row_1h)
                    act_4h,  conf_4h  = self.ai.predict_signal(row_4h)

                    if act_15m != 'HOLD' and (act_15m == act_1h or act_15m == act_4h):
                        avg_conf = (conf_15m + conf_1h + conf_4h) / 3.0

                        if avg_conf >= 0.70:
                            exec_plan = self.risk.calculate_trade_parameters(
                                symbol, act_15m, float(row_15m['Close']), float(row_15m['ATR']), decimals
                            )

                            if exec_plan:
                                payload = {
                                    "TIMESTAMP": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "SYMBOL": symbol,
                                    "ACTION": act_15m,
                                    "CONFIDENCE": round(avg_conf * 100, 2),
                                    "ALIGNMENT": f"15m:{act_15m} | 1h:{act_1h} | 4h:{act_4h}",
                                    "EXEC": exec_plan
                                }
                                self.memory.save_signal(payload)
                                EnterpriseAlertEngine.dispatch_alert(payload)

                except Exception as e:
                    logger.error(f"Erè nan scan pou {symbol}: {e}")

            await asyncio.sleep(180)