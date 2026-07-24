import os
import sys
import threading
import time
import sqlite3
import logging
import asyncio
from flask import Flask, render_template_string, jsonify
from datetime import datetime

# Importasyon Modil Pwojè yo
from api_connector import InstitutionalAPIConnector
from feature_engineering import QuantitativeFeatureEngine
from ml_engine import InstitutionalMLEngine
from risk_engine import EnterpriseRiskEngine
from memory_engine import EnterpriseMemoryEngine
from alert_engine import EnterpriseAlertEngine

# Setup System Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
)
logger = logging.getLogger("AlphaCore.Web")

# Setup Flask Server
app = Flask(__name__)

# Dashboard UI HTML/CSS High-End
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
        .container {
            max-width: 1100px;
            margin: 0 auto;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #232d3f;
            padding-bottom: 15px;
            margin-bottom: 25px;
        }
        .header h1 {
            margin: 0;
            font-size: 24px;
            color: var(--accent-color);
            letter-spacing: 1px;
        }
        .badge-live {
            background: rgba(0, 242, 254, 0.1);
            color: var(--accent-color);
            border: 1px solid var(--accent-color);
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: var(--card-bg);
            border-radius: 10px;
            padding: 20px;
            border: 1px solid #232d3f;
        }
        .card h3 {
            margin-top: 0;
            font-size: 12px;
            color: var(--text-dim);
            text-transform: uppercase;
        }
        .card .value {
            font-size: 26px;
            font-weight: bold;
        }
        .table-container {
            background: var(--card-bg);
            border-radius: 10px;
            border: 1px solid #232d3f;
            overflow-x: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }
        th, td {
            padding: 15px;
            border-bottom: 1px solid #232d3f;
        }
        th {
            background-color: #1a222d;
            color: var(--text-dim);
            font-size: 12px;
            text-transform: uppercase;
        }
        .action-BUY { color: var(--buy-color); font-weight: bold; }
        .action-SELL { color: var(--sell-color); font-weight: bold; }
        .footer {
            text-align: center;
            margin-top: 40px;
            font-size: 12px;
            color: var(--text-dim);
        }
    </style>
    <script>
        setInterval(() => { window.location.reload(); }, 30000);
    </script>
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

# ---------------------------------------------------------------------
# BACKGROUND WORKER (AI Quant Scanning Loop)
# ---------------------------------------------------------------------
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
                    df_15m, df_1h, df_4h = await asyncio.gather(
                        self.api.fetch_candles_async(symbol, "15m"),
                        self.api.fetch_candles_async(symbol, "1h"),
                        self.api.fetch_candles_async(symbol, "4h")
                    )

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

            # Espace tan ant chak scan (3 minit)
            await asyncio.sleep(180)

def start_background_loop():
    worker = QuantBackgroundWorker()
    asyncio.run(worker.scan_loop())

# Lanse Thread an background pou bot la
bg_thread = threading.Thread(target=start_background_loop, daemon=True)
bg_thread.start()

# Entrada prensipal
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)