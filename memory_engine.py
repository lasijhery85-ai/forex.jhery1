import sqlite3
import logging
from typing import Dict, Any

logger = logging.getLogger("AlphaCore.Memory")

class EnterpriseMemoryEngine:
    """
    Baz de done SQLite an mòd WAL (Write-Ahead Logging).
    """
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