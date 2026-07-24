import logging
from typing import Dict, Any

logger = logging.getLogger("AlphaCore.Alert")

class EnterpriseAlertEngine:
    """
    Modil Afichaj Notifikasyon an tan reyèl nan Terminal la.
    """
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