import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("AlphaCore.Risk")

class EnterpriseRiskEngine:
    """
    Sistèm Kalkil Lot Size, Stop Loss ak Take Profit pwofesyonèl.
    """
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