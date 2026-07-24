import aiohttp
import asyncio
import pandas as pd
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger("AlphaCore.API")

class InstitutionalAPIConnector:
    """
    Konektè Rezo Asenkawon High-Performance pou rale done piblik san kle API.
    """
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