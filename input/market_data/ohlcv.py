import datetime
from dataclasses import dataclass
from typing import Optional

@dataclass
class OHLCVBar:
    """
    Represents a raw OHLCV bar ingested into the system.
    Field reference is based on Section 3 Part 4 (1.5 OHLCV Bar Data).
    """
    ticker: str
    timestamp: datetime.datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    vwap: Optional[float] = None
    trade_count: Optional[int] = None
    adj_close: Optional[float] = None

    def validate(self) -> bool:
        """
        Validate the OHLCV bar data.
        Returns True if the data is valid, False otherwise.
        """
        if not self.ticker:
            return False
        if not isinstance(self.timestamp, datetime.datetime):
            return False
        if self.open_price <= 0 or self.high_price <= 0 or self.low_price <= 0 or self.close_price <= 0:
            return False
        if self.volume < 0:
            return False
        # High must be greater than or equal to Open, Low, and Close
        if self.high_price < max(self.open_price, self.low_price, self.close_price):
            return False
        # Low must be less than or equal to Open, High, and Close
        if self.low_price > min(self.open_price, self.high_price, self.close_price):
            return False
        return True
