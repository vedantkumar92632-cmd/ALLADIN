import datetime
from zoneinfo import ZoneInfo
from core.time_utils import IST

class Clock:
    def __init__(self):
        self._mode = "LIVE"  # "LIVE" or "BACKTEST"
        self._backtest_time = None

    def set_mode(self, mode: str):
        """Set clock mode to either 'LIVE' or 'BACKTEST'."""
        mode_upper = mode.upper()
        if mode_upper not in ("LIVE", "BACKTEST"):
            raise ValueError("Mode must be 'LIVE' or 'BACKTEST'")
        self._mode = mode_upper
        if self._mode == "BACKTEST" and self._backtest_time is None:
            # Default initialization for backtest time if not set
            self._backtest_time = datetime.datetime.now(IST)

    def get_mode(self) -> str:
        """Get the current mode of the clock."""
        return self._mode

    def set_time(self, dt: datetime.datetime):
        """Set the current backtest time (applicable only in BACKTEST mode)."""
        if self._mode != "BACKTEST":
            raise RuntimeError("Cannot set time when clock is in LIVE mode.")
        
        # Ensure it is timezone-aware and converted to IST
        if dt.tzinfo is None:
            self._backtest_time = dt.replace(tzinfo=datetime.timezone.utc).astimezone(IST)
        else:
            self._backtest_time = dt.astimezone(IST)

    def step(self, delta: datetime.timedelta):
        """Advance backtest time by a timedelta (applicable only in BACKTEST mode)."""
        if self._mode != "BACKTEST":
            raise RuntimeError("Cannot step time when clock is in LIVE mode.")
        if self._backtest_time is None:
            raise RuntimeError("Backtest time is not initialized.")
        self._backtest_time += delta

    def now(self) -> datetime.datetime:
        """Get the current time (real-time IST in LIVE, simulated in BACKTEST)."""
        if self._mode == "LIVE":
            return datetime.datetime.now(IST)
        else:
            if self._backtest_time is None:
                raise RuntimeError("Backtest time is not initialized.")
            return self._backtest_time

# Global clock instance
clock = Clock()
