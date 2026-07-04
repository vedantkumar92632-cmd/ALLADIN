from dataclasses import dataclass
from typing import Dict, Any, Optional
from config.settings import settings

@dataclass
class RiskSettingsSchema:
    # Position Caps
    max_position_size_pct: float  # Maximum % weight of portfolio for any single position
    max_sector_exposure_pct: float # Maximum % weight of portfolio for any single sector
    max_leverage: float           # Maximum gross leverage (e.g., 1.0 or 1.5)
    
    # Drawdown Thresholds
    max_daily_drawdown_pct: float  # Threshold for daily drawdown breach
    max_weekly_drawdown_pct: float # Threshold for weekly drawdown breach
    max_portfolio_drawdown_pct: float # Threshold for peak-to-trough absolute portfolio drawdown
    
    # Kill-Switch / Halt Levels
    emergency_kill_switch_enabled: bool # Whether kill switch is enabled
    consecutive_loss_limit: int          # Max consecutive losing trades before trading halt
    max_slippage_bps: int                # Maximum acceptable slippage in basis points per trade

def load_risk_settings() -> RiskSettingsSchema:
    """Loads risk settings from the global settings object under the [risk] section."""
    return RiskSettingsSchema(
        max_position_size_pct=float(settings.get("risk.max_position_size_pct", 0.05)),
        max_sector_exposure_pct=float(settings.get("risk.max_sector_exposure_pct", 0.20)),
        max_leverage=float(settings.get("risk.max_leverage", 1.0)),
        
        max_daily_drawdown_pct=float(settings.get("risk.max_daily_drawdown_pct", 0.02)),
        max_weekly_drawdown_pct=float(settings.get("risk.max_weekly_drawdown_pct", 0.05)),
        max_portfolio_drawdown_pct=float(settings.get("risk.max_portfolio_drawdown_pct", 0.10)),
        
        emergency_kill_switch_enabled=bool(settings.get("risk.emergency_kill_switch_enabled", True)),
        consecutive_loss_limit=int(settings.get("risk.consecutive_loss_limit", 5)),
        max_slippage_bps=int(settings.get("risk.max_slippage_bps", 50))
    )
