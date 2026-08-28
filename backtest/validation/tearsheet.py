"""
Backtest Tearsheet Report Generator (P3 / L7).

Produces structured performance summaries including equity metrics, risk,
statistical significance (PSR/DSR/Monte Carlo), and trade statistics.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from core.logger import get_logger

logger = get_logger("backtest.validation.tearsheet")


def generate_tearsheet(
    metrics: Dict[str, Any],
    psr: float,
    dsr: float,
    mc_p_value: float,
    var_95: float,
    cvar_95: float,
    output_filepath: Optional[str] = None,
) -> str:
    """
    Generate a formatted Markdown tearsheet report for a backtest run.
    
    Args:
        metrics: Dict returned by SimpleBacktest.run()
        psr: Probabilistic Sharpe Ratio
        dsr: Deflated Sharpe Ratio
        mc_p_value: Monte Carlo permutation test p-value
        var_95: 95% VaR
        cvar_95: 95% CVaR
        output_filepath: Optional filepath to save the markdown report.
        
    Returns:
        Formatted markdown string.
    """
    initial_cap = metrics.get('initial_capital', 1000000.0)
    final_eq = metrics.get('final_equity', initial_cap)
    tot_ret = metrics.get('total_return', 0.0)
    sharpe = metrics.get('sharpe', 0.0)
    max_dd = metrics.get('max_drawdown', 0.0)
    n_trades = metrics.get('n_trades', 0)
    
    report = []
    report.append("# ALLADIN V1 — Backtest Performance Tearsheet\n")
    report.append("## Executive Summary\n")
    report.append(f"- **Initial Capital**: Rs {initial_cap:,.2f}")
    report.append(f"- **Final Equity**:   Rs {final_eq:,.2f}")
    report.append(f"- **Total Return**:   {tot_ret:.2%}")
    report.append(f"- **Sharpe Ratio**:   {sharpe:.2f}")
    report.append(f"- **Max Drawdown**:   {max_dd:.2%}")
    report.append(f"- **Total Trades**:   {n_trades}\n")
    
    report.append("## Statistical Validation\n")
    report.append(f"- **Probabilistic Sharpe Ratio (PSR)**: {psr:.4f} " + ("(Statistically Significant > 0.95)" if psr > 0.95 else "(Not Significant)"))
    report.append(f"- **Deflated Sharpe Ratio (DSR)**:     {dsr:.4f}")
    report.append(f"- **Monte Carlo p-value**:            {mc_p_value:.4f}\n")
    
    report.append("## Risk Profile\n")
    report.append(f"- **Value at Risk (95% VaR)**:        {var_95:.4%}")
    report.append(f"- **Conditional VaR (95% CVaR)**:     {cvar_95:.4%}\n")
    
    report_text = "\n".join(report)
    
    if output_filepath:
        try:
            with open(output_filepath, "w", encoding="utf-8") as f:
                f.write(report_text)
            logger.info(f"Tearsheet report saved to {output_filepath}")
        except Exception as e:
            logger.error(f"Failed to save tearsheet: {e}")
            
    return report_text
