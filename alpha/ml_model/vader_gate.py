"""
VADER pre-screen gate for NLP pipeline.

VADER is a PRE-SCREEN GATE ONLY — filters non-financial text before the
expensive FinBERT call. NOT a sentiment score to trade on directly.
Part 2 gives it Value=4, deliberately low.

VADER requires zero tuning. Runs in <1ms per document.
Threshold: |compound| < 0.05 -> skip FinBERT (document is neutral noise).

Resources:
- VADER: github.com/cjhutto/vaderSentiment
"""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List, Tuple
from core.logger import get_logger

logger = get_logger("alpha.ml_model.vader_gate")

# Singleton analyzer (stateless, safe to share)
_analyzer = SentimentIntensityAnalyzer()

# Threshold for "financially relevant" — documents with |compound| < this skip FinBERT
COMPOUND_THRESHOLD = 0.05


def vader_gate(text: str, threshold: float = COMPOUND_THRESHOLD) -> Tuple[bool, float]:
    """
    Pre-screen a single document through VADER.
    
    Args:
        text: Raw text to screen.
        threshold: Minimum |compound| score to pass through to FinBERT.
        
    Returns:
        Tuple of (passes_gate: bool, compound_score: float).
        passes_gate=True means the text has enough sentiment signal to
        warrant the expensive FinBERT inference.
    """
    if not isinstance(text, str) or not text.strip(): return False, 0.0
    scores = _analyzer.polarity_scores(text)
    compound = scores['compound']
    passes = abs(compound) >= threshold
    return passes, compound


def vader_gate_batch(texts: List[str], threshold: float = COMPOUND_THRESHOLD) -> List[Tuple[bool, float]]:
    """
    Pre-screen a batch of documents through VADER.
    
    Returns:
        List of (passes_gate, compound_score) tuples.
    """
    results = []
    passed_count = 0
    for text in texts:
        passes, compound = vader_gate(text, threshold)
        results.append((passes, compound))
        if passes:
            passed_count += 1
    
    logger.info(f"VADER gate: {passed_count}/{len(texts)} documents passed (threshold={threshold})")
    return results
