"""
Financial Named Entity Recognition using spaCy.

Off-the-shelf spaCy NER does NOT recognize NSE tickers, Indian regulators,
or financial events. This module provides a rule-based + pattern matcher
approach for Indian financial entities until a fine-tuned model is available.

Per Part 3A Domain 5: Without entity resolution, a bearish headline about
HDFC Life Insurance will be tagged as a sell signal for HDFC Bank — fatal error.

Custom entity types:
- ORG_NSE: NSE-listed company names and tickers
- INDEX_NSE: Index names (Nifty 50, Sensex, Bank Nifty)
- REGULATOR_IN: Indian regulators (SEBI, RBI, IRDAI)
- MACRO_EVENT: Financial events (Q3 results, budget, rate cut)

Resources:
- spaCy official fine-tuning docs: spacy.io/usage/training
- ML4T Chapter 16: Financial NER fine-tuning
"""
import re
from typing import List, Dict, Optional, Set
from core.logger import get_logger

logger = get_logger("alpha.ml_model.financial_ner")

# Lazy-loaded spaCy model
_nlp = None

# Known Indian financial entities (rule-based baseline)
# These will be expanded with a fine-tuned model in production
NSE_TICKERS: Set[str] = {
    'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN', 'BHARTIARTL',
    'ITC', 'KOTAKBANK', 'LT', 'HCLTECH', 'AXISBANK', 'MARUTI', 'SUNPHARMA',
    'TITAN', 'BAJFINANCE', 'WIPRO', 'ASIANPAINT', 'ULTRACEMCO', 'ONGC',
    'NTPC', 'POWERGRID', 'TATAMOTORS', 'TATASTEEL', 'HINDUNILVR', 'COALINDIA',
    'ADANIENT', 'ADANIPORTS', 'BAJAJFINSV', 'DRREDDY', 'NESTLEIND', 'CIPLA',
    'DIVISLAB', 'HEROMOTOCO', 'TECHM', 'GRASIM', 'INDUSINDBK', 'APOLLOHOSP',
    'EICHERMOT', 'BPCL', 'JSWSTEEL', 'M&M', 'BRITANNIA', 'HINDALCO',
    'SBILIFE', 'HDFCLIFE', 'TATACONSUM', 'UPL',
}

NSE_COMPANY_NAMES: Dict[str, str] = {
    'reliance industries': 'RELIANCE', 'reliance': 'RELIANCE',
    'tata consultancy': 'TCS', 'tcs': 'TCS',
    'infosys': 'INFY', 'hdfc bank': 'HDFCBANK',
    'icici bank': 'ICICIBANK', 'state bank': 'SBIN', 'sbi': 'SBIN',
    'bharti airtel': 'BHARTIARTL', 'airtel': 'BHARTIARTL',
    'itc': 'ITC', 'kotak mahindra': 'KOTAKBANK', 'kotak bank': 'KOTAKBANK',
    'larsen': 'LT', 'l&t': 'LT', 'hcl tech': 'HCLTECH',
    'axis bank': 'AXISBANK', 'maruti suzuki': 'MARUTI', 'maruti': 'MARUTI',
    'sun pharma': 'SUNPHARMA', 'titan': 'TITAN',
    'bajaj finance': 'BAJFINANCE', 'wipro': 'WIPRO',
    'asian paints': 'ASIANPAINT', 'ongc': 'ONGC',
    'tata motors': 'TATAMOTORS', 'tata steel': 'TATASTEEL',
    'hindustan unilever': 'HINDUNILVR', 'hul': 'HINDUNILVR',
    'adani enterprises': 'ADANIENT', 'adani ports': 'ADANIPORTS',
    'dr reddy': 'DRREDDY', 'cipla': 'CIPLA',
    'hdfc life': 'HDFCLIFE', 'sbi life': 'SBILIFE',
}

INDEX_NAMES: Set[str] = {
    'nifty 50', 'nifty50', 'sensex', 'bank nifty', 'banknifty',
    'nifty it', 'nifty pharma', 'nifty auto', 'nifty metal',
    'nifty fmcg', 'nifty realty', 'nifty midcap', 'nifty smallcap',
    'bse 500', 'nifty next 50',
}

REGULATORS: Set[str] = {
    'sebi', 'rbi', 'irdai', 'pfrda', 'nabard', 'sidbi',
    'reserve bank', 'securities and exchange board',
}

MACRO_EVENTS: Set[str] = {
    'q1 results', 'q2 results', 'q3 results', 'q4 results',
    'quarterly results', 'annual results', 'budget', 'union budget',
    'rate cut', 'rate hike', 'repo rate', 'monetary policy',
    'gdp growth', 'inflation', 'cpi', 'wpi', 'iip',
    'ipo', 'fpo', 'buyback', 'dividend', 'stock split',
    'bonus issue', 'rights issue', 'merger', 'acquisition',
    'demerger', 'delisting', 'listing',
}


def _get_nlp():
    """Lazy-load spaCy model."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy en_core_web_sm loaded.")
        except (OSError, ImportError):
            logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            import spacy
            _nlp = spacy.blank("en")
            logger.info("Using blank spaCy model (rule-based NER only).")
    return _nlp


def extract_financial_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract financial entities from text using rule-based matching.
    
    Returns:
        Dict with keys: 'tickers', 'indices', 'regulators', 'events', 'spacy_entities'
    """
    text_lower = text.lower()
    
    entities = {
        'tickers': [],
        'indices': [],
        'regulators': [],
        'events': [],
        'spacy_entities': [],
    }
    
    # Match NSE tickers (exact word match in uppercase)
    for word in text.split():
        clean_word = word.strip('.,;:!?()[]"\'').upper()
        if clean_word in NSE_TICKERS:
            if clean_word not in entities['tickers']:
                entities['tickers'].append(clean_word)
    
    # Match company names (word boundary check for short names)
    for company_name, ticker in NSE_COMPANY_NAMES.items():
        if len(company_name) <= 5:
            # Short names need word boundary matching to avoid false positives
            if re.search(rf'\b{re.escape(company_name)}\b', text_lower):
                if ticker not in entities['tickers']:
                    entities['tickers'].append(ticker)
        else:
            if company_name in text_lower:
                if ticker not in entities['tickers']:
                    entities['tickers'].append(ticker)
    
    # Match index names
    for idx_name in INDEX_NAMES:
        if idx_name in text_lower:
            entities['indices'].append(idx_name.upper())
    
    # Match regulators
    for reg in REGULATORS:
        if reg in text_lower:
            entities['regulators'].append(reg.upper())
    
    # Match macro events
    for event in MACRO_EVENTS:
        if event in text_lower:
            entities['events'].append(event)
    
    # Also run spaCy NER for general entities (ORG, PERSON, etc.)
    nlp = _get_nlp()
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ('ORG', 'PERSON', 'GPE', 'MONEY', 'PERCENT'):
            entities['spacy_entities'].append({
                'text': ent.text,
                'label': ent.label_,
            })
    
    return entities


def resolve_ticker(text: str) -> Optional[str]:
    """
    Resolve text to a single NSE ticker, if possible.
    Returns None if no ticker can be resolved.
    """
    entities = extract_financial_entities(text)
    if entities['tickers']:
        return entities['tickers'][0]
    return None
