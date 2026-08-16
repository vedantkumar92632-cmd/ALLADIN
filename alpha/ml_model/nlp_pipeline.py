"""
NLP pipeline: raw text -> VADER gate -> FinBERT -> NER -> FAISS.

Per L3.S014: Wire S010->S011->S012->S013 into one small pipeline function.
Text that fails the VADER gate (|compound| < 0.05) skips the expensive
FinBERT/NER/FAISS stages entirely — this is the compute gate.

Flow:
1. VADER gate: fast pre-screen (<1ms). Filters neutral noise.
2. FinBERT: sentiment scoring (positive/negative/neutral + confidence).
3. Financial NER: extract tickers, indices, regulators, events.
4. FAISS: find historically similar headlines (if index is populated).
"""
from typing import List, Dict, Any, Optional
from core.logger import get_logger
from alpha.ml_model.vader_gate import vader_gate
from alpha.ml_model.financial_ner import extract_financial_entities

logger = get_logger("alpha.ml_model.nlp_pipeline")


def process_headline(
    text: str,
    faiss_index=None,
    get_embedding=None,
    vader_threshold: float = 0.05,
    skip_finbert: bool = False,
) -> Dict[str, Any]:
    """
    Process a single headline through the full NLP pipeline.
    
    Args:
        text: Raw headline text.
        faiss_index: Optional FAISSIndex instance for similarity search.
        get_embedding: Optional callable that returns a numpy embedding for text.
        vader_threshold: VADER compound threshold for gate.
        skip_finbert: If True, skip FinBERT (useful when model not downloaded yet).
        
    Returns:
        Dict with keys:
            'text': original text
            'vader_passed': bool
            'vader_compound': float
            'finbert_label': str or None (if gated out)
            'finbert_score': float or None
            'finbert_numeric': float or None
            'entities': dict of extracted entities
            'similar_events': list of similar historical events (if FAISS available)
    """
    result = {
        'text': text,
        'vader_passed': False,
        'vader_compound': 0.0,
        'finbert_label': None,
        'finbert_score': None,
        'finbert_numeric': None,
        'entities': {'tickers': [], 'indices': [], 'regulators': [], 'events': [], 'spacy_entities': []},
        'similar_events': [],
    }
    
    # Step 1: VADER gate
    passes, compound = vader_gate(text, threshold=vader_threshold)
    result['vader_passed'] = passes
    result['vader_compound'] = compound
    
    if not passes:
        # Gated out — skip expensive stages
        return result
    
    # Step 2: FinBERT sentiment
    if not skip_finbert:
        try:
            from alpha.ml_model.finbert_sentiment import finbert_sentiment, finbert_numeric_score
            fb_result = finbert_sentiment(text)
            result['finbert_label'] = fb_result['label']
            result['finbert_score'] = fb_result['score']
            result['finbert_numeric'] = finbert_numeric_score(text)
        except Exception as e:
            logger.warning(f"FinBERT inference failed: {e}")
            result['finbert_label'] = 'error'
            result['finbert_score'] = 0.0
            result['finbert_numeric'] = 0.0
    
    # Step 3: Financial NER
    result['entities'] = extract_financial_entities(text)
    
    # Step 4: FAISS similarity search (if index and embedding function available)
    if faiss_index is not None and get_embedding is not None:
        try:
            embedding = get_embedding(text)
            similar = faiss_index.search(embedding)
            result['similar_events'] = [
                {'score': score, 'metadata': meta}
                for score, meta in similar
            ]
        except Exception as e:
            logger.warning(f"FAISS search failed: {e}")
    
    return result


def process_headlines_batch(
    texts: List[str],
    faiss_index=None,
    get_embedding=None,
    vader_threshold: float = 0.05,
    skip_finbert: bool = False,
) -> List[Dict[str, Any]]:
    """
    Process a batch of headlines through the NLP pipeline.
    
    Returns:
        List of result dicts (same format as process_headline).
    """
    results = []
    passed_count = 0
    
    for text in texts:
        r = process_headline(
            text,
            faiss_index=faiss_index,
            get_embedding=get_embedding,
            vader_threshold=vader_threshold,
            skip_finbert=skip_finbert,
        )
        results.append(r)
        if r['vader_passed']:
            passed_count += 1
    
    logger.info(
        f"NLP pipeline: processed {len(texts)} headlines, "
        f"{passed_count} passed VADER gate ({passed_count/max(len(texts),1)*100:.0f}%)"
    )
    return results
