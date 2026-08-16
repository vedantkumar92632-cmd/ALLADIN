"""
FAISS similarity search over historical news/events.

Provides semantic similarity search to find historically similar events
and their market impact. Used in the NLP pipeline after FinBERT + NER
to find precedents for current headlines.

Per Part 3B Domain 5: Semantic similarity search over historical news/events.

Resources:
- FAISS: github.com/facebookresearch/faiss
"""
import numpy as np
import faiss
from typing import List, Optional, Tuple, Dict, Any
from core.logger import get_logger

logger = get_logger("alpha.ml_model.similarity_search")


class FAISSIndex:
    """
    FAISS-based semantic similarity search for financial news/events.
    
    Stores embeddings of historical headlines/events and retrieves
    the most similar historical events for a new headline.
    """

    def __init__(self, dimension: int = 768, n_results: int = 5):
        """
        Args:
            dimension: Embedding dimension (768 for FinBERT/BERT-base).
            n_results: Default number of similar results to return.
        """
        self.dimension = dimension
        self.n_results = n_results
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine after L2 norm)
        self._metadata: List[Dict[str, Any]] = []  # Parallel metadata for each vector
        self._is_built = False
        
        logger.info(f"FAISSIndex initialized: dim={dimension}, n_results={n_results}")

    def add_vectors(
        self,
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add embedding vectors to the index.
        
        Args:
            embeddings: (N, dimension) array of embeddings.
            metadata: Optional list of dicts with info about each vector
                     (e.g., {'text': '...', 'date': '...', 'ticker': '...', 'impact': 0.02}).
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        # L2-normalize for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = (embeddings / norms).astype(np.float32)
        
        self.index.add(normalized)
        
        if metadata:
            if isinstance(metadata, dict):
                metadata = [metadata]
            self._metadata.extend(metadata)
        else:
            # Fill with placeholder metadata
            for i in range(len(embeddings)):
                self._metadata.append({'index': len(self._metadata)})
        
        self._is_built = True
        logger.info(f"Added {len(embeddings)} vectors. Total: {self.index.ntotal}")

    def search(
        self,
        query_embedding: np.ndarray,
        n_results: Optional[int] = None
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Find most similar historical events.
        
        Args:
            query_embedding: (dimension,) or (1, dimension) query vector.
            n_results: Number of results to return (default: self.n_results).
            
        Returns:
            List of (similarity_score, metadata_dict) tuples, sorted by similarity.
        """
        if not self._is_built:
            logger.warning("FAISS index is empty. No results to return.")
            return []
        
        if n_results is None:
            n_results = self.n_results
        
        n_results = min(n_results, self.index.ntotal)
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # L2-normalize query
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        query_embedding = query_embedding.astype(np.float32)
        
        distances, indices = self.index.search(query_embedding, n_results)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self._metadata):
                results.append((float(dist), self._metadata[idx]))
        
        return results

    def search_batch(
        self,
        query_embeddings: np.ndarray,
        n_results: Optional[int] = None
    ) -> List[List[Tuple[float, Dict[str, Any]]]]:
        """
        Batch similarity search.
        
        Args:
            query_embeddings: (N, dimension) array of query vectors.
            
        Returns:
            List of result lists, one per query.
        """
        if not self._is_built:
            return [[] for _ in range(len(query_embeddings))]
        
        if n_results is None:
            n_results = self.n_results
        n_results = min(n_results, self.index.ntotal)
        
        # L2-normalize queries
        norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = (query_embeddings / norms).astype(np.float32)
        
        distances, indices = self.index.search(normalized, n_results)
        
        all_results = []
        for dists, idxs in zip(distances, indices):
            results = []
            for dist, idx in zip(dists, idxs):
                if idx >= 0 and idx < len(self._metadata):
                    results.append((float(dist), self._metadata[idx]))
            all_results.append(results)
        
        return all_results

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self.index.ntotal

    def clear(self) -> None:
        """Reset the index."""
        self.index.reset()
        self._metadata.clear()
        self._is_built = False
        logger.info("FAISS index cleared.")
