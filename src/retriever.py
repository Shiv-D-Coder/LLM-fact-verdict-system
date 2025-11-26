# retriever.py
# retrieve top-k

import numpy as np
import faiss
from typing import List, Dict, Any
import asyncio

from embedder import get_embedder
from indexer import get_vector_store, Document
from config import TOP_K_RETRIEVAL, SIMILARITY_THRESHOLD


class RetrievalResult:
    """Represents a single retrieval result with metadata."""
    
    def __init__(self, document: Document, similarity_score: float, rank: int):
        self.document = document
        self.similarity_score = similarity_score
        self.rank = rank
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "similarity_score": float(self.similarity_score),
            "source_id": self.document.source_id,
            "source_type": self.document.source_type,
            "source_name": self.document.source_name,
            "date": self.document.date,
            "text": self.document.text,
            "metadata": self.document.metadata
        }


class HybridRetriever:
    """Hybrid retrieval combining vector similarity and keyword matching."""
    
    def __init__(self):
        self.vector_store = get_vector_store()
        self.embedder = get_embedder()
        self.is_loaded = False
    
    def load(self):
        """Load the vector store index."""
        if not self.is_loaded:
            self.vector_store.load()
            self.is_loaded = True
    
    async def retrieve(
        self, 
        query: str, 
        top_k: int = TOP_K_RETRIEVAL,
        keywords: List[str] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve top-k most relevant documents using hybrid search.
        
        Args:
            query: The search query/claim
            top_k: Number of results to return
            keywords: Optional list of keywords for boosting
            
        Returns:
            List of RetrievalResult objects with metadata
        """
        self.load()
        
        # Generate query embedding
        query_embedding = await self.embedder.embed_text(query)
        query_vector = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_vector)
        
        # Perform vector search (get more than top_k for reranking)
        search_k = min(top_k * 3, len(self.vector_store.documents))
        distances, indices = self.vector_store.index.search(query_vector, search_k)
        
        # Create initial results
        results = []
        for rank, (idx, distance) in enumerate(zip(indices[0], distances[0]), start=1):
            if idx < len(self.vector_store.documents):
                doc = self.vector_store.documents[idx]
                similarity = float(distance)  # Already cosine similarity due to normalization
                
                # Apply keyword boosting
                if keywords:
                    boost = self._calculate_keyword_boost(doc.text, keywords)
                    similarity = similarity * (1 + boost * 0.2)  # Up to 20% boost
                
                if similarity >= SIMILARITY_THRESHOLD:
                    results.append(RetrievalResult(doc, similarity, rank))
        
        # Sort by boosted similarity and take top_k
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        results = results[:top_k]
        
        # Update ranks after reranking
        for rank, result in enumerate(results, start=1):
            result.rank = rank
        
        return results
    
    def _calculate_keyword_boost(self, text: str, keywords: List[str]) -> float:
        """
        Calculate boost score based on keyword matches.
        
        Returns:
            Boost factor between 0 and 1
        """
        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw.lower() in text_lower)
        return min(matches / max(len(keywords), 1), 1.0)
    
    def retrieve_sync(self, query: str, top_k: int = TOP_K_RETRIEVAL, keywords: List[str] = None) -> List[RetrievalResult]:
        """Synchronous wrapper for retrieve."""
        return asyncio.run(self.retrieve(query, top_k, keywords))


# Singleton instance
_retriever_instance = None

def get_retriever() -> HybridRetriever:
    """Get or create the global retriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
    return _retriever_instance


async def main():
    """Test the retriever."""
    retriever = get_retriever()
    
    test_query = "The Indian government has announced free electricity to all farmers starting July 2025."
    print(f"Query: {test_query}\n")
    
    results = await retriever.retrieve(test_query, top_k=5)
    
    print(f"Found {len(results)} results:\n")
    for result in results:
        print(f"Rank {result.rank} (Similarity: {result.similarity_score:.3f})")
        print(f"Source: {result.document.source_id} ({result.document.source_type})")
        print(f"Date: {result.document.date}")
        print(f"Text: {result.document.text[:200]}...")
        print()


if __name__ == "__main__":
    asyncio.run(main())

