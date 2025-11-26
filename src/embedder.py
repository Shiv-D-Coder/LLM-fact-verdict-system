# embedder.py
# embedding wrapper

import asyncio
from typing import List, Union
from openai import AsyncOpenAI
import numpy as np
from config import OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS


class Embedder:
    """Wrapper for OpenAI embeddings with async support and batching."""
    
    def __init__(self, model: str = EMBEDDING_MODEL):
        self.model = model
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.dimensions = EMBEDDING_DIMENSIONS
    
    async def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        embeddings = await self.embed_batch([text])
        return embeddings[0]
    
    async def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[np.ndarray]:
        """
        Embed multiple texts with batching for efficiency.
        
        Args:
            texts: List of text strings to embed
            batch_size: Maximum batch size for API calls
            
        Returns:
            List of embedding vectors as numpy arrays
        """
        all_embeddings = []
        
        # Process in batches to avoid API limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                batch_embeddings = [np.array(item.embedding) for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error embedding batch {i//batch_size + 1}: {e}")
                # Return zero vectors for failed batches
                all_embeddings.extend([np.zeros(self.dimensions) for _ in batch])
        
        return all_embeddings
    
    def embed_text_sync(self, text: str) -> np.ndarray:
        """Synchronous wrapper for single text embedding."""
        return asyncio.run(self.embed_text(text))
    
    def embed_batch_sync(self, texts: List[str]) -> List[np.ndarray]:
        """Synchronous wrapper for batch embedding."""
        return asyncio.run(self.embed_batch(texts))


# Singleton instance
_embedder_instance = None

def get_embedder() -> Embedder:
    """Get or create the global embedder instance."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = Embedder()
    return _embedder_instance

