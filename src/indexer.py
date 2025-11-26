# indexer.py
# FAISS/Chroma build & load

import os
import pickle
import pandas as pd
import numpy as np
import faiss
from typing import List, Dict, Any
from pathlib import Path
import asyncio

from embedder import get_embedder
from config import (
    FACTS_CSV, KB_DIR, INDEX_DIR, 
    FAISS_INDEX_PATH, METADATA_PATH, EMBEDDING_DIMENSIONS
)


class Document:
    """Represents a document with metadata for retrieval."""
    
    def __init__(
        self, 
        text: str, 
        source_id: str, 
        source_type: str, 
        source_name: str, 
        date: str = None,
        metadata: Dict[str, Any] = None
    ):
        self.text = text
        self.source_id = source_id
        self.source_type = source_type  # "PIB_FACT" or "KB_DOCUMENT"
        self.source_name = source_name
        self.date = date
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "date": self.date,
            "metadata": self.metadata
        }


class VectorStore:
    """FAISS-based vector store with metadata tracking."""
    
    def __init__(self):
        self.index = None
        self.documents: List[Document] = []
        self.embedder = get_embedder()
    
    def load_facts_csv(self) -> List[Document]:
        """Load PIB facts from CSV file."""
        print(f"Loading facts from {FACTS_CSV}...")
        # Use quotechar to handle commas in text
        df = pd.read_csv(
            FACTS_CSV, 
            header=None, 
            names=["id", "source", "date", "text"],
            on_bad_lines='skip'  # Skip malformed lines
        )
        
        documents = []
        for _, row in df.iterrows():
            if pd.notna(row["text"]) and str(row["text"]).strip():
                doc = Document(
                    text=str(row["text"]).strip(),
                    source_id=f"PIB_{row['id']}",
                    source_type="PIB_FACT",
                    source_name=row["source"],
                    date=str(row["date"]) if pd.notna(row["date"]) else None,
                    metadata={"fact_id": int(row["id"])}
                )
                documents.append(doc)
        
        print(f"Loaded {len(documents)} facts from CSV")
        return documents
    
    def load_kb_documents(self) -> List[Document]:
        """Load knowledge base documents from KB directory."""
        print(f"Loading KB documents from {KB_DIR}...")
        documents = []
        
        if not os.path.exists(KB_DIR):
            print(f"KB directory not found: {KB_DIR}")
            return documents
        
        kb_files = list(Path(KB_DIR).glob("*.txt"))
        for idx, kb_file in enumerate(kb_files, start=1):
            try:
                with open(kb_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                if text:
                    doc = Document(
                        text=text,
                        source_id=f"KB_{idx}",
                        source_type="KB_DOCUMENT",
                        source_name=kb_file.stem,
                        metadata={"filename": kb_file.name}
                    )
                    documents.append(doc)
            except Exception as e:
                print(f"Error loading {kb_file}: {e}")
        
        print(f"Loaded {len(documents)} KB documents")
        return documents
    
    async def build_index(self):
        """Build FAISS index from all documents."""
        print("\n=== Building Vector Index ===")
        
        # Load all documents
        self.documents = []
        self.documents.extend(self.load_facts_csv())
        self.documents.extend(self.load_kb_documents())
        
        if not self.documents:
            raise ValueError("No documents loaded!")
        
        print(f"\nTotal documents: {len(self.documents)}")
        
        # Extract texts for embedding
        texts = [doc.text for doc in self.documents]
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = await self.embedder.embed_batch(texts)
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Build FAISS index
        print("Building FAISS index...")
        self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)  # Inner product (cosine similarity)
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings_array)
        self.index.add(embeddings_array)
        
        print(f"Index built with {self.index.ntotal} vectors")
    
    def save(self):
        """Save index and metadata to disk."""
        os.makedirs(INDEX_DIR, exist_ok=True)
        
        print(f"\nSaving index to {FAISS_INDEX_PATH}...")
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        
        print(f"Saving metadata to {METADATA_PATH}...")
        metadata = [doc.to_dict() for doc in self.documents]
        with open(METADATA_PATH, 'wb') as f:
            pickle.dump(metadata, f)
        
        print("✓ Index and metadata saved successfully")
    
    def load(self):
        """Load index and metadata from disk."""
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
            raise FileNotFoundError("Index files not found. Please build the index first.")
        
        print(f"Loading index from {FAISS_INDEX_PATH}...")
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        
        print(f"Loading metadata from {METADATA_PATH}...")
        with open(METADATA_PATH, 'rb') as f:
            metadata = pickle.load(f)
        
        self.documents = [
            Document(
                text=m["text"],
                source_id=m["source_id"],
                source_type=m["source_type"],
                source_name=m["source_name"],
                date=m.get("date"),
                metadata=m.get("metadata", {})
            )
            for m in metadata
        ]
        
        print(f"✓ Loaded {len(self.documents)} documents")


# Singleton instance
_vector_store_instance = None

def get_vector_store() -> VectorStore:
    """Get or create the global vector store instance."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance


async def main():
    """Build and save the vector index."""
    store = get_vector_store()
    await store.build_index()
    store.save()
    print("\n✓ Indexing complete!")


if __name__ == "__main__":
    asyncio.run(main())

