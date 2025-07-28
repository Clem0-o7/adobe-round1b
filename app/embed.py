"""
Embedding and FAISS Indexing Module
Creates embeddings using sentence-transformers and builds FAISS index for fast similarity search.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentEmbedder:
    def __init__(self, model_name: str = "distiluse-base-multilingual-cased-v1"):
        """
        Initialize the embedder with a sentence transformer model.
        
        Args:
            model_name (str): Name of the sentence-transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.index = None
        self.sections_metadata = []
        self.embedding_dim = 512  # Default dimension for distiluse models
        
    def load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def create_embeddings(self, documents: List[Dict]) -> np.ndarray:
        """
        Create embeddings for all document sections.
        
        Args:
            documents (List[Dict]): List of document data from PDF extraction
            
        Returns:
            np.ndarray: Array of embeddings
        """
        if not self.model:
            self.load_model()
        
        # Prepare text chunks and metadata
        text_chunks = []
        self.sections_metadata = []
        
        for doc in documents:
            for section in doc['sections']:
                # Combine section title and content for better context
                text_to_embed = f"{section['section_title']} {section['content']}"
                text_chunks.append(text_to_embed)
                
                # Store metadata for retrieval
                self.sections_metadata.append({
                    'document': section['document'],
                    'page_number': section['page_number'],
                    'section_title': section['section_title'],
                    'content': section['content'],
                    'word_count': section.get('word_count', 0),
                    'full_text': text_to_embed
                })
        
        logger.info(f"Creating embeddings for {len(text_chunks)} sections")
        
        # Create embeddings in batches for memory efficiency
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        logger.info(f"Created embeddings with shape: {embeddings.shape}")
        
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build FAISS index for fast similarity search.
        
        Args:
            embeddings (np.ndarray): Array of embeddings
            
        Returns:
            faiss.Index: FAISS index for similarity search
        """
        logger.info("Building FAISS index")
        
        # Use flat index for exact search (good for small to medium datasets)
        # For larger datasets, consider using HNSW: faiss.IndexHNSWFlat(embedding_dim, 32)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        index.add(embeddings.astype('float32'))
        
        logger.info(f"FAISS index built with {index.ntotal} vectors")
        
        self.index = index
        return index
    
    def setup_index(self, documents: List[Dict]) -> Tuple[faiss.Index, List[Dict]]:
        """
        Complete setup: create embeddings and build index.
        
        Args:
            documents (List[Dict]): List of document data from PDF extraction
            
        Returns:
            Tuple[faiss.Index, List[Dict]]: FAISS index and sections metadata
        """
        embeddings = self.create_embeddings(documents)
        index = self.build_faiss_index(embeddings)
        
        return index, self.sections_metadata
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a query string into an embedding vector.
        
        Args:
            query (str): Query string to encode
            
        Returns:
            np.ndarray: Query embedding vector
        """
        if not self.model:
            self.load_model()
        
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        return query_embedding
    
    def save_index(self, index_path: str, metadata_path: str):
        """
        Save FAISS index and metadata to disk.
        
        Args:
            index_path (str): Path to save FAISS index
            metadata_path (str): Path to save metadata
        """
        if self.index:
            faiss.write_index(self.index, index_path)
            
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.sections_metadata, f)
        
        logger.info(f"Index and metadata saved to {index_path} and {metadata_path}")
    
    def load_index(self, index_path: str, metadata_path: str):
        """
        Load FAISS index and metadata from disk.
        
        Args:
            index_path (str): Path to FAISS index file
            metadata_path (str): Path to metadata file
        """
        self.index = faiss.read_index(index_path)
        
        with open(metadata_path, 'rb') as f:
            self.sections_metadata = pickle.load(f)
        
        logger.info(f"Index and metadata loaded from {index_path} and {metadata_path}")

# Helper functions for easy import
def create_document_index(documents: List[Dict], model_name: str = "distiluse-base-multilingual-cased-v1") -> Tuple[DocumentEmbedder, faiss.Index, List[Dict]]:
    """
    Convenience function to create document index.
    
    Args:
        documents (List[Dict]): List of document data from PDF extraction
        model_name (str): Name of the sentence-transformer model to use
        
    Returns:
        Tuple[DocumentEmbedder, faiss.Index, List[Dict]]: Embedder, index, and metadata
    """
    embedder = DocumentEmbedder(model_name)
    index, metadata = embedder.setup_index(documents)
    
    return embedder, index, metadata
