"""
Query Embedding and FAISS Search Module
Handles query formation, embedding, and retrieval of relevant document sections.
"""

import numpy as np
import faiss
from typing import List, Dict, Tuple
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentRetriever:
    def __init__(self, embedder, index: faiss.Index, sections_metadata: List[Dict]):
        """
        Initialize the retriever with an embedder, FAISS index, and metadata.
        
        Args:
            embedder: DocumentEmbedder instance
            index (faiss.Index): FAISS index for similarity search
            sections_metadata (List[Dict]): Metadata for all sections
        """
        self.embedder = embedder
        self.index = index
        self.sections_metadata = sections_metadata
        
    def create_persona_query(self, persona: str, job_to_be_done: str) -> str:
        """
        Create a search query based on persona and job to be done.
        
        Args:
            persona (str): Description of the persona (role, expertise)
            job_to_be_done (str): Specific task the persona needs to accomplish
            
        Returns:
            str: Formatted query string
        """
        # Combine persona and job into a comprehensive query
        query = f"As a {persona}, I need to {job_to_be_done}. "
        query += f"Find relevant information for {persona} to complete: {job_to_be_done}"
        
        return query
    
    def search_relevant_sections(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Search for relevant sections using semantic similarity.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            
        Returns:
            List[Dict]: List of relevant sections with similarity scores
        """
        # Encode the query
        query_embedding = self.embedder.encode_query(query)
        
        # Search the index
        similarities, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results with metadata
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.sections_metadata):
                section = self.sections_metadata[idx].copy()
                section['similarity_score'] = float(similarity)
                section['rank'] = i + 1
                results.append(section)
        
        logger.info(f"Retrieved {len(results)} relevant sections for query")
        
        return results
    
    def rank_sections_by_relevance(self, sections: List[Dict], persona: str, job_to_be_done: str) -> List[Dict]:
        """
        Re-rank sections based on additional relevance criteria.
        
        Args:
            sections (List[Dict]): Initial search results
            persona (str): Persona description
            job_to_be_done (str): Job to be done description
            
        Returns:
            List[Dict]: Re-ranked sections
        """
        # Additional ranking factors
        persona_keywords = self._extract_keywords(persona.lower())
        job_keywords = self._extract_keywords(job_to_be_done.lower())
        
        for section in sections:
            content_lower = section['content'].lower()
            title_lower = section['section_title'].lower()
            
            # Calculate keyword matches
            persona_matches = sum(1 for kw in persona_keywords if kw in content_lower or kw in title_lower)
            job_matches = sum(1 for kw in job_keywords if kw in content_lower or kw in title_lower)
            
            # Calculate additional relevance score
            keyword_score = (persona_matches * 0.3) + (job_matches * 0.7)
            length_score = min(section.get('word_count', 0) / 100, 1.0)  # Favor substantial content
            
            # Combine with similarity score
            section['relevance_score'] = (
                section['similarity_score'] * 0.6 + 
                keyword_score * 0.3 + 
                length_score * 0.1
            )
        
        # Sort by relevance score
        sections.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Update ranks
        for i, section in enumerate(sections):
            section['importance_rank'] = i + 1
        
        return sections
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract meaningful keywords from text.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of keywords
        """
        # Simple keyword extraction (can be enhanced with NLP libraries)
        import re
        
        # Remove common stop words
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'i', 'me', 'my', 'myself', 'we',
            'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself'
        }
        
        # Extract words (3+ characters, not stop words)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        keywords = [word.lower() for word in words if word.lower() not in stop_words]
        
        return list(set(keywords))  # Remove duplicates
    
    def extract_subsections(self, sections: List[Dict], max_subsections: int = 5) -> List[Dict]:
        """
        Extract and refine subsections from top-ranked sections.
        
        Args:
            sections (List[Dict]): Top-ranked sections
            max_subsections (int): Maximum number of subsections to extract
            
        Returns:
            List[Dict]: Refined subsections
        """
        subsections = []
        
        for section in sections[:max_subsections]:
            # Split content into logical subsections (by sentences or paragraphs)
            content = section['content']
            
            # Simple subsection splitting by sentences
            sentences = self._split_into_sentences(content)
            
            # Group sentences into meaningful chunks
            chunks = self._group_sentences(sentences, chunk_size=3)
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) > 50:  # Only include substantial chunks
                    subsection = {
                        'document': section['document'],
                        'page_number': section['page_number'],
                        'section_title': section['section_title'],
                        'subsection_index': i + 1,
                        'refined_text': chunk.strip(),
                        'parent_rank': section['importance_rank']
                    }
                    subsections.append(subsection)
        
        return subsections
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of sentences
        """
        import re
        
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _group_sentences(self, sentences: List[str], chunk_size: int = 3) -> List[str]:
        """
        Group sentences into chunks.
        
        Args:
            sentences (List[str]): List of sentences
            chunk_size (int): Number of sentences per chunk
            
        Returns:
            List[str]: List of sentence chunks
        """
        chunks = []
        
        for i in range(0, len(sentences), chunk_size):
            chunk = ' '.join(sentences[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def retrieve_for_persona(self, persona: str, job_to_be_done: str, top_k: int = 10) -> Tuple[List[Dict], List[Dict]]:
        """
        Complete retrieval pipeline for a given persona and job.
        
        Args:
            persona (str): Persona description
            job_to_be_done (str): Job to be done description
            top_k (int): Number of top sections to retrieve
            
        Returns:
            Tuple[List[Dict], List[Dict]]: Ranked sections and subsections
        """
        # Create persona-specific query
        query = self.create_persona_query(persona, job_to_be_done)
        logger.info(f"Created query: {query[:100]}...")
        
        # Search for relevant sections
        sections = self.search_relevant_sections(query, top_k)
        
        # Re-rank sections
        ranked_sections = self.rank_sections_by_relevance(sections, persona, job_to_be_done)
        
        # Extract subsections
        subsections = self.extract_subsections(ranked_sections)
        
        logger.info(f"Retrieved {len(ranked_sections)} sections and {len(subsections)} subsections")
        
        return ranked_sections, subsections

# Helper function for easy import
def retrieve_relevant_content(embedder, index: faiss.Index, metadata: List[Dict], 
                            persona: str, job_to_be_done: str, top_k: int = 10) -> Tuple[List[Dict], List[Dict]]:
    """
    Convenience function for content retrieval.
    
    Args:
        embedder: DocumentEmbedder instance
        index (faiss.Index): FAISS index
        metadata (List[Dict]): Sections metadata
        persona (str): Persona description
        job_to_be_done (str): Job to be done description
        top_k (int): Number of top sections to retrieve
        
    Returns:
        Tuple[List[Dict], List[Dict]]: Ranked sections and subsections
    """
    retriever = DocumentRetriever(embedder, index, metadata)
    return retriever.retrieve_for_persona(persona, job_to_be_done, top_k)
