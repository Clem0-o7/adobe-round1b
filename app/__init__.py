"""
Persona-Driven Document Intelligence RAG Pipeline

A CPU-optimized system for extracting and prioritizing document sections
based on user personas and their specific job requirements.
"""

__version__ = "1.0.0"
__author__ = "Adobe Hackathon Team"
__description__ = "Persona-Driven Document Intelligence RAG Pipeline"

# Import main components for easy access
from .extract import PDFExtractor, extract_documents
from .embed import DocumentEmbedder, create_document_index
from .retrieve import DocumentRetriever, retrieve_relevant_content
from .generate import DocumentGenerator, generate_final_response
from .utils import (
    load_input_files, save_output, validate_input,
    estimate_processing_time, log_system_info
)

# Import advanced heading detection components
from .pdf_processor import PDFProcessor, PageData, TextSpan
from .heading_detector import HeadingDetector, HeadingCandidate

__all__ = [
    'PDFExtractor', 'extract_documents',
    'DocumentEmbedder', 'create_document_index',
    'DocumentRetriever', 'retrieve_relevant_content',
    'DocumentGenerator', 'generate_final_response',
    'load_input_files', 'save_output', 'validate_input',
    'estimate_processing_time', 'log_system_info',
    'PDFProcessor', 'PageData', 'TextSpan',
    'HeadingDetector', 'HeadingCandidate'
]
