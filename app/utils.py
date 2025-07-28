"""
Utility Functions for the RAG Pipeline
Helper functions for text processing, file handling, and common operations.
"""

import os
import json
import logging
from typing import List, Dict, Tuple, Any
from pathlib import Path
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_input_files(input_dir: str) -> Tuple[List[str], str, str]:
    """
    Load input files from the input directory.
    
    Args:
        input_dir (str): Path to input directory
        
    Returns:
        Tuple[List[str], str, str]: PDF files, persona, and job description
    """
    # Find PDF files
    pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
    
    # Load persona and job description
    persona = ""
    job_to_be_done = ""
    
    # Try to load persona from file
    persona_file = os.path.join(input_dir, "persona.txt")
    if os.path.exists(persona_file):
        with open(persona_file, 'r', encoding='utf-8') as f:
            persona = f.read().strip()
    
    # Try to load job description from file
    job_file = os.path.join(input_dir, "job.txt")
    if os.path.exists(job_file):
        with open(job_file, 'r', encoding='utf-8') as f:
            job_to_be_done = f.read().strip()
    
    # Alternative: load from JSON file
    config_file = os.path.join(input_dir, "config.json")
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                persona = config.get('persona', persona)
                job_to_be_done = config.get('job_to_be_done', job_to_be_done)
        except Exception as e:
            logger.warning(f"Could not load config.json: {str(e)}")
    
    logger.info(f"Loaded {len(pdf_files)} PDF files")
    logger.info(f"Persona: {persona[:100]}...")
    logger.info(f"Job: {job_to_be_done[:100]}...")
    
    return pdf_files, persona, job_to_be_done

def save_output(output_data: Dict, output_dir: str, filename: str = "answer.json"):
    """
    Save the output data to a JSON file.
    
    Args:
        output_data (Dict): Data to save
        output_dir (str): Output directory path
        filename (str): Output filename
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    output_path = os.path.join(output_dir, filename)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Output saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving output: {str(e)}")

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text (str): Text to chunk
        chunk_size (int): Size of each chunk in characters
        overlap (int): Overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to end at a sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 100 characters
            search_start = max(end - 100, start)
            sentence_end = -1
            
            for punct in ['.', '!', '?', '\n']:
                pos = text.rfind(punct, search_start, end)
                if pos > sentence_end:
                    sentence_end = pos
            
            if sentence_end > start:
                end = sentence_end + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks

def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned text
    """
    import re
    
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)\[\]\"\'\/\%\$\@\#]', '', text)
    
    # Remove very long sequences of the same character
    text = re.sub(r'(.)\1{5,}', r'\1\1\1', text)
    
    return text.strip()

def validate_input(pdf_files: List[str], persona: str, job_to_be_done: str) -> bool:
    """
    Validate that all required inputs are provided.
    
    Args:
        pdf_files (List[str]): List of PDF file paths
        persona (str): Persona description
        job_to_be_done (str): Job description
        
    Returns:
        bool: True if valid, False otherwise
    """
    errors = []
    
    if not pdf_files:
        errors.append("No PDF files found")
    
    if not persona or len(persona.strip()) < 10:
        errors.append("Persona description is missing or too short")
    
    if not job_to_be_done or len(job_to_be_done.strip()) < 10:
        errors.append("Job description is missing or too short")
    
    # Check if PDF files exist
    for pdf_file in pdf_files:
        if not os.path.exists(pdf_file):
            errors.append(f"PDF file does not exist: {pdf_file}")
    
    if errors:
        for error in errors:
            logger.error(f"Validation error: {error}")
        return False
    
    return True

def estimate_processing_time(num_documents: int, total_pages: int) -> float:
    """
    Estimate processing time based on document count and pages.
    
    Args:
        num_documents (int): Number of documents
        total_pages (int): Total number of pages
        
    Returns:
        float: Estimated processing time in seconds
    """
    # Rough estimates based on CPU processing
    pdf_extraction_time = total_pages * 0.5  # 0.5 seconds per page
    embedding_time = num_documents * 5  # 5 seconds per document
    search_time = 2  # 2 seconds for search
    generation_time = 3  # 3 seconds for generation
    
    total_time = pdf_extraction_time + embedding_time + search_time + generation_time
    
    return total_time

def log_system_info():
    """
    Log system information for debugging.
    """
    import platform
    import psutil
    
    logger.info(f"System: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"CPU Count: {psutil.cpu_count()}")
    logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")

def format_time_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable format.
    
    Args:
        seconds (float): Duration in seconds
        
    Returns:
        str: Formatted duration
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"

def create_sample_input_files(input_dir: str):
    """
    Create sample input files for testing.
    
    Args:
        input_dir (str): Directory to create sample files in
    """
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    
    # Sample persona
    persona_content = """PhD Researcher in Computational Biology with expertise in machine learning applications for drug discovery. 
    Specializes in graph neural networks, molecular modeling, and bioinformatics. 
    Has 5 years of research experience and is currently working on developing AI models for predicting drug-target interactions."""
    
    with open(os.path.join(input_dir, "persona.txt"), 'w', encoding='utf-8') as f:
        f.write(persona_content)
    
    # Sample job description
    job_content = """Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks 
    for graph neural networks applied to drug discovery. Need to identify key algorithms, compare their effectiveness, 
    and summarize the most promising approaches for future research."""
    
    with open(os.path.join(input_dir, "job.txt"), 'w', encoding='utf-8') as f:
        f.write(job_content)
    
    # Sample config.json
    config = {
        "persona": persona_content,
        "job_to_be_done": job_content,
        "max_sections": 10,
        "max_subsections": 10
    }
    
    with open(os.path.join(input_dir, "config.json"), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Sample input files created in {input_dir}")

def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path (str): Path to file
        
    Returns:
        float: File size in MB
    """
    try:
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    except Exception:
        return 0.0

def monitor_memory_usage():
    """
    Monitor and log current memory usage.
    """
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"Memory usage: {memory.percent}% ({memory.used / (1024**3):.1f} GB / {memory.total / (1024**3):.1f} GB)")
    except ImportError:
        logger.warning("psutil not available, cannot monitor memory usage")

# Configuration constants
DEFAULT_MODEL_NAME = "distiluse-base-multilingual-cased-v1"
MAX_SECTIONS = 10
MAX_SUBSECTIONS = 10
MAX_PROCESSING_TIME = 60  # seconds
MAX_MODEL_SIZE_GB = 1.0
