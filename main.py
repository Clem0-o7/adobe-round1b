#!/usr/bin/env python3
"""
Main Orchestrator for Persona-Driven Document Intelligence
Coordinates the entire RAG pipeline from PDF extraction to final output generation.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.extract import extract_documents
from app.embed import create_document_index
from app.retrieve import retrieve_relevant_content
from app.generate import generate_final_response
from app.utils import (
    load_input_files, save_output, validate_input, 
    estimate_processing_time, log_system_info, 
    format_time_duration, monitor_memory_usage
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class PersonaDrivenRAG:
    """
    Main RAG pipeline orchestrator for persona-driven document intelligence.
    """
    
    def __init__(self, input_dir: str = "/app/input", output_dir: str = "/app/output"):
        """
        Initialize the RAG pipeline.
        
        Args:
            input_dir (str): Input directory path
            output_dir (str): Output directory path
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.start_time = None
        
    def run_pipeline(self) -> bool:
        """
        Run the complete RAG pipeline.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.start_time = time.time()
            
            logger.info("="*60)
            logger.info("Starting Persona-Driven Document Intelligence Pipeline")
            logger.info("="*60)
            
            # Log system information
            log_system_info()
            
            # Step 1: Load and validate inputs
            logger.info("\nStep 1: Loading and validating inputs...")
            pdf_files, persona, job_to_be_done = load_input_files(self.input_dir)
            
            if not validate_input(pdf_files, persona, job_to_be_done):
                logger.error("Input validation failed")
                return False
            
            # Estimate processing time
            total_pages = self._estimate_total_pages(pdf_files)
            estimated_time = estimate_processing_time(len(pdf_files), total_pages)
            logger.info(f"Estimated processing time: {format_time_duration(estimated_time)}")
            
            if estimated_time > 60:
                logger.warning("Estimated processing time exceeds 60-second constraint")
            
            # Step 2: Extract text from PDFs
            logger.info(f"\nStep 2: Extracting text from {len(pdf_files)} PDF files...")
            monitor_memory_usage()
            
            documents = extract_documents(pdf_files)
            if not documents:
                logger.error("No documents could be processed")
                return False
            
            total_sections = sum(len(doc['sections']) for doc in documents)
            logger.info(f"Extracted {total_sections} sections from {len(documents)} documents")
            
            # Step 3: Create embeddings and build index
            logger.info("\nStep 3: Creating embeddings and building search index...")
            monitor_memory_usage()
            
            embedder, index, metadata = create_document_index(documents)
            logger.info(f"Built search index with {len(metadata)} sections")
            
            # Step 4: Retrieve relevant content
            logger.info("\nStep 4: Retrieving relevant content for persona and job...")
            monitor_memory_usage()
            
            sections, subsections = retrieve_relevant_content(
                embedder, index, metadata, persona, job_to_be_done, top_k=15
            )
            
            logger.info(f"Retrieved {len(sections)} relevant sections and {len(subsections)} subsections")
            
            # Step 5: Generate final response
            logger.info("\nStep 5: Generating final response...")
            monitor_memory_usage()
            
            document_names = [os.path.basename(pdf) for pdf in pdf_files]
            final_output = generate_final_response(
                document_names, persona, job_to_be_done, sections, subsections
            )
            
            # Step 6: Save output
            logger.info("\nStep 6: Saving output...")
            save_output(final_output, self.output_dir, "answer.json")
            
            # Log completion
            end_time = time.time()
            total_time = end_time - self.start_time
            
            logger.info("="*60)
            logger.info("Pipeline completed successfully!")
            logger.info(f"Total processing time: {format_time_duration(total_time)}")
            logger.info(f"Output saved to: {os.path.join(self.output_dir, 'answer.json')}")
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _estimate_total_pages(self, pdf_files: list) -> int:
        """
        Estimate total number of pages across all PDFs.
        
        Args:
            pdf_files (list): List of PDF file paths
            
        Returns:
            int: Estimated total pages
        """
        try:
            import pdfplumber
            total_pages = 0
            
            for pdf_file in pdf_files[:3]:  # Check first 3 files for estimation
                try:
                    with pdfplumber.open(pdf_file) as pdf:
                        total_pages += len(pdf.pages)
                except Exception:
                    total_pages += 10  # Rough estimate if can't open
            
            # Estimate for remaining files
            if len(pdf_files) > 3:
                avg_pages = total_pages / min(3, len(pdf_files))
                total_pages += int(avg_pages * (len(pdf_files) - 3))
            
            return total_pages
            
        except Exception:
            # Fallback estimate
            return len(pdf_files) * 10

def main():
    """
    Main entry point for the application.
    """
    # Default paths (can be overridden by environment variables)
    input_dir = os.environ.get('INPUT_DIR', '/app/input')
    output_dir = os.environ.get('OUTPUT_DIR', '/app/output')
    
    # For local development, use relative paths
    if not os.path.exists(input_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.join(script_dir, 'input')
        output_dir = os.path.join(script_dir, 'output')
    
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create directories if they don't exist
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run the pipeline
    pipeline = PersonaDrivenRAG(input_dir, output_dir)
    success = pipeline.run_pipeline()
    
    if success:
        logger.info("Application completed successfully")
        sys.exit(0)
    else:
        logger.error("Application failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
