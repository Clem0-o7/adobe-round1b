"""
PDF Text Extraction Module
Extracts structured text with page numbers and sections from PDFs using advanced heading detection.
Integrates the Adobe Hackathon Round 1A heading detection approach.
"""

import pdfplumber
import re
import os
from typing import List, Dict, Tuple
import logging
from .pdf_processor import PDFProcessor
from .heading_detector import HeadingDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFExtractor:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.heading_detector = HeadingDetector()
        
        # Fallback patterns for simple heading detection (when advanced detection fails)
        self.fallback_heading_patterns = [
            r'^[A-Z][A-Z\s\d\.]{2,50}$',  # ALL CAPS headings
            r'^\d+\.\s+[A-Z]',             # Numbered sections (1. Title)
            r'^\d+\.\d+\s+[A-Z]',          # Sub-numbered (1.1 Title)
            r'^[A-Z][a-z]+(?:\s[A-Z][a-z]+)*:',  # Title Case with colon
            r'^Chapter\s\d+',              # Chapter headings
            r'^Section\s\d+',              # Section headings
        ]
    
    def extract_text_with_structure(self, pdf_path: str) -> Dict:
        """
        Extract structured text from PDF with page numbers and identified sections using advanced heading detection.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Dict: Structured text data with sections and page info
        """
        document_data = {
            'filename': pdf_path.split('/')[-1],
            'sections': [],
            'full_text': '',
            'total_pages': 0,
            'title': ''
        }
        
        try:
            # Use advanced PDF processing to extract structured data
            pages_data = self.pdf_processor.process_pdf(pdf_path)
            
            if not pages_data:
                logger.warning(f"No pages could be processed from {pdf_path}")
                return document_data
            
            document_data['total_pages'] = len(pages_data)
            
            # Extract title and headings using advanced detection
            title, headings = self.heading_detector.detect_title_and_headings(pages_data)
            document_data['title'] = title
            
            logger.info(f"Detected title: '{title}' and {len(headings)} headings")
            
            # Build full text and extract sections
            all_text_spans = []
            for page_data in pages_data:
                all_text_spans.extend(page_data.spans)
            
            # Sort spans by page and position
            all_text_spans.sort(key=lambda s: (s.page_num, s.bbox[1], s.bbox[0]))
            
            # Build full text
            current_page = 0
            for span in all_text_spans:
                if span.page_num != current_page:
                    document_data['full_text'] += f"\n[Page {span.page_num}]\n"
                    current_page = span.page_num
                document_data['full_text'] += span.text + " "
            
            # Convert headings to sections and extract content
            sections = self._create_sections_from_headings(headings, all_text_spans, pdf_path)
            
            # If no headings found, fall back to content-based sections
            if not sections:
                logger.info("No headings detected, falling back to content-based extraction")
                sections = self._extract_content_based_sections(pages_data, pdf_path)
            
            document_data['sections'] = sections
            
            logger.info(f"Extracted {len(sections)} sections from {pdf_path}")
                
        except Exception as e:
            logger.error(f"Error extracting from {pdf_path}: {str(e)}")
            # Fallback to simple extraction
            document_data = self._fallback_extraction(pdf_path)
            
        return document_data
    
    def _create_sections_from_headings(self, headings: List[Dict], all_spans: List, pdf_path: str) -> List[Dict]:
        """
        Create sections based on detected headings and extract content for each section.
        
        Args:
            headings: List of detected headings
            all_spans: All text spans from the document
            pdf_path: Path to the PDF file
            
        Returns:
            List of section dictionaries
        """
        if not headings:
            return []
        
        sections = []
        filename = pdf_path.split('/')[-1]
        
        # Convert spans to a more convenient format for content extraction
        spans_by_page = {}
        for span in all_spans:
            if span.page_num not in spans_by_page:
                spans_by_page[span.page_num] = []
            spans_by_page[span.page_num].append(span)
        
        # Sort spans within each page by position
        for page_num in spans_by_page:
            spans_by_page[page_num].sort(key=lambda s: (s.bbox[1], s.bbox[0]))
        
        # Create sections from headings
        for i, heading in enumerate(headings):
            heading_text = heading['text']
            heading_page = heading['page']
            heading_level = heading['level']
            
            # Find the heading span to get its position
            heading_span = None
            if heading_page in spans_by_page:
                for span in spans_by_page[heading_page]:
                    if heading_text in span.text or span.text in heading_text:
                        heading_span = span
                        break
            
            if not heading_span:
                continue
            
            # Determine section content boundaries
            content_spans = []
            
            # Get content from current page (after heading)
            if heading_page in spans_by_page:
                heading_y = heading_span.bbox[1]
                for span in spans_by_page[heading_page]:
                    # Include spans that come after this heading
                    if span.bbox[1] > heading_y and span != heading_span:
                        # Check if this span is another heading that should end this section
                        is_next_heading = any(
                            h['page'] == span.page_num and h['text'] in span.text 
                            for h in headings[i+1:i+3]  # Check next couple of headings
                        )
                        if is_next_heading:
                            break
                        content_spans.append(span)
            
            # Get content from subsequent pages until next major heading
            current_page = heading_page + 1
            max_pages_to_check = 3  # Limit to avoid pulling in unrelated content
            
            while current_page <= heading_page + max_pages_to_check:
                if current_page not in spans_by_page:
                    current_page += 1
                    continue
                
                # Check if this page has a major heading that should stop content extraction
                page_has_major_heading = any(
                    h['page'] == current_page and h['level'] in ['H1', 'H2']
                    for h in headings[i+1:]
                )
                
                if page_has_major_heading:
                    break
                
                # Add spans from this page
                for span in spans_by_page[current_page]:
                    # Stop if we hit another heading
                    is_heading = any(
                        h['page'] == span.page_num and h['text'] in span.text
                        for h in headings[i+1:]
                    )
                    if is_heading:
                        break
                    content_spans.append(span)
                
                current_page += 1
            
            # Combine content text
            content_text = ' '.join(span.text.strip() for span in content_spans if span.text.strip())
            content_text = self._clean_text(content_text)
            
            # Create section
            if content_text and len(content_text.strip()) > 20:  # Only include substantial content
                section = {
                    'document': os.path.basename(pdf_path),  # Only filename, not full path
                    'page_number': heading_page,
                    'section_title': heading_text,
                    'content': content_text,
                    'word_count': len(content_text.split()),
                    'importance_rank': 0,  # Will be calculated later
                    'heading_level': heading_level
                }
                sections.append(section)
        
        return sections
    
    def _extract_content_based_sections(self, pages_data: List, pdf_path: str) -> List[Dict]:
        """
        Fallback method to extract sections based on content when no headings are detected.
        
        Args:
            pages_data: List of page data
            pdf_path: Path to PDF file
            
        Returns:
            List of section dictionaries
        """
        sections = []
        filename = pdf_path.split('/')[-1]
        
        for page_data in pages_data:
            # Group spans into logical chunks
            page_text = ' '.join(span.text for span in page_data.spans if span.text.strip())
            page_text = self._clean_text(page_text)
            
            if len(page_text.strip()) > 50:  # Only include substantial content
                # Try to identify potential headings using fallback patterns
                lines = page_text.split('\n')
                current_section = None
                section_content = []
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if line might be a heading using fallback patterns
                    is_heading = self._is_fallback_heading(line)
                    
                    if is_heading:
                        # Save previous section
                        if current_section and section_content:
                            current_section['content'] = ' '.join(section_content)
                            current_section['word_count'] = len(section_content)
                            sections.append(current_section)
                        
                        # Start new section
                        current_section = {
                            'document': os.path.basename(pdf_path),  # Only filename, not full path
                            'page_number': page_data.page_num,
                            'section_title': line,
                            'content': '',
                            'word_count': 0,
                            'importance_rank': 0
                        }
                        section_content = []
                    else:
                        if current_section:
                            section_content.append(line)
                        else:
                            # Create general section for orphaned content
                            if len(section_content) > 20:  # Start new section after enough content
                                if section_content:
                                    sections.append({
                                        'document': filename,
                                        'page_number': page_data.page_num,
                                        'section_title': f"Content Section (Page {page_data.page_num})",
                                        'content': ' '.join(section_content),
                                        'word_count': len(section_content),
                                        'importance_rank': 0
                                    })
                                section_content = [line]
                            else:
                                section_content.append(line)
                
                # Handle last section
                if current_section and section_content:
                    current_section['content'] = ' '.join(section_content)
                    current_section['word_count'] = len(section_content)
                    sections.append(current_section)
                elif section_content:
                    sections.append({
                        'document': os.path.basename(pdf_path),  # Only filename, not full path
                        'page_number': page_data.page_num,
                        'section_title': f"Content Section (Page {page_data.page_num})",
                        'content': ' '.join(section_content),
                        'word_count': len(section_content),
                        'importance_rank': 0
                    })
        
        return sections
    
    def _is_fallback_heading(self, line: str) -> bool:
        """
        Determine if a line is likely a heading using fallback patterns.
        
        Args:
            line: Text line to check
            
        Returns:
            True if line appears to be a heading
        """
        if len(line) < 3 or len(line) > 100:
            return False
        
        for pattern in self.fallback_heading_patterns:
            if re.match(pattern, line):
                return True
        
        # Additional simple heuristics
        if len(line) < 50 and (
            line.isupper() or 
            re.match(r'^[A-Z][a-z]+(?:\s[A-Z][a-z]+)*$', line) or
            line.endswith(':')
        ):
            return True
        
        return False
    
    def _fallback_extraction(self, pdf_path: str) -> Dict:
        """
        Simple fallback extraction when advanced methods fail.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Basic document data
        """
        document_data = {
            'filename': pdf_path.split('/')[-1],
            'sections': [],
            'full_text': '',
            'total_pages': 0,
            'title': ''
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                document_data['total_pages'] = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if not text:
                        continue
                    
                    text = self._clean_text(text)
                    document_data['full_text'] += f"\n[Page {page_num}]\n{text}\n"
                    
                    # Create a basic section for each page
                    if len(text.strip()) > 50:
                        section = {
                            'document': document_data['filename'],
                            'page_number': page_num,
                            'section_title': f"Page {page_num} Content",
                            'content': text,
                            'word_count': len(text.split()),
                            'importance_rank': 0
                        }
                        document_data['sections'].append(section)
                
                # Try to extract title from first page
                if document_data['full_text']:
                    first_lines = document_data['full_text'].split('\n')[:5]
                    for line in first_lines:
                        line = line.strip()
                        if len(line) > 10 and len(line) < 100:
                            document_data['title'] = line
                            break
                
        except Exception as e:
            logger.error(f"Fallback extraction failed for {pdf_path}: {str(e)}")
        
        return document_data
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page headers/footers patterns
        text = re.sub(r'Page \d+.*?\n', '', text)
        # Clean up common PDF artifacts
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)\[\]\"\'\/\%\$\@\#]', '', text)
        
        return text.strip()
    
    def extract_multiple_pdfs(self, pdf_paths: List[str]) -> List[Dict]:
        """
        Extract text from multiple PDF files using advanced heading detection.
        
        Args:
            pdf_paths (List[str]): List of PDF file paths
            
        Returns:
            List[Dict]: List of extracted document data
        """
        documents = []
        for pdf_path in pdf_paths:
            logger.info(f"Processing: {pdf_path}")
            doc_data = self.extract_text_with_structure(pdf_path)
            if doc_data['sections']:
                documents.append(doc_data)
        
        return documents

# Helper function for easy import
def extract_documents(pdf_paths: List[str]) -> List[Dict]:
    """
    Convenience function to extract text from multiple PDFs.
    """
    extractor = PDFExtractor()
    return extractor.extract_multiple_pdfs(pdf_paths)
