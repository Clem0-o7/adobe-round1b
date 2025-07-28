"""
PDF Processor Module
Handles PDF parsing and creates structured data objects for heading detection.
"""

import pdfplumber
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class TextSpan:
    """Represents a text span with formatting information."""
    text: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    size: float
    font: str
    page_num: int
    
    def __post_init__(self):
        """Clean up text and ensure proper formatting."""
        self.text = self.text.strip()

@dataclass 
class PageData:
    """Represents processed data for a single page."""
    page_num: int
    spans: List[TextSpan]
    page_width: float
    page_height: float

class PDFProcessor:
    """
    Processes PDF files and extracts structured text data with formatting information.
    """
    
    def __init__(self):
        self.min_font_size = 8.0  # Minimum font size to consider
        self.max_font_size = 72.0  # Maximum reasonable font size
        
    def process_pdf(self, pdf_path: str) -> List[PageData]:
        """
        Process a PDF file and extract structured page data.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of PageData objects
        """
        pages_data = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_data = self._process_page(page, page_num)
                    if page_data and page_data.spans:
                        pages_data.append(page_data)
                        
            logger.info(f"Processed {len(pages_data)} pages from {pdf_path}")
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            
        return pages_data
    
    def _process_page(self, page, page_num: int) -> Optional[PageData]:
        """
        Process a single page and extract text spans with formatting.
        
        Args:
            page: pdfplumber page object
            page_num: Page number
            
        Returns:
            PageData object or None if processing fails
        """
        try:
            # Get page dimensions
            page_width = float(page.width)
            page_height = float(page.height)
            
            # Extract characters with formatting information
            chars = page.chars
            if not chars:
                return None
            
            # Group characters into text spans
            spans = self._group_chars_into_spans(chars, page_num)
            
            # Filter and clean spans
            spans = self._filter_spans(spans)
            
            return PageData(
                page_num=page_num,
                spans=spans,
                page_width=page_width,
                page_height=page_height
            )
            
        except Exception as e:
            logger.warning(f"Error processing page {page_num}: {str(e)}")
            return None
    
    def _group_chars_into_spans(self, chars: List[Dict], page_num: int) -> List[TextSpan]:
        """
        Group individual characters into text spans based on formatting and position.
        
        Args:
            chars: List of character dictionaries from pdfplumber
            page_num: Page number
            
        Returns:
            List of TextSpan objects
        """
        if not chars:
            return []
        
        spans = []
        current_span_chars = []
        current_font = None
        current_size = None
        
        for char in chars:
            char_text = char.get('text', '')
            char_font = char.get('fontname', 'unknown')
            char_size = float(char.get('size', 12))
            char_bbox = (
                float(char.get('x0', 0)),
                float(char.get('y0', 0)), 
                float(char.get('x1', 0)),
                float(char.get('y1', 0))
            )
            
            # Skip whitespace-only characters for grouping purposes
            if not char_text.strip():
                if current_span_chars:
                    current_span_chars.append(char)
                continue
            
            # Check if this character belongs to the current span
            if (current_font is None or 
                (char_font == current_font and abs(char_size - current_size) < 0.1)):
                # Same formatting - add to current span
                current_span_chars.append(char)
                current_font = char_font
                current_size = char_size
            else:
                # Different formatting - save current span and start new one
                if current_span_chars:
                    span = self._create_span_from_chars(current_span_chars, page_num)
                    if span:
                        spans.append(span)
                
                # Start new span
                current_span_chars = [char]
                current_font = char_font
                current_size = char_size
        
        # Don't forget the last span
        if current_span_chars:
            span = self._create_span_from_chars(current_span_chars, page_num)
            if span:
                spans.append(span)
        
        return spans
    
    def _create_span_from_chars(self, chars: List[Dict], page_num: int) -> Optional[TextSpan]:
        """
        Create a TextSpan from a list of characters.
        
        Args:
            chars: List of character dictionaries
            page_num: Page number
            
        Returns:
            TextSpan object or None
        """
        if not chars:
            return None
        
        # Combine text
        text = ''.join(char.get('text', '') for char in chars)
        text = text.strip()
        
        if not text:
            return None
        
        # Calculate bounding box
        x0 = min(float(char.get('x0', 0)) for char in chars)
        y0 = min(float(char.get('y0', 0)) for char in chars)
        x1 = max(float(char.get('x1', 0)) for char in chars)
        y1 = max(float(char.get('y1', 0)) for char in chars)
        
        # Get formatting from first non-whitespace character
        font = 'unknown'
        size = 12.0
        
        for char in chars:
            if char.get('text', '').strip():
                font = char.get('fontname', 'unknown')
                size = float(char.get('size', 12))
                break
        
        return TextSpan(
            text=text,
            bbox=(x0, y0, x1, y1),
            size=size,
            font=font,
            page_num=page_num
        )
    
    def _filter_spans(self, spans: List[TextSpan]) -> List[TextSpan]:
        """
        Filter out invalid or unwanted text spans.
        
        Args:
            spans: List of TextSpan objects
            
        Returns:
            Filtered list of spans
        """
        filtered = []
        
        for span in spans:
            # Skip empty or whitespace-only text
            if not span.text or not span.text.strip():
                continue
            
            # Skip very small text (likely artifacts)
            if span.size < self.min_font_size:
                continue
            
            # Skip unreasonably large text (likely errors)
            if span.size > self.max_font_size:
                continue
            
            # Skip single characters unless they are meaningful
            if len(span.text.strip()) == 1 and span.text.strip() not in ['§', '•', '◦', '▪']:
                continue
            
            filtered.append(span)
        
        return filtered
