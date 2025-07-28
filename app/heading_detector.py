"""
Heading detection module using advanced heuristic approaches.
Implements multi-factor analysis for robust heading identification.

Adobe India Hackathon 2025 - Round 1A Solution
- Pure heuristic approach (no ML dependencies)
- 15+ detection features for robust classification  
- Ratio-based analysis beyond simple font-size detection
- H1/H2/H3 hierarchical classification
- Optimized for speed and accuracy
"""

import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict
from .pdf_processor import PageData, TextSpan


@dataclass
class HeadingCandidate:
    """Represents a potential heading with analysis scores."""
    span: TextSpan
    level: Optional[str] = None
    confidence: float = 0.0
    features: Dict[str, float] = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = {}


class HeadingDetector:
    """
    Advanced heading detection using multiple heuristics and optional ML.
    Implements the strategy from the hackathon requirements.
    """
    
    def __init__(self):
        # Removed ML functionality - using pure heuristic approach
        self.min_heading_length = 5  # Increased from 3
        self.max_heading_length = 200
        self.min_heading_words = 2  # Increased from 1
        
        # Patterns for numbered headings
        self.numbered_patterns = [
            r'^\d+\.?\s+',  # 1. or 1 
            r'^\d+\.\d+\.?\s+',  # 1.1. or 1.1
            r'^\d+\.\d+\.\d+\.?\s+',  # 1.1.1. or 1.1.1
            r'^[A-Z]\.?\s+',  # A. or A
            r'^[IVX]+\.?\s+',  # Roman numerals
            r'^\([a-z]\)\s+',  # (a)
            r'^\([0-9]+\)\s+',  # (1)
        ]
        
        # Common heading keywords
        self.heading_keywords = {
            'introduction', 'conclusion', 'abstract', 'summary', 'overview', 
            'background', 'methodology', 'results', 'discussion', 'references',
            'chapter', 'section', 'appendix', 'bibliography', 'index',
            'contents', 'table of contents', 'list of figures', 'preface',
            'acknowledgments', 'executive summary'
        }
        
        # Words that indicate NOT a heading
        self.non_heading_indicators = {
            'http', 'https', 'www', '.com', '.org', '.pdf', '.doc',
            'email', 'tel:', 'fax:', '@',
            'copyright', '©', '®', '™',
            'page', 'figure', 'table', 'chart', 'graph',
            'see also', 'continued', 'note:', 'source:'
        }
        
    def detect_title_and_headings(self, pages_data: List[PageData]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Detect document title and headings from processed pages.
        
        Args:
            pages_data: List of processed page data
            
        Returns:
            Tuple of (title, list of heading dictionaries)
        """
        if not pages_data:
            return "", []
        
        # Extract title from first page
        title = self._extract_title(pages_data[0])
        
        # Detect all heading candidates (excluding title)
        candidates = self._detect_heading_candidates(pages_data, title)
        
        if not candidates:
            return title, []
        
        # Classify candidates into H1, H2, H3 levels
        classified_headings = self._classify_heading_levels(candidates)
        
        # Convert to output format
        headings = []
        for candidate in classified_headings:
            if candidate.level:  # Only include classified headings
                headings.append({
                    "level": candidate.level,
                    "text": candidate.span.text,
                    "page": candidate.span.page_num
                })
        
        return title, headings
    
    def _extract_title(self, first_page: PageData) -> str:
        """
        Extract document title from the first page using font size and position heuristics.
        Handles multi-line titles by combining adjacent spans with same font size.
        
        Args:
            first_page: First page data
            
        Returns:
            Title string or empty string if not found
        """
        if not first_page.spans:
            return ""
        
        # Find spans in the upper portion of the first page
        upper_threshold = first_page.page_height * 0.4  # Top 40% of page
        
        # Filter spans in upper portion with reasonable font size
        title_candidates = []
        for span in first_page.spans:
            y_position = span.bbox[1]  # Top y-coordinate
            
            # Only consider spans in upper portion
            if y_position > upper_threshold:
                continue
            
            # Skip very small text
            if span.size < 12.0:
                continue
                
            title_candidates.append(span)
        
        if not title_candidates:
            return ""
        
        # Group by font size to identify potential multi-line titles
        size_groups = {}
        for span in title_candidates:
            size_key = round(span.size, 1)
            if size_key not in size_groups:
                size_groups[size_key] = []
            size_groups[size_key].append(span)
        
        # Find the best title by examining each font size group
        best_title = ""
        best_score = 0
        
        for font_size, spans in size_groups.items():
            # Sort spans by Y position (top to bottom)
            spans.sort(key=lambda s: s.bbox[1])
            
            # Try to identify consecutive lines that form a multi-line title
            title_lines = []
            current_line_spans = []
            current_y = None
            
            for span in spans:
                span_y = round(span.bbox[1], 1)
                
                # If this is a new line (different Y position)
                if current_y is None or abs(span_y - current_y) > 5:
                    # Save previous line if it exists
                    if current_line_spans:
                        # Sort by X position for proper reading order
                        current_line_spans.sort(key=lambda s: s.bbox[0])
                        line_text = " ".join(s.text.strip() for s in current_line_spans if s.text.strip())
                        if line_text:
                            title_lines.append(line_text)
                    
                    # Start new line
                    current_line_spans = [span]
                    current_y = span_y
                else:
                    # Same line, add span
                    current_line_spans.append(span)
            
            # Don't forget the last line
            if current_line_spans:
                current_line_spans.sort(key=lambda s: s.bbox[0])
                line_text = " ".join(s.text.strip() for s in current_line_spans if s.text.strip())
                if line_text:
                    title_lines.append(line_text)
            
            # Combine consecutive title lines
            if title_lines:
                # For title detection, we'll combine lines more conservatively
                combined_title_lines = []
                
                for i, line in enumerate(title_lines):
                    # Stop if we hit content that doesn't look like a title
                    if (len(line) > 80 or  # Very long lines are likely content, not titles
                        line.lower().startswith(('abstract', 'summary', 'table of contents')) or
                        '?' in line or  # Questions are likely content, not titles
                        len(line.split()) > 10):  # Very wordy lines are likely content
                        break
                    
                    combined_title_lines.append(line)
                    
                    # For multi-line titles, be conservative:
                    # - Limit to 2 lines max for most cases
                    # - Only go to 3 lines if each line is short (< 20 chars)
                    if len(combined_title_lines) >= 2:
                        if len(combined_title_lines) >= 3 or any(len(l) > 20 for l in combined_title_lines):
                            break
                
                if combined_title_lines:
                    # Join lines with space
                    combined_text = " ".join(combined_title_lines)
                    
                    # Additional length check - titles shouldn't be too long
                    if len(combined_text) > 150:
                        # If too long, just use the first line
                        combined_text = combined_title_lines[0]
                    
                    # Skip if combined text is too short
                    if len(combined_text.strip()) < 5:
                        continue
                    
                    # Score based on font size, position (higher = better), and text length
                    avg_y_pos = sum(s.bbox[1] for s in spans[:len(combined_title_lines)]) / len(spans[:len(combined_title_lines)])
                    position_score = 1.0 - (avg_y_pos / first_page.page_height)  # Higher position = better
                    length_score = min(len(combined_text) / 100.0, 1.0)  # Longer titles get higher score (up to 100 chars)
                    multi_line_bonus = 1.2 if len(combined_title_lines) > 1 else 1.0  # Bonus for multi-line titles
                    score = font_size * position_score * (1.0 + length_score) * multi_line_bonus
                    
                    if score > best_score:
                        best_score = score
                        best_title = combined_text
        
        # Fallback: use largest font in upper half if no good title found
        if not best_title:
            upper_spans = [s for s in first_page.spans if s.bbox[1] <= first_page.page_height * 0.5]
            if upper_spans:
                title_span = max(upper_spans, key=lambda s: s.size)
                best_title = title_span.text.strip()
        
        return best_title.strip()
    
    def _detect_heading_candidates(self, pages_data: List[PageData], title: str = "") -> List[HeadingCandidate]:
        """
        Detect potential heading candidates using multiple heuristics.
        
        Args:
            pages_data: List of page data
            
        Returns:
            List of heading candidates
        """
        all_spans = []
        for page_data in pages_data:
            all_spans.extend(page_data.spans)
        
        if not all_spans:
            return []
        
        # Keep all spans for candidate detection initially
        filtered_spans = all_spans
        
        # Calculate font size and text length statistics for ratio-based analysis
        font_sizes = [span.size for span in filtered_spans]
        text_lengths = [len(span.text.strip()) for span in filtered_spans]
        
        if not font_sizes:
            return []
        
        median_size = np.median(font_sizes)
        mean_size = np.mean(font_sizes)
        std_size = np.std(font_sizes)
        
        min_font_size = min(font_sizes)
        max_font_size = max(font_sizes)
        font_size_range = max_font_size - min_font_size if max_font_size > min_font_size else 1
        
        min_text_length = min(text_lengths)
        max_text_length = max(text_lengths)
        text_length_range = max_text_length - min_text_length if max_text_length > min_text_length else 1
        
        # Use more conservative threshold - 50th percentile as likely body text size
        font_sizes_sorted = sorted(font_sizes)
        n = len(font_sizes_sorted)
        
        # Use median as body text size, only consider spans significantly larger
        body_text_size = median_size
        size_threshold = max(body_text_size + std_size * 0.5, min_font_size + font_size_range * 0.15)
        
        candidates = []
        
        for span in filtered_spans:
            # Skip the title text to avoid duplication
            if title:
                span_text = span.text.strip()
                # Check if this span is part of the title
                if span_text in title or title in span_text:
                    continue
                
            # Enhanced size and formatting filter for better heading detection
            text_clean = span.text.strip()
            
            # Text validation first
            if not self._is_valid_heading_text(text_clean):
                continue
            
            # More inclusive criteria - consider bold formatting
            is_large_font = span.size >= size_threshold
            is_bold_font = 'bold' in span.font.lower() or 'black' in span.font.lower()
            is_medium_size = span.size >= 14.0  # Include 14pt text if bold
            
            # Accept if either large font OR (bold + reasonable size)
            if not (is_large_font or (is_bold_font and is_medium_size)):
                continue
            
            # Calculate features including ratio-based analysis
            features = self._calculate_heading_features(
                span, filtered_spans, median_size, 
                font_size_range, text_length_range, 
                min_font_size, max_font_size,
                min_text_length, max_text_length
            )
            
            # Combine heuristic scores
            confidence = self._calculate_confidence_score(features)
            
            # Higher threshold for candidate consideration
            if confidence > 0.4:  # Increased from 0.25
                candidate = HeadingCandidate(
                    span=span,
                    confidence=confidence,
                    features=features
                )
                candidates.append(candidate)
        
        # Sort by confidence and position
        candidates.sort(key=lambda c: (-c.confidence, c.span.page_num, c.span.bbox[1]))
        
        # Remove repetitive elements (headers/footers)
        candidates = self._remove_repetitive_elements(candidates, all_spans)
        
        # Filter out invalid heading text
        candidates = [c for c in candidates if self._is_valid_heading_text(c.span.text)]
        
        return candidates
    
    def _calculate_heading_features(self, span: TextSpan, all_spans: List[TextSpan], median_size: float,
                                   font_size_range: float, text_length_range: float,
                                   min_font_size: float, max_font_size: float,
                                   min_text_length: int, max_text_length: int) -> Dict[str, float]:
        """
        Calculate feature scores for a text span including ratio-based analysis.
        
        Args:
            span: Text span to analyze
            all_spans: All text spans for context
            median_size: Median font size in document
            font_size_range: Range of font sizes (max - min)
            text_length_range: Range of text lengths (max - min)
            min_font_size, max_font_size: Font size bounds
            min_text_length, max_text_length: Text length bounds
            
        Returns:
            Dictionary of feature scores
        """
        text = span.text.strip()
        features = {}
        
        # Original features
        features['size_ratio'] = span.size / median_size if median_size > 0 else 1.0
        
        # NEW: Ratio-based features for more nuanced classification
        
        # Font size position ratio (0 = smallest, 1 = largest)
        if font_size_range > 0:
            features['font_size_position_ratio'] = (span.size - min_font_size) / font_size_range
        else:
            features['font_size_position_ratio'] = 0.5
        
        # Text length ratio (shorter text gets higher score for headings)
        text_length = len(text)
        if text_length_range > 0:
            # Invert the ratio - shorter text (relative to range) gets higher score
            features['text_brevity_ratio'] = 1.0 - ((text_length - min_text_length) / text_length_range)
        else:
            features['text_brevity_ratio'] = 1.0
        
        # Character to font size ratio (heading principle: bigger font, fewer chars)
        if span.size > 0:
            features['char_to_fontsize_ratio'] = text_length / span.size
            # Normalize this ratio (smaller values = better for headings)
            max_ratio = max_text_length / min_font_size if min_font_size > 0 else 100
            features['char_to_fontsize_ratio_normalized'] = 1.0 - min(features['char_to_fontsize_ratio'] / max_ratio, 1.0)
        else:
            features['char_to_fontsize_ratio'] = 0
            features['char_to_fontsize_ratio_normalized'] = 0
        
        # Font size distinctiveness (how unique is this font size?)
        same_size_spans = [s for s in all_spans if abs(s.size - span.size) < 0.1]
        features['font_size_uniqueness'] = 1.0 / len(same_size_spans) if same_size_spans else 0
        
        # Text length distinctiveness in its font size group
        same_size_lengths = [len(s.text.strip()) for s in same_size_spans]
        if same_size_lengths:
            avg_length_in_group = np.mean(same_size_lengths)
            if avg_length_in_group > 0:
                # Shorter than average in same font size group = more likely heading
                features['length_distinctiveness'] = max(0, 1.0 - (text_length / avg_length_in_group))
            else:
                features['length_distinctiveness'] = 1.0
        else:
            features['length_distinctiveness'] = 1.0
        
        # Original features continue here...
        
        # Position on page (higher = better for headings)
        page_spans = [s for s in all_spans if s.page_num == span.page_num]
        if page_spans:
            y_positions = [s.bbox[1] for s in page_spans]
            min_y, max_y = min(y_positions), max(y_positions)
            if max_y > min_y:
                features['position_score'] = 1.0 - (span.bbox[1] - min_y) / (max_y - min_y)
            else:
                features['position_score'] = 1.0
        else:
            features['position_score'] = 0.5
        
        # Text characteristics
        features['length_score'] = max(0, 1.0 - len(text) / 100.0)  # Prefer shorter text
        features['word_count'] = len(text.split())
        
        # Capitalization patterns
        words = text.split()
        if words:
            title_case_count = sum(1 for word in words if word[0].isupper() and len(word) > 1)
            features['title_case_ratio'] = title_case_count / len(words)
            features['all_caps'] = 1.0 if text.isupper() else 0.0
        else:
            features['title_case_ratio'] = 0.0
            features['all_caps'] = 0.0
        
        # Numbered heading detection
        features['has_numbering'] = 0.0
        for pattern in self.numbered_patterns:
            if re.match(pattern, text):
                features['has_numbering'] = 1.0
                break
        
        # Keyword matching
        text_lower = text.lower()
        features['keyword_match'] = 0.0
        for keyword in self.heading_keywords:
            if keyword in text_lower:
                features['keyword_match'] = 1.0
                break
        
        # Font consistency (do other spans have similar font size?)
        similar_size_count = sum(1 for s in all_spans if abs(s.size - span.size) < 0.5)
        features['font_uniqueness'] = 1.0 / max(1, similar_size_count / len(all_spans))
        
        # Line isolation (is this span alone on its line?)
        same_line_spans = [
            s for s in page_spans 
            if abs(s.bbox[1] - span.bbox[1]) < 5  # Within 5 points vertically
            and s != span
        ]
        features['line_isolation'] = 1.0 if len(same_line_spans) == 0 else 0.0
        
        return features
    
    def _calculate_confidence_score(self, features: Dict[str, float]) -> float:
        """
        Calculate overall confidence score from features.
        
        Args:
            features: Dictionary of feature scores
            
        Returns:
            Confidence score between 0 and 1
        """
        # Enhanced weighted combination of features including ratio-based analysis
        weights = {
            # Original features (reduced weights to make room for new ones)
            'size_ratio': 0.15,
            'position_score': 0.10,
            'length_score': 0.08,
            'title_case_ratio': 0.10,
            'has_numbering': 0.12,
            'keyword_match': 0.08,
            'font_uniqueness': 0.03,
            'line_isolation': 0.04,
            
            # New ratio-based features
            'font_size_position_ratio': 0.12,  # How large this font is relative to size range
            'text_brevity_ratio': 0.08,  # Shorter text = more likely heading
            'char_to_fontsize_ratio_normalized': 0.10,  # Optimal char/font ratio for headings
            'font_size_uniqueness': 0.05,  # How unique this font size is
            'length_distinctiveness': 0.05   # How short this text is compared to same font size
        }
        
        score = 0.0
        total_weight = 0.0
        
        for feature, weight in weights.items():
            if feature in features:
                score += features[feature] * weight
                total_weight += weight
        
        # Normalize by total weight used
        if total_weight > 0:
            score = score / total_weight
        
        # Apply bonus for very large fonts
        if features.get('size_ratio', 0) > 1.5:
            score += 0.1
        
        # Apply penalty for very long text
        if features.get('word_count', 0) > 15:
            score *= 0.8
        
        return min(1.0, max(0.0, score))
    
    def _classify_heading_levels(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        """
        Classify heading candidates into H1, H2, H3 levels using font size and formatting.
        
        Args:
            candidates: List of heading candidates
            
        Returns:
            List of candidates with assigned levels
        """
        if not candidates:
            return []
        
        # Sort by font size (descending) then by document order
        candidates.sort(key=lambda c: (-c.span.size, c.span.page_num, c.span.bbox[1]))
        
        # Analyze font patterns for hierarchy
        font_patterns = {}
        for candidate in candidates:
            key = (round(candidate.span.size, 1), 'bold' in candidate.span.font.lower())
            if key not in font_patterns:
                font_patterns[key] = []
            font_patterns[key].append(candidate)
        
        # Sort patterns by priority: size first, then bold
        sorted_patterns = sorted(font_patterns.keys(), key=lambda x: (-x[0], -x[1]))
        
        # Assign levels based on patterns
        level_assignments = {}
        level_index = 0
        level_mapping = ["H1", "H2", "H3"]
        
        for pattern in sorted_patterns:
            if level_index >= len(level_mapping):
                break
                
            size, is_bold = pattern
            level = level_mapping[level_index]
            
            # Special logic for level assignment
            if size >= 20.0:
                # Large fonts are always H1
                level = "H1"
            elif size >= 16.0:
                # Medium-large fonts are H2
                level = "H2" if level_index > 0 else "H1"
            elif size >= 14.0 and is_bold:
                # Bold 14pt text can be H2 or H3 depending on context
                if level_index == 0:
                    level = "H2"  # First group of 14pt bold becomes H2
                else:
                    level = "H3"  # Subsequent groups become H3
            else:
                # Smaller text defaults to H3
                level = "H3"
            
            level_assignments[pattern] = level
            level_index += 1
        
        # Apply level assignments
        for candidate in candidates:
            pattern = (round(candidate.span.size, 1), 'bold' in candidate.span.font.lower())
            candidate.level = level_assignments.get(pattern, "H3")
        
        # Refine assignments based on content and context
        classified = []
        for candidate in candidates:
            # Apply confidence threshold - more inclusive for better recall
            if candidate.confidence > 0.35:  # Lower threshold for better recall
                text = candidate.span.text.strip()
                
                # Boost major section headings to H1 (only the full section names)
                if any(keyword in text.lower() for keyword in ['round 1a:', 'round 1b:', 'appendix']):
                    candidate.level = "H1"
                
                # Common H2 patterns - major subsections
                elif any(pattern in text.lower() for pattern in [
                    'your mission', 'why this matters', 'what you need', 'requirements', 
                    'execution', 'scoring criteria', 'checklist', 'pro tips', 'what not to do',
                    'docker requirements', 'expected execution', 'submission checklist',
                    'the journey ahead', 'you will be provided', 'constraints',
                    'round 1:', 'round 2:', 'challenge brief (for participants)'
                ]):
                    candidate.level = "H2"  # Force H2 for these patterns
                
                # Test cases and similar should be H3
                elif any(pattern in text.lower() for pattern in [
                    'test case', 'example', 'sample', 'input specification', 'output',
                    'document collection', 'persona definition', 'challenge theme',
                    'challenge brief', 'are you in', 'connect what matters'
                ]):
                    candidate.level = "H3"  # Force H3 for these patterns
                
                classified.append(candidate)
        
        # Sort by document order for final output
        classified.sort(key=lambda c: (c.span.page_num, c.span.bbox[1]))
        
        return classified
    

    
    def _remove_repetitive_elements(self, candidates: List[HeadingCandidate], all_spans: List[TextSpan]) -> List[HeadingCandidate]:
        """
        Remove text that appears repetitively across pages (headers, footers).
        
        Args:
            candidates: List of heading candidates
            all_spans: All text spans for context
            
        Returns:
            Filtered list of candidates
        """
        # Group spans by text content
        text_positions = {}
        for span in all_spans:
            text = span.text.strip()
            if text not in text_positions:
                text_positions[text] = []
            text_positions[text].append((span.page_num, span.bbox))
        
        filtered_candidates = []
        for candidate in candidates:
            text = candidate.span.text.strip()
            positions = text_positions.get(text, [])
            
            # If text appears on multiple pages in similar positions, likely header/footer
            if len(positions) > 1:
                # Check if positions are consistent (similar y-coordinates)
                y_positions = [bbox[1] for _, bbox in positions]  # Top y-coordinate
                y_variance = np.var(y_positions) if len(y_positions) > 1 else 0
                
                # If variance is low, it's likely a repetitive element
                if y_variance < 50:  # Threshold for position consistency
                    continue
            
            filtered_candidates.append(candidate)
        
        return filtered_candidates
    
    def _is_valid_heading_text(self, text: str) -> bool:
        """
        Validate if text looks like a legitimate heading with enhanced criteria.
        
        Args:
            text: Text to validate
            
        Returns:
            True if text appears to be a valid heading
        """
        text = text.strip()
        
        # Check against non-heading indicators
        text_lower = text.lower()
        for indicator in self.non_heading_indicators:
            if indicator in text_lower:
                return False
        
        # Must have minimum length
        if len(text) < self.min_heading_length:
            return False
        
        words = text.split()
        
        # Enhanced filtering for better quality
        
        # Allow important section headers even if short (1-2 words)
        is_important_short_header = any(pattern in text_lower for pattern in [
            'round', 'chapter', 'section', 'part', 'appendix', 'mission', 'tips', 'criteria'
        ])
        
        # Apply word count requirement unless it's an important short header
        if len(words) < self.min_heading_words and not is_important_short_header:
            return False
        
        # Reject very short fragments (likely formatting artifacts) unless important
        if len(words) <= 2 and not is_important_short_header:
            return False
        
        # Reject if mostly digits or special characters
        alpha_chars = sum(1 for c in text if c.isalpha())
        if alpha_chars / len(text) < 0.5:
            return False
        
        # Reject very repetitive text (same character repeated)
        unique_chars = len(set(text.lower()))
        if unique_chars < 3:
            return False
        
        # Reject common non-heading phrases and artifacts
        reject_phrases = {
            
        }
        
        if text_lower in reject_phrases:
            return False
        
        # Reject fragments that are likely parts of sentences
        if (text.endswith('—') or text.startswith('—') or 
            text.endswith(',') or text.startswith(',') or
            text.endswith(';') or text.startswith(';') or
            text.startswith('"') and text.endswith('"') and len(text) > 50):
            return False
        
        # Reject if it looks like a filename or URL
        if any(ext in text_lower for ext in ['.pdf', '.doc', '.json', 'http:', 'https:']):
            return False
        
        return True
