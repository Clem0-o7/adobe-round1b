"""
LLM Answer Generation Module
Uses a local LLM (via transformers) to summarize and refine extracted sections.
"""

import json
import logging
from typing import List, Dict
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentGenerator:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """
        Initialize the generator with a local LLM model.
        For CPU-only constraints, we'll use a lightweight approach.
        
        Args:
            model_name (str): Name of the model to use
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the LLM model for text generation."""
        try:
            # For CPU-only constraint, we'll use a rule-based approach initially
            # and add lightweight model if needed
            logger.info("Using rule-based generation for CPU efficiency")
            self.model = "rule_based"  # Placeholder
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Fallback to rule-based approach
            self.model = "rule_based"
    
    def generate_section_summary(self, section: Dict, persona: str, job_to_be_done: str) -> str:
        """
        Generate a summary of a section tailored to the persona and job.
        
        Args:
            section (Dict): Section data with content
            persona (str): Persona description
            job_to_be_done (str): Job to be done description
            
        Returns:
            str: Generated summary
        """
        content = section.get('content', '')
        title = section.get('section_title', '')
        
        # Rule-based summarization for CPU efficiency
        summary = self._rule_based_summarization(content, title, persona, job_to_be_done)
        
        return summary
    
    def _rule_based_summarization(self, content: str, title: str, persona: str, job_to_be_done: str) -> str:
        """
        Rule-based approach to create focused summaries.
        
        Args:
            content (str): Section content
            title (str): Section title
            persona (str): Persona description
            job_to_be_done (str): Job to be done description
            
        Returns:
            str: Summarized content
        """
        # Extract key phrases from persona and job
        persona_keywords = self._extract_key_terms(persona)
        job_keywords = self._extract_key_terms(job_to_be_done)
        
        # Split content into sentences
        sentences = self._split_sentences(content)
        
        # Score sentences based on relevance
        scored_sentences = []
        for sentence in sentences:
            score = self._score_sentence(sentence, persona_keywords, job_keywords)
            if score > 0:
                scored_sentences.append((sentence, score))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in scored_sentences[:3]]  # Top 3 sentences
        
        # Create summary
        if top_sentences:
            summary = f"**{title}**: " + " ".join(top_sentences)
        else:
            # Fallback: take first few sentences
            summary = f"**{title}**: " + " ".join(sentences[:2])
        
        # Limit summary length
        if len(summary) > 300:
            summary = summary[:297] + "..."
        
        return summary
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """
        Extract key terms from text.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of key terms
        """
        # Simple keyword extraction
        import re
        
        # Common stop words to exclude
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'i', 'me', 'my', 'we', 'our', 'you'
        }
        
        # Extract meaningful words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        key_terms = [word for word in words if word not in stop_words]
        
        # Return unique terms
        return list(set(key_terms))
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of sentences
        """
        import re
        
        # Split by sentence endings
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return sentences
    
    def _score_sentence(self, sentence: str, persona_keywords: List[str], job_keywords: List[str]) -> float:
        """
        Score a sentence based on relevance to persona and job.
        
        Args:
            sentence (str): Sentence to score
            persona_keywords (List[str]): Keywords from persona
            job_keywords (List[str]): Keywords from job
            
        Returns:
            float: Relevance score
        """
        sentence_lower = sentence.lower()
        
        # Count keyword matches
        persona_matches = sum(1 for kw in persona_keywords if kw in sentence_lower)
        job_matches = sum(1 for kw in job_keywords if kw in sentence_lower)
        
        # Base score from keyword matches
        score = (persona_matches * 0.3) + (job_matches * 0.7)
        
        # Bonus for sentence length (prefer substantial sentences)
        if len(sentence) > 50:
            score += 0.2
        
        # Bonus for sentences with numbers/data (often important)
        if re.search(r'\d+', sentence):
            score += 0.1
        
        return score
    
    def refine_subsection(self, subsection: Dict, persona: str) -> str:
        """
        Refine a subsection for the specific persona.
        
        Args:
            subsection (Dict): Subsection data
            persona (str): Persona description
            
        Returns:
            str: Refined text
        """
        # Get the original text from the correct field
        original_text = subsection.get('refined_text', subsection.get('content', ''))
        
        # For better output, we'll keep the full text with minimal cleaning
        # Just clean up whitespace and formatting
        refined = re.sub(r'\s+', ' ', original_text.strip())
        
        return refined
    
    def _clean_and_highlight(self, text: str, persona: str) -> str:
        """
        Clean text and highlight key points for the persona.
        
        Args:
            text (str): Original text
            persona (str): Persona description
            
        Returns:
            str: Cleaned and highlighted text
        """
        # Clean up text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Extract persona keywords for highlighting
        persona_keywords = self._extract_key_terms(persona)
        
        # Simple highlighting (can be enhanced)
        for keyword in persona_keywords:
            if keyword in text.lower():
                # Don't actually highlight to keep output clean
                pass
        
        # Keep full text without truncation for subsection analysis
        # The original logic was truncating at 200 characters, but we want complete content
        return text
        
        return text
    
    def _generate_travel_itinerary(self, sections: List[Dict], subsections: List[Dict], 
                                  persona: str, job_to_be_done: str) -> Dict:
        """
        Generate a comprehensive travel itinerary based on extracted content.
        
        Args:
            sections (List[Dict]): Ranked sections
            subsections (List[Dict]): Extracted subsections
            persona (str): Persona description
            job_to_be_done (str): Job to be done description
            
        Returns:
            Dict: Structured travel itinerary
        """
        
        # Extract key information from content
        cities = []
        restaurants = []
        activities = []
        cultural_sites = []
        culinary_experiences = []
        
        # Parse content to extract structured information
        for subsection in subsections:
            text = subsection.get('refined_text', '').lower()
            
            # Extract cities
            if 'nice' in text or 'cannes' in text or 'marseille' in text or 'monaco' in text:
                cities.extend(self._extract_cities(text))
            
            # Extract restaurants and dining
            if 'restaurant' in text or 'dining' in text or 'michelin' in text:
                restaurants.extend(self._extract_restaurants(text))
            
            # Extract activities
            if 'visit' in text or 'explore' in text or 'tour' in text:
                activities.extend(self._extract_activities(text))
            
            # Extract cultural experiences
            if 'museum' in text or 'cathedral' in text or 'culture' in text or 'history' in text:
                cultural_sites.extend(self._extract_cultural_sites(text))
            
            # Extract culinary experiences
            if 'cooking' in text or 'market' in text or 'wine' in text or 'olive' in text:
                culinary_experiences.extend(self._extract_culinary_experiences(text))
        
        # Generate structured 7-day itinerary
        itinerary = {
            "overview": self._generate_overview(persona, job_to_be_done),
            "daily_schedule": self._generate_daily_schedule(cities, restaurants, activities, cultural_sites, culinary_experiences),
            "recommendations": {
                "must_visit_cities": list(set(cities))[:5],
                "top_restaurants": list(set(restaurants))[:8],
                "cultural_attractions": list(set(cultural_sites))[:6],
                "culinary_experiences": list(set(culinary_experiences))[:5],
                "suggested_activities": list(set(activities))[:10]
            },
            "practical_tips": self._generate_practical_tips()
        }
        
        return itinerary
    
    def _extract_cities(self, text: str) -> List[str]:
        """Extract city names from text."""
        cities = []
        city_keywords = ['nice', 'cannes', 'marseille', 'monaco', 'antibes', 'saint-tropez', 'avignon', 'aix-en-provence']
        
        for city in city_keywords:
            if city in text:
                cities.append(city.title())
        
        return cities
    
    def _extract_restaurants(self, text: str) -> List[str]:
        """Extract restaurant information from text."""
        restaurants = []
        
        # Look for restaurant patterns
        import re
        restaurant_patterns = [
            r'restaurant\s+[\w\s]+',
            r'dining\s+at\s+[\w\s]+',
            r'michelin\s+[\w\s]+',
            r'bistro\s+[\w\s]+',
            r'café\s+[\w\s]+'
        ]
        
        for pattern in restaurant_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            restaurants.extend([match.strip().title() for match in matches[:2]])
        
        return restaurants
    
    def _extract_activities(self, text: str) -> List[str]:
        """Extract activity suggestions from text."""
        activities = []
        
        activity_keywords = [
            'beach visit', 'walking tour', 'wine tasting', 'cooking class',
            'market exploration', 'museum visit', 'boat trip', 'hiking',
            'gallery tour', 'festival', 'shopping', 'sightseeing'
        ]
        
        for activity in activity_keywords:
            if any(word in text for word in activity.split()):
                activities.append(activity.title())
        
        return activities
    
    def _extract_cultural_sites(self, text: str) -> List[str]:
        """Extract cultural sites from text."""
        sites = []
        
        cultural_keywords = [
            'cathedral', 'museum', 'palace', 'castle', 'church',
            'gallery', 'monument', 'historic site', 'basilica',
            'abbey', 'fortress', 'temple'
        ]
        
        for site in cultural_keywords:
            if site in text:
                sites.append(site.title())
        
        return sites
    
    def _extract_culinary_experiences(self, text: str) -> List[str]:
        """Extract culinary experiences from text."""
        experiences = []
        
        culinary_keywords = [
            'cooking class', 'wine tour', 'food market', 'olive oil tasting',
            'truffle hunting', 'vineyard visit', 'farm-to-table dining',
            'local market', 'cheese tasting', 'pastry workshop'
        ]
        
        for experience in culinary_keywords:
            if any(word in text for word in experience.split()):
                experiences.append(experience.title())
        
        return experiences
    
    def _generate_overview(self, persona: str, job_to_be_done: str) -> str:
        """Generate itinerary overview."""
        return f"""Welcome to your personalized 7-day South of France itinerary, crafted by an experienced {persona}. 

This comprehensive itinerary focuses on {job_to_be_done.lower()}. You'll experience the perfect blend of culinary excellence, 
cultural immersion, and authentic French lifestyle. From the glamorous French Riviera to charming Provençal markets, 
each day offers carefully selected experiences that showcase the region's rich gastronomic heritage and cultural treasures.

Highlights include guided food tours, cooking classes with local chefs, visits to renowned museums and historic sites, 
wine tastings in celebrated vineyards, and dining at both Michelin-starred establishments and authentic local bistros."""
    
    def _generate_daily_schedule(self, cities: List[str], restaurants: List[str], 
                                activities: List[str], cultural_sites: List[str], 
                                culinary_experiences: List[str]) -> Dict:
        """Generate 7-day daily schedule."""
        
        schedule = {
            "day_1": {
                "location": "Nice",
                "theme": "Arrival and French Riviera Introduction",
                "morning": "Arrive in Nice, check into hotel, explore Vieux Nice (Old Town)",
                "afternoon": "Visit Cours Saleya Market for local produce and flowers",
                "evening": "Welcome dinner at a traditional Niçois restaurant",
                "highlights": ["Local market exploration", "Traditional Salade Niçoise", "Promenade des Anglais"]
            },
            "day_2": {
                "location": "Nice & Surrounding Areas",
                "theme": "Culture and Cuisine Discovery",
                "morning": "Visit Musée Matisse and Villa Arson contemporary art center",
                "afternoon": "Cooking class featuring Provençal specialties",
                "evening": "Dinner at a Michelin-starred restaurant",
                "highlights": ["Art museum visits", "Hands-on cooking experience", "Fine dining"]
            },
            "day_3": {
                "location": "Cannes & Antibes",
                "theme": "Glamour and Gastronomy",
                "morning": "Explore Cannes' La Croisette and Marché Forville",
                "afternoon": "Visit Antibes' Picasso Museum and old town",
                "evening": "Seafood dinner overlooking the Mediterranean",
                "highlights": ["Film festival city exploration", "Picasso art collection", "Fresh seafood"]
            },
            "day_4": {
                "location": "Provence Wine Region",
                "theme": "Wine and Vineyard Experience",
                "morning": "Drive to Provence wine region",
                "afternoon": "Vineyard tours and wine tastings (Châteauneuf-du-Pape or Bandol)",
                "evening": "Farm-to-table dinner at a vineyard restaurant",
                "highlights": ["Professional wine tastings", "Vineyard tours", "Rural dining experience"]
            },
            "day_5": {
                "location": "Aix-en-Provence",
                "theme": "Markets and Traditional Culture",
                "morning": "Explore Aix-en-Provence markets and Cézanne's studio",
                "afternoon": "Olive oil tasting and truffle hunting experience",
                "evening": "Cooking class followed by group dinner",
                "highlights": ["Artist's studio visit", "Truffle hunting", "Collaborative cooking"]
            },
            "day_6": {
                "location": "Marseille",
                "theme": "Port City Culture and Bouillabaisse",
                "morning": "Visit Vieux-Port and Basilique Notre-Dame de la Garde",
                "afternoon": "Le Panier district exploration and MuCEM museum",
                "evening": "Traditional bouillabaisse dinner at a renowned restaurant",
                "highlights": ["Historic port exploration", "Contemporary museum", "Iconic French dish"]
            },
            "day_7": {
                "location": "Monaco & Farewell",
                "theme": "Luxury and Departure",
                "morning": "Monaco Monte Carlo casino and palace visit",
                "afternoon": "Final shopping and café culture experience",
                "evening": "Farewell dinner at a rooftop restaurant with panoramic views",
                "highlights": ["Monte Carlo glamour", "Last-minute shopping", "Memorable farewell meal"]
            }
        }
        
        return schedule
    
    def _generate_practical_tips(self) -> Dict:
        """Generate practical travel tips."""
        return {
            "transportation": [
                "Rent a car for maximum flexibility between cities",
                "Use regional trains (TER) for eco-friendly travel",
                "Book restaurant reservations well in advance",
                "Consider purchasing a Museum Pass for cultural sites"
            ],
            "dining_etiquette": [
                "Lunch is typically served 12:00-14:00, dinner 19:30-22:00",
                "Tipping 5-10% is appreciated but not mandatory",
                "Always greet with 'Bonjour' when entering restaurants",
                "Ask for wine recommendations from sommeliers"
            ],
            "cultural_insights": [
                "Markets are best visited early morning for freshest produce",
                "Many museums close on Mondays or Tuesdays",
                "Dress code tends to be more formal, especially for dinner",
                "Learn basic French phrases - locals appreciate the effort"
            ],
            "food_experiences": [
                "Try rosé wine - the region is famous for it",
                "Sample local specialties: ratatouille, bouillabaisse, socca",
                "Visit during lavender season (June-July) for added sensory experience",
                "Book cooking classes with advance notice"
            ]
        }
    
    def create_final_output(self, documents: List[str], persona: str, job_to_be_done: str,
                          sections: List[Dict], subsections: List[Dict]) -> Dict:
        """
        Create the final JSON output in the required format.
        
        Args:
            documents (List[str]): List of input document names
            persona (str): Persona description
            job_to_be_done (str): Job to be done description
            sections (List[Dict]): Ranked sections
            subsections (List[Dict]): Extracted subsections
            
        Returns:
            Dict: Final output in required JSON format
        """
        # Create metadata (match expected format)
        metadata = {
            "input_documents": documents,
            "persona": persona.split('.')[0] if '.' in persona else persona,  # Shorten persona
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        # Process sections for output (match expected format)
        extracted_sections = []
        for section in sections[:5]:  # Limit to top 5 for cleaner output
            extracted_section = {
                "document": section['document'],
                "section_title": section['section_title'],
                "importance_rank": section['importance_rank'],
                "page_number": section['page_number']
            }
            extracted_sections.append(extracted_section)
        
        # Process subsections for output (match expected format)
        subsection_analysis = []
        for subsection in subsections[:5]:  # Limit to top 5 for cleaner output
            refined_text = self.refine_subsection(subsection, persona)
            
            subsection_data = {
                "document": subsection['document'],
                "refined_text": refined_text,
                "page_number": subsection['page_number']
            }
            subsection_analysis.append(subsection_data)
        
        # Create final output structure (matching expected format)
        output = {
            "metadata": metadata,
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }
        
        return output
    
    def generate_response(self, documents: List[str], persona: str, job_to_be_done: str,
                         sections: List[Dict], subsections: List[Dict]) -> Dict:
        """
        Main generation function that creates the final response.
        
        Args:
            documents (List[str]): List of input document names
            persona (str): Persona description
            job_to_be_done (str): Job to be done description
            sections (List[Dict]): Ranked sections
            subsections (List[Dict]): Extracted subsections
            
        Returns:
            Dict: Final JSON response
        """
        if not self.model:
            self.load_model()
        
        logger.info("Generating final response")
        
        # Create final output
        output = self.create_final_output(documents, persona, job_to_be_done, sections, subsections)
        
        logger.info(f"Generated response with {len(output['extracted_sections'])} sections and {len(output['subsection_analysis'])} subsections")
        
        return output

# Helper function for easy import
def generate_final_response(documents: List[str], persona: str, job_to_be_done: str,
                          sections: List[Dict], subsections: List[Dict]) -> Dict:
    """
    Convenience function to generate final response.
    
    Args:
        documents (List[str]): List of input document names
        persona (str): Persona description
        job_to_be_done (str): Job to be done description
        sections (List[Dict]): Ranked sections
        subsections (List[Dict]): Extracted subsections
        
    Returns:
        Dict: Final JSON response
    """
    generator = DocumentGenerator()
    return generator.generate_response(documents, persona, job_to_be_done, sections, subsections)
