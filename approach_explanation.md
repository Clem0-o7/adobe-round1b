# Approach Explanation: Persona-Driven Document Intelligence

## Overview

Our solution implements a **Retrieval-Augmented Generation (RAG) pipeline** specifically designed for persona-driven document analysis. The system extracts, indexes, and intelligently retrieves document sections based on user personas and their specific job requirements, ensuring CPU-only operation within strict performance constraints. **We integrate advanced heading detection from Adobe Hackathon Round 1A** for superior document structure understanding.

## Methodology

### 1. **Advanced PDF Processing & Heading Detection**
We employ a **two-stage approach** combining the award-winning heading detection algorithm from Round 1A with content-based extraction. Our system uses `pdfplumber` for character-level PDF parsing, extracting formatting information (font size, font family, positioning) to create structured `TextSpan` objects. The **advanced heading detector** implements **15+ heuristic features** including ratio-based analysis, font size distinctiveness, position scoring, and multi-factor confidence calculation. This goes far beyond simple font-size detection, using sophisticated pattern matching and hierarchical classification for H1/H2/H3 level assignment.

### 2. **Intelligent Document Structure Analysis**
Our heading detection system implements **multi-factor analysis** with features like:
- **Font size position ratio** (relative to document font range)
- **Text brevity ratio** (shorter text preferred for headings)
- **Character-to-font-size ratio** (optimal balance for headings)
- **Font uniqueness scoring** (rare font sizes weighted higher)
- **Repetitive element filtering** (removes headers/footers)
- **Content-aware classification** (domain-specific heading patterns)

### 3. **Semantic Embedding & Indexing**
The pipeline employs `sentence-transformers` with the `distiluse-base-multilingual-cased-v1` model (≤1GB) for creating dense vector representations of document sections. We build a FAISS index for efficient similarity search, enabling sub-second retrieval even with large document collections. The embedding process combines section titles with content for enhanced contextual understanding.

### 4. **Persona-Aware Query Generation**
Our system dynamically constructs search queries by combining persona characteristics with job requirements. This approach ensures that retrieval is specifically tailored to what matters most for the given user profile, rather than generic document summarization.

### 5. **Multi-Stage Relevance Ranking**
Retrieved sections undergo a sophisticated re-ranking process that considers:
- **Semantic similarity** to the persona-job query (60% weight)
- **Keyword matching** with persona and job descriptions (30% weight)  
- **Content substantiality** and structure quality (10% weight)

This multi-factor approach ensures that the most relevant sections for the specific persona rise to the top.

### 6. **Intelligent Subsection Extraction**
From top-ranked sections, we extract refined subsections using rule-based text processing. The system identifies key sentences through relevance scoring, considering persona-specific keywords and job-related terms. This granular analysis provides actionable insights directly aligned with user needs.

## Technical Optimizations

**Advanced Structure Understanding**: Our Round 1A heading detection algorithm provides superior document structure analysis compared to simple pattern matching, enabling more accurate section boundaries and content extraction.

**CPU Efficiency**: We replaced heavy LLM inference with efficient rule-based text processing and lightweight pattern matching, ensuring sub-60-second processing times while maintaining the sophisticated heading analysis.

**Memory Management**: Batch processing of embeddings and streamlined data structures minimize memory footprint while maintaining search quality.

**Offline Operation**: All models and dependencies are pre-downloaded and cached within the Docker container, ensuring no internet access is required during execution.

## Architecture Benefits

Our approach balances accuracy with performance constraints by leveraging semantic understanding where it matters most (retrieval) while using efficient heuristics for summarization tasks. The **integration of advanced heading detection** significantly improves document structure understanding, leading to better section extraction and more relevant content retrieval. The modular design allows easy adaptation to different domains and persona types, making it genuinely generic across the diverse test scenarios specified in the challenge requirements.
