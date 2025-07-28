# Persona-Driven Document Intelligence RAG Pipeline

A CPU-optimized Retrieval-Augmented Generation (RAG) system that intelligently extracts and prioritizes document sections based on user personas and their specific job requirements. **Features advanced heading detection from Adobe Hackathon Round 1A** for superior document structure understanding.

## 🚀 Quick Start

### Build the Docker Image
```bash
docker build --platform linux/amd64 -t persona-rag:latest .
```

### Run the Pipeline
```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  persona-rag:latest
```

## 📁 Input Structure

Place your input files in the `input/` directory:

```
input/
├── document1.pdf
├── document2.pdf
├── document3.pdf
├── persona.txt          # Persona description
└── job.txt             # Job to be done description
```

### Alternative Input Format
You can also use a single `config.json` file:

```json
{
  "persona": "PhD Researcher in Computational Biology with expertise in machine learning...",
  "job_to_be_done": "Prepare a comprehensive literature review focusing on methodologies...",
  "max_sections": 10,
  "max_subsections": 10
}
```

## 📤 Output

The system generates `output/answer.json` with the following structure:

```json
{
  "metadata": {
    "input_documents": ["doc1.pdf", "doc2.pdf"],
    "persona": "PhD Researcher in Computational Biology...",
    "job_to_be_done": "Prepare a comprehensive literature review...",
    "processing_timestamp": "2024-01-15T10:30:00"
  },
  "extracted_sections": [
    {
      "document": "doc1.pdf",
      "page_number": 3,
      "section_title": "Graph Neural Networks for Drug Discovery",
      "importance_rank": 1
    }
  ],
  "subsection_analysis": [
    {
      "document": "doc1.pdf",
      "page_number": 3,
      "refined_text": "Graph neural networks have shown promising results..."
    }
  ]
}
```

## 🏗️ Architecture

### Pipeline Stages

1. **Advanced PDF Processing** (`app/pdf_processor.py` + `app/heading_detector.py`)
   - Character-level PDF parsing with formatting extraction
   - **Round 1A award-winning heading detection algorithm**
   - 15+ heuristic features including ratio-based analysis
   - Multi-factor confidence scoring and hierarchical classification
   - Handles multi-line titles and complex document structures

2. **Embedding & Indexing** (`app/embed.py`)
   - Creates semantic embeddings using sentence-transformers
   - Builds FAISS index for fast similarity search
   - Optimized for CPU-only operation

3. **Persona-Driven Retrieval** (`app/retrieve.py`)
   - Generates persona-specific queries
   - Performs semantic search and re-ranking
   - Extracts relevant subsections

4. **Response Generation** (`app/generate.py`)
   - Rule-based text summarization
   - Persona-aware content refinement
   - Formats final JSON output

### Key Components

- **Advanced Heading Detection**: Multi-factor analysis with font size ratios, position scoring, and content-aware classification
- **Sentence Transformers**: `distiluse-base-multilingual-cased-v1` (≤1GB)
- **Search Engine**: FAISS with cosine similarity
- **PDF Processing**: pdfplumber for robust text extraction with formatting
- **Text Processing**: Rule-based NLP for CPU efficiency

## ⚡ Performance

- **Processing Time**: ≤60 seconds for 3-5 documents
- **Model Size**: ≤1GB total
- **CPU Only**: No GPU dependencies
- **Memory Efficient**: Optimized for 16GB RAM systems
- **Offline**: No internet access required
- **Superior Structure Understanding**: Advanced heading detection for better section extraction

## 🧪 Testing

### Sample Test Cases

**Academic Research**:
```
Persona: PhD Researcher in Computational Biology
Job: Prepare literature review on Graph Neural Networks for Drug Discovery
Documents: 4 research papers
```

**Business Analysis**:
```
Persona: Investment Analyst  
Job: Analyze revenue trends and R&D investments
Documents: 3 annual reports from tech companies
```

**Educational Content**:
```
Persona: Undergraduate Chemistry Student
Job: Identify key concepts for exam preparation on reaction kinetics
Documents: 5 organic chemistry textbook chapters
```

### Create Sample Input
```bash
python -c "
from app.utils import create_sample_input_files
create_sample_input_files('input')
"
```

## 🔧 Configuration

### Environment Variables
- `INPUT_DIR`: Input directory path (default: `/app/input`)
- `OUTPUT_DIR`: Output directory path (default: `/app/output`)

### Model Configuration
- Embedding model: `distiluse-base-multilingual-cased-v1`
- FAISS index type: `IndexFlatIP` (exact cosine similarity)
- Maximum sections: 10
- Maximum subsections: 10

## 📊 Scoring Optimization

Our solution is optimized for the challenge scoring criteria:

- **Section Relevance (60 points)**: Advanced semantic search with persona-aware re-ranking + superior heading detection
- **Sub-Section Relevance (40 points)**: Intelligent text extraction with context preservation + structured section boundaries

## 🛠️ Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python main.py
```

### Project Structure
```
rag_pipeline/
├── app/
│   ├── extract.py           # Enhanced PDF text extraction
│   ├── pdf_processor.py     # Character-level PDF processing
│   ├── heading_detector.py  # Advanced heading detection (Round 1A)
│   ├── embed.py            # Embedding & indexing
│   ├── retrieve.py         # Query & retrieval
│   ├── generate.py         # Response generation
│   └── utils.py            # Utility functions
├── main.py                 # Main orchestrator
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
├── approach_explanation.md
└── README.md
```

## 🚨 Constraints Compliance

✅ **CPU Only**: No GPU dependencies  
✅ **Model Size**: ≤1GB total  
✅ **Processing Time**: ≤60 seconds  
✅ **Offline**: No internet access  
✅ **Generic**: Works across domains/personas  
✅ **Output Format**: Matches required JSON schema  
✅ **Advanced Structure Understanding**: Integrated Round 1A heading detection

## 📝 License

This project is developed for the Adobe India Hackathon Challenge Round 1B.
