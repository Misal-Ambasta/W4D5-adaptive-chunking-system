# Intelligent Document Chunking System

## Challenge

An enterprise software company's knowledge base contains diverse content types—technical docs, support tickets, API references, policies, and tutorials. Current uniform chunking breaks code snippets, separates troubleshooting steps, and disconnects policy requirements, causing poor retrieval accuracy.

This project aims to build an adaptive chunking system that automatically detects document types and applies appropriate chunking strategies (semantic, code-aware, hierarchical) to improve knowledge retrieval for internal teams and support automation.

## Overview

This system provides a solution for intelligent document chunking for enterprise knowledge management. It automatically detects document types from sources like Confluence, Jira, and GitHub wikis, and applies appropriate chunking strategies to improve retrieval accuracy and preserve context.

## Architecture

- **Backend**: FastAPI with intelligent chunking algorithms
- **Frontend**: Streamlit web interface
- **Document Classification**: Rule-based classification with content analysis
- **Chunking Strategies**: Semantic, code-aware, hierarchical, and fixed-size chunking

## Features

### Document Type Detection
- **Technical Documentation**: Architecture, design, implementation docs
- **Support Tickets**: Issue tracking and resolution documents
- **API References**: REST API, GraphQL documentation
- **Policies**: Compliance and governance documents
- **Tutorials**: Step-by-step guides and how-tos
- **Code Files**: Source code with structure preservation

### Chunking Strategies
- **Semantic Chunking**: Context-aware splitting for policies and general content
- **Code-Aware Chunking**: Preserves code blocks and function boundaries
- **Hierarchical Chunking**: Maintains document structure and section hierarchy
- **Fixed-Size Chunking**: Uniform chunks for consistent processing

### Analytics & Monitoring
- Token distribution analysis
- Chunk size progression tracking
- Processing metrics and performance monitoring
- Export capabilities (JSON, CSV, Text)

## Requirements

### Python Dependencies

```txt
# Core Framework
fastapi==0.104.1
uvicorn==0.24.0
streamlit==1.28.1

# Data Processing
pandas==2.1.3
numpy==1.24.3
tiktoken==0.5.1

# Web Requests
requests==2.31.0
httpx==0.25.2

# Visualization
plotly==5.17.0

# File Handling
python-multipart==0.0.6

# CORS Support
fastapi[all]==0.104.1
```

### System Requirements
- Python 3.8+
- 4GB+ RAM (recommended)
- 1GB+ disk space

## Installation

### 1. Clone or Create Project Structure

```bash
mkdir intelligent-chunking-system
cd intelligent-chunking-system
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install fastapi uvicorn streamlit pandas numpy tiktoken requests plotly python-multipart
```

### 4. Create Project Files

Create the following files in your project directory:

- `main.py` - FastAPI backend (use the FastAPI code provided)
- `app.py` - Streamlit frontend (use the Streamlit code provided)

## Usage

### Starting the System

#### 1. Start FastAPI Backend

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at: `http://localhost:8000`

#### 2. Start Streamlit Frontend

```bash
streamlit run app.py
```

The web interface will be available at: `http://localhost:8501`

### Using the Web Interface

1. **Text Input**: Paste document content directly
2. **File Upload**: Upload text files (.txt, .md, .py, .js, etc.)
3. **Sample Documents**: Try pre-loaded examples

### API Endpoints

#### Process Document
```bash
POST /chunk
Content-Type: application/json

{
  "content": "Document content here...",
  "filename": "optional-filename.txt",
  "document_id": "optional-custom-id"
}
```

#### Upload File
```bash
POST /chunk-file
Content-Type: multipart/form-data

# Upload file directly
```

#### Health Check
```bash
GET /health
```

#### Get Document Types
```bash
GET /document-types
```

#### Get Chunking Strategies
```bash
GET /chunking-strategies
```

## Configuration

### Chunking Parameters

You can modify chunking behavior by adjusting these parameters in the code:

```python
# Token limits for different strategies
SEMANTIC_MAX_TOKENS = 512
CODE_AWARE_MAX_TOKENS = 512
HIERARCHICAL_MAX_TOKENS = 512
FIXED_SIZE_MAX_TOKENS = 256

# Classification thresholds
CLASSIFICATION_THRESHOLD = 2  # Minimum pattern matches for classification
```

### API Configuration

```python
# FastAPI settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_RELOAD = True

# CORS settings
CORS_ORIGINS = ["*"]  # Restrict in production
```

## Performance Optimization

### For Large Documents
- Use hierarchical chunking for structured content
- Enable pagination for chunk display
- Implement caching for repeated processing

### For High Volume
- Add Redis caching
- Implement request queuing
- Use async processing for large files

## Deployment

### Local Development
```bash
# Terminal 1: API Server
uvicorn main:app --reload --port 8000

# Terminal 2: Streamlit App
streamlit run app.py
```

### Production Deployment

#### Using Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# For FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# For Streamlit (separate container)
# CMD ["streamlit", "run", "app.py", "--server.port", "8501"]
```

#### Using Gunicorn (FastAPI)
```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Monitoring & Analytics

### Metrics Tracked
- Processing time per document
- Chunk count and size distribution
- Document type classification accuracy
- Token usage statistics

### Performance Monitoring
- API response times
- Memory usage
- Error rates
- User interaction patterns

## Troubleshooting

### Common Issues

#### API Server Not Starting
```bash
# Check if port is in use
lsof -i :8000

# Try different port
uvicorn main:app --port 8001
```

#### Streamlit Connection Error
- Ensure FastAPI server is running on port 8000
- Check firewall settings
- Verify API_BASE_URL in app.py

#### Large File Processing
- Increase timeout settings
- Use streaming for very large files
- Implement progress tracking

### Debug Mode
```bash
# Enable FastAPI debug mode
uvicorn main:app --reload --log-level debug

# Enable Streamlit debug mode
streamlit run app.py --logger.level debug
```

## Extending the System

### Adding New Document Types
1. Update `DocumentType` enum
2. Add classification patterns
3. Test with sample documents

### Custom Chunking Strategies
1. Create new chunker class
2. Implement `chunk()` method
3. Add to strategy mapping

### Integration with Vector Databases
```python
# Example: Pinecone integration
import pinecone

def store_chunks(chunks, metadata):
    # Convert chunks to vectors
    vectors = embed_chunks(chunks)
    
    # Store in Pinecone
    pinecone.upsert(vectors, metadata)
```

## Testing

### Unit Tests
```bash
# Install pytest
pip install pytest

# Run tests
pytest test_chunking.py -v
```

### Integration Tests
```bash
# Test API endpoints
pytest test_api.py -v

# Test chunking strategies
pytest test_strategies.py -v
```

## Support

### Documentation
- API documentation available at: `http://localhost:8000/docs`
- Interactive testing at: `http://localhost:8000/redoc`

### Performance Tips
- Use appropriate chunking strategy for document type
- Monitor token usage for cost optimization
- Implement caching for repeated processing
- Use batch processing for multiple documents

## License

This project is provided as-is for educational and development purposes.