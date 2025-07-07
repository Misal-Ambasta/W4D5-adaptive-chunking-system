from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import re
import json
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import tiktoken
import hashlib

app = FastAPI(title="Intelligent Document Chunking API", version="1.0.0")

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DocumentType(str, Enum):
    TECHNICAL_DOC = "technical_doc"
    SUPPORT_TICKET = "support_ticket"
    API_REFERENCE = "api_reference"
    POLICY = "policy"
    TUTORIAL = "tutorial"
    CODE = "code"
    UNKNOWN = "unknown"

class ChunkingStrategy(str, Enum):
    SEMANTIC = "semantic"
    CODE_AWARE = "code_aware"
    HIERARCHICAL = "hierarchical"
    FIXED_SIZE = "fixed_size"

@dataclass
class ChunkMetadata:
    chunk_id: str
    document_id: str
    chunk_index: int
    document_type: DocumentType
    strategy_used: ChunkingStrategy
    token_count: int
    section_title: Optional[str] = None
    code_language: Optional[str] = None

class DocumentClassifier:
    """Intelligent document type classifier"""
    
    def __init__(self):
        self.patterns = {
            DocumentType.API_REFERENCE: [
                r'(?i)\b(endpoint|api|rest|graphql|request|response|parameter)\b',
                r'(?i)\b(get|post|put|delete|patch)\s+/',
                r'(?i)\b(json|xml|swagger|openapi)\b',
                r'(?i)\b(authentication|authorization|token)\b'
            ],
            DocumentType.TECHNICAL_DOC: [
                r'(?i)\b(architecture|design|implementation|configuration)\b',
                r'(?i)\b(system|component|module|service)\b',
                r'(?i)\b(requirements|specifications|technical)\b'
            ],
            DocumentType.SUPPORT_TICKET: [
                r'(?i)\b(issue|problem|bug|error|ticket)\b',
                r'(?i)\b(reproduce|steps|workaround|solution)\b',
                r'(?i)\b(customer|user|reported|priority)\b'
            ],
            DocumentType.POLICY: [
                r'(?i)\b(policy|procedure|guideline|compliance)\b',
                r'(?i)\b(must|shall|should|required|mandatory)\b',
                r'(?i)\b(approval|review|governance|standard)\b'
            ],
            DocumentType.TUTORIAL: [
                r'(?i)\b(tutorial|guide|how-to|step-by-step)\b',
                r'(?i)\b(example|demo|walkthrough|getting started)\b',
                r'(?i)\b(learn|install|setup|configure)\b'
            ],
            DocumentType.CODE: [
                r'(?i)\b(function|class|method|variable|import)\b',
                r'(?i)\b(def|class|if|else|for|while|return)\b',
                r'(?i)\b(javascript|python|java|c\+\+|sql)\b'
            ]
        }
    
    def classify(self, content: str, filename: str = "") -> DocumentType:
        """Classify document based on content and filename patterns"""
        
        # Check filename patterns first
        filename_lower = filename.lower()
        if any(ext in filename_lower for ext in ['.py', '.js', '.java', '.cpp', '.sql']):
            return DocumentType.CODE
        if 'api' in filename_lower or 'swagger' in filename_lower:
            return DocumentType.API_REFERENCE
        if 'policy' in filename_lower or 'procedure' in filename_lower:
            return DocumentType.POLICY
        if 'tutorial' in filename_lower or 'guide' in filename_lower:
            return DocumentType.TUTORIAL
        
        # Content-based classification
        scores = {}
        for doc_type, patterns in self.patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, content))
                score += matches
            scores[doc_type] = score
        
        # Return the type with highest score, or UNKNOWN if all scores are low
        best_type = max(scores, key=scores.get)
        return best_type if scores[best_type] > 2 else DocumentType.UNKNOWN

class SemanticChunker:
    """Semantic-based chunking for policy docs and general content"""
    
    def __init__(self, max_tokens: int = 512):
        self.max_tokens = max_tokens
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def chunk(self, content: str, doc_type: DocumentType) -> List[str]:
        """Split content into semantic chunks"""
        
        # Split by sections first
        sections = self._split_by_sections(content)
        chunks = []
        
        for section in sections:
            if self._count_tokens(section) <= self.max_tokens:
                chunks.append(section)
            else:
                # Further split large sections
                sub_chunks = self._split_by_sentences(section)
                chunks.extend(sub_chunks)
        
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    
    def _split_by_sections(self, content: str) -> List[str]:
        """Split content by section headers"""
        # Look for markdown headers or numbered sections
        section_pattern = r'(?m)^(#{1,6}\s+.*|^\d+\.\s+.*|^[A-Z][A-Z\s]+:)'
        sections = re.split(section_pattern, content)
        
        # Combine headers with their content
        result = []
        for i in range(0, len(sections), 2):
            if i + 1 < len(sections):
                result.append(sections[i] + sections[i + 1])
            else:
                result.append(sections[i])
        
        return result
    
    def _split_by_sentences(self, content: str) -> List[str]:
        """Split content by sentences while respecting token limits"""
        sentences = re.split(r'(?<=[.!?])\s+', content)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if self._count_tokens(test_chunk) <= self.max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))

class CodeAwareChunker:
    """Code-aware chunking for technical documents and code"""
    
    def __init__(self, max_tokens: int = 512):
        self.max_tokens = max_tokens
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def chunk(self, content: str, doc_type: DocumentType) -> List[str]:
        """Split content preserving code blocks and structure"""
        
        # First, extract and preserve code blocks
        code_blocks = self._extract_code_blocks(content)
        
        # Split by code blocks
        chunks = []
        parts = re.split(r'```[\s\S]*?```', content)
        
        for i, part in enumerate(parts):
            if part.strip():
                # Regular content - use semantic chunking
                semantic_chunks = self._semantic_split(part)
                chunks.extend(semantic_chunks)
            
            # Add corresponding code block if it exists
            if i < len(code_blocks):
                code_chunk = code_blocks[i]
                if self._count_tokens(code_chunk) <= self.max_tokens:
                    chunks.append(code_chunk)
                else:
                    # Split large code blocks by functions/classes
                    sub_chunks = self._split_code_block(code_chunk)
                    chunks.extend(sub_chunks)
        
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    
    def _extract_code_blocks(self, content: str) -> List[str]:
        """Extract code blocks from content"""
        pattern = r'```[\s\S]*?```'
        return re.findall(pattern, content)
    
    def _split_code_block(self, code_block: str) -> List[str]:
        """Split large code blocks by functions/classes"""
        # Remove code fence markers
        code_content = re.sub(r'^```.*\n|```$', '', code_block, flags=re.MULTILINE)
        
        # Split by function/class definitions
        patterns = [
            r'(?m)^def\s+\w+.*?(?=^def\s+|\Z)',  # Python functions
            r'(?m)^class\s+\w+.*?(?=^class\s+|\Z)',  # Python classes
            r'(?m)^function\s+\w+.*?(?=^function\s+|\Z)',  # JavaScript functions
            r'(?m)^public\s+\w+.*?(?=^public\s+|\Z)',  # Java methods
        ]
        
        chunks = []
        for pattern in patterns:
            matches = re.findall(pattern, code_content, re.DOTALL)
            if matches:
                chunks.extend(matches)
                break
        
        # If no structure found, split by lines
        if not chunks:
            lines = code_content.split('\n')
            current_chunk = ""
            for line in lines:
                test_chunk = current_chunk + "\n" + line if current_chunk else line
                if self._count_tokens(test_chunk) <= self.max_tokens:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append(f"```\n{current_chunk}\n```")
                    current_chunk = line
            
            if current_chunk:
                chunks.append(f"```\n{current_chunk}\n```")
        else:
            chunks = [f"```\n{chunk}\n```" for chunk in chunks]
        
        return chunks
    
    def _semantic_split(self, content: str) -> List[str]:
        """Split non-code content semantically"""
        chunker = SemanticChunker(self.max_tokens)
        return chunker.chunk(content, DocumentType.TECHNICAL_DOC)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))

class HierarchicalChunker:
    """Hierarchical chunking for structured documents"""
    
    def __init__(self, max_tokens: int = 512):
        self.max_tokens = max_tokens
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def chunk(self, content: str, doc_type: DocumentType) -> List[str]:
        """Split content hierarchically preserving structure"""
        
        # Parse document structure
        structure = self._parse_structure(content)
        
        # Create chunks with hierarchical context
        chunks = []
        for section in structure:
            section_chunks = self._chunk_section(section)
            chunks.extend(section_chunks)
        
        return chunks
    
    def _parse_structure(self, content: str) -> List[Dict[str, Any]]:
        """Parse document structure"""
        lines = content.split('\n')
        structure = []
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if it's a header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2)
                
                if current_section:
                    structure.append(current_section)
                
                current_section = {
                    'level': level,
                    'title': title,
                    'content': []
                }
            else:
                if current_section:
                    current_section['content'].append(line)
                else:
                    # Content without header
                    if not structure or structure[-1]['level'] != 0:
                        structure.append({
                            'level': 0,
                            'title': 'Introduction',
                            'content': []
                        })
                    structure[-1]['content'].append(line)
        
        if current_section:
            structure.append(current_section)
        
        return structure
    
    def _chunk_section(self, section: Dict[str, Any]) -> List[str]:
        """Chunk a section with hierarchical context"""
        title = section['title']
        content = '\n'.join(section['content'])
        
        # Add title as context
        full_content = f"# {title}\n\n{content}"
        
        if self._count_tokens(full_content) <= self.max_tokens:
            return [full_content]
        
        # Split content while preserving title context
        chunks = []
        content_parts = content.split('\n\n')  # Split by paragraphs
        
        current_chunk = f"# {title}\n\n"
        
        for part in content_parts:
            test_chunk = current_chunk + part + '\n\n'
            
            if self._count_tokens(test_chunk) <= self.max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk != f"# {title}\n\n":
                    chunks.append(current_chunk.strip())
                current_chunk = f"# {title}\n\n{part}\n\n"
        
        if current_chunk != f"# {title}\n\n":
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))

class IntelligentChunker:
    """Main chunking orchestrator"""
    
    def __init__(self):
        self.classifier = DocumentClassifier()
        self.chunkers = {
            ChunkingStrategy.SEMANTIC: SemanticChunker(),
            ChunkingStrategy.CODE_AWARE: CodeAwareChunker(),
            ChunkingStrategy.HIERARCHICAL: HierarchicalChunker(),
            ChunkingStrategy.FIXED_SIZE: SemanticChunker(max_tokens=256)
        }
        
        # Strategy mapping based on document type
        self.strategy_map = {
            DocumentType.TECHNICAL_DOC: ChunkingStrategy.HIERARCHICAL,
            DocumentType.SUPPORT_TICKET: ChunkingStrategy.SEMANTIC,
            DocumentType.API_REFERENCE: ChunkingStrategy.CODE_AWARE,
            DocumentType.POLICY: ChunkingStrategy.HIERARCHICAL,
            DocumentType.TUTORIAL: ChunkingStrategy.HIERARCHICAL,
            DocumentType.CODE: ChunkingStrategy.CODE_AWARE,
            DocumentType.UNKNOWN: ChunkingStrategy.SEMANTIC
        }
    
    def process_document(self, content: str, filename: str = "", doc_id: str = None) -> Dict[str, Any]:
        """Process document with intelligent chunking"""
        
        if not doc_id:
            doc_id = hashlib.md5(content.encode()).hexdigest()[:8]
        
        # Classify document
        doc_type = self.classifier.classify(content, filename)
        
        # Select chunking strategy
        strategy = self.strategy_map[doc_type]
        
        # Apply chunking
        chunker = self.chunkers[strategy]
        chunks = chunker.chunk(content, doc_type)
        
        # Create metadata for each chunk
        chunk_metadata = []
        for i, chunk in enumerate(chunks):
            metadata = ChunkMetadata(
                chunk_id=f"{doc_id}_chunk_{i}",
                document_id=doc_id,
                chunk_index=i,
                document_type=doc_type,
                strategy_used=strategy,
                token_count=len(tiktoken.get_encoding("cl100k_base").encode(chunk)),
                section_title=self._extract_section_title(chunk)
            )
            chunk_metadata.append(metadata)
        
        return {
            'document_id': doc_id,
            'document_type': doc_type.value,
            'chunking_strategy': strategy.value,
            'chunks': chunks,
            'metadata': [
                {
                    'chunk_id': meta.chunk_id,
                    'chunk_index': meta.chunk_index,
                    'token_count': meta.token_count,
                    'section_title': meta.section_title
                } for meta in chunk_metadata
            ],
            'total_chunks': len(chunks),
            'processing_timestamp': datetime.now().isoformat()
        }
    
    def _extract_section_title(self, chunk: str) -> Optional[str]:
        """Extract section title from chunk"""
        lines = chunk.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                return line.lstrip('#').strip()
        return None

# Global chunker instance
chunker = IntelligentChunker()

# API Models
class DocumentRequest(BaseModel):
    content: str
    filename: Optional[str] = ""
    document_id: Optional[str] = None

class ChunkingResponse(BaseModel):
    document_id: str
    document_type: str
    chunking_strategy: str
    chunks: List[str]
    metadata: List[Dict[str, Any]]
    total_chunks: int
    processing_timestamp: str

# API Endpoints
@app.post("/chunk", response_model=ChunkingResponse)
async def chunk_document(request: DocumentRequest):
    """Chunk a document using intelligent chunking"""
    try:
        result = chunker.process_document(
            content=request.content,
            filename=request.filename or "",
            doc_id=request.document_id
        )
        return ChunkingResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chunk-file")
async def chunk_file(file: UploadFile = File(...)):
    """Chunk an uploaded file"""
    try:
        content = await file.read()
        content_str = content.decode('utf-8')
        
        result = chunker.process_document(
            content=content_str,
            filename=file.filename or ""
        )
        return ChunkingResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document-types")
async def get_document_types():
    """Get available document types"""
    return {"document_types": [dt.value for dt in DocumentType]}

@app.get("/chunking-strategies")
async def get_chunking_strategies():
    """Get available chunking strategies"""
    return {"strategies": [cs.value for cs in ChunkingStrategy]}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)