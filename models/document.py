from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime

class FileMetadata(BaseModel):
    """Metadata for the uploaded file."""
    filename: str
    file_type: str
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    author: Optional[str] = None
    size_bytes: Optional[int] = None
    additional_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class TranscriptionInput(BaseModel):
    """Input model for transcription text and metadata."""
    transcription: str
    metadata: FileMetadata
    
class DocumentChunk(BaseModel):
    """A chunk of a document with its extracted information."""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    keywords: List[str] = Field(default_factory=list)
    
class RAGResponse(BaseModel):
    """Response from the RAG system."""
    answer: str
    source_chunks: List[DocumentChunk]
    modellm_used: str
    processing_time_ms: float

class QueryRequest(BaseModel):
    query: str
    filter_metadata: Optional[Dict[str, Any]] = None
    model: str 
    top_k: int = 5

class KeywordSearchRequest(BaseModel):
    keywords: List[str]
    filter_metadata: Optional[Dict[str, Any]] = None
    model: str
    top_k: int = 5
    query: Optional[str] = None

class GenerateRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    model: str
    temperature: float = 0.7
    max_tokens: int = 2000
    use_rag: bool = False  # Whether to use RAG context
    top_k: int = 3  # Number of chunks to retrieve when using RAG