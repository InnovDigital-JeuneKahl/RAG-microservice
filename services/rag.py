import uuid
import time
from typing import List, Dict, Any, Optional
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP, MAX_KEYWORDS_PER_CHUNK, DEFAULT_LLM  # Changed from ..config
from models.document import DocumentChunk, TranscriptionInput, RAGResponse  # Changed from ..models
from services.keyword import KeywordExtractor  # Changed from .keyword
from services.vectorstore import VectorStore  # Changed from .vectorstore
from services.ollama import OllamaClient  # Changed from .ollama


class RAGService:
    """Main service for the RAG pipeline."""
    
    def __init__(self, use_ollama_for_keywords: bool = True):
        """Initialize the RAG service components.
        
        Args:
            use_ollama_for_keywords: Whether to use Ollama for keyword extraction
        """
        self.keyword_extractor = KeywordExtractor(use_ollama=use_ollama_for_keywords)
        self.vector_store = VectorStore()
        self.ollama_client = OllamaClient()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP,
        )
    
    def process_transcription(self, input_data: TranscriptionInput, model: str = DEFAULT_LLM) -> List[str]:
        """Process transcription and store in vector database.
        
        Args:
            input_data: The transcription and metadata
            model: The LLM model to use for keyword extraction
            
        Returns:
            List of chunk IDs created
        """
        # Extract text and metadata
        text = input_data.transcription
        metadata = input_data.metadata.dict()
        
        # Split text into chunks
        text_chunks = self.text_splitter.split_text(text)
        
        # Extract metadata keywords once
        metadata_keywords = self.keyword_extractor.extract_metadata_keywords(metadata, model)
        
        # Process each chunk
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            # Extract keywords from chunk using the specified model
            content_keywords = self.keyword_extractor.extract_keywords(
                chunk_text, 
                max_keywords=MAX_KEYWORDS_PER_CHUNK,
                model=model
            )
            
            # Combine content and metadata keywords
            combined_keywords = list(set(content_keywords + metadata_keywords))
            if len(combined_keywords) > MAX_KEYWORDS_PER_CHUNK:
                combined_keywords = combined_keywords[:MAX_KEYWORDS_PER_CHUNK]
            
            # Create chunk metadata (include original metadata plus chunk-specific info)
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["chunk_count"] = len(text_chunks)
            
            # Create document chunk
            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                content=chunk_text,
                metadata=chunk_metadata,
                keywords=combined_keywords
            )
            chunks.append(chunk)
        
        # Add chunks to vector store
        chunk_ids = self.vector_store.add_chunks(chunks)
        
        return chunk_ids
    
    def query(
        self, 
        query: str, 
        filter_metadata: Optional[Dict[str, Any]] = None,
        model: str = DEFAULT_LLM,
        top_k: int = 5
    ) -> RAGResponse:
        """Query the RAG system.
        
        Args:
            query: The user query
            filter_metadata: Optional metadata filters
            model: The LLM model to use for generation
            top_k: Number of documents to retrieve
            
        Returns:
            RAG response with answer and source chunks
        """
        start_time = time.time()
        
        # Extract query keywords to improve retrieval
        query_keywords = self.keyword_extractor.extract_keywords(
            text=query,
            max_keywords=5,
            model=model
        )
        
        # Search for relevant document chunks
        relevant_chunks = self.vector_store.search(
            query=query,
            filter_metadata=filter_metadata,
            top_k=top_k
        )
        
        # Enhance retrieval with keyword-based search
        if query_keywords:
            keyword_chunks = self.vector_store.search_by_keywords(
                keywords=query_keywords,
                filter_metadata=filter_metadata,
                top_k=top_k
            )
            
            # Merge and deduplicate chunks
            seen_chunk_ids = set(chunk.chunk_id for chunk in relevant_chunks)
            for chunk in keyword_chunks:
                if chunk.chunk_id not in seen_chunk_ids:
                    relevant_chunks.append(chunk)
                    seen_chunk_ids.add(chunk.chunk_id)
            
            # Keep only top_k chunks if we have more
            if len(relevant_chunks) > top_k:
                relevant_chunks = relevant_chunks[:top_k]
        
        # Extract content from chunks for context
        context = [chunk.content for chunk in relevant_chunks]
        
        # Generate response using Ollama
        result = self.ollama_client.generate_with_context(
            query=query,
            context=context,
            model=model
        )
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Create response
        response = RAGResponse(
            answer=result.get("response", "Failed to generate response"),
            source_chunks=relevant_chunks,
            modellm_used=model,
            processing_time_ms=processing_time_ms
        )
        
        return response
    
    def keyword_search(
        self,
        keywords: List[str],
        filter_metadata: Optional[Dict[str, Any]] = None,
        model: str = DEFAULT_LLM,
        top_k: int = 5,
        query: Optional[str] = None
    ) -> RAGResponse:
        """Search by keywords and generate a response.
        
        Args:
            keywords: List of keywords to search for
            filter_metadata: Optional metadata filters
            model: The LLM model to use for generation
            top_k: Number of documents to retrieve
            query: Optional query to use instead of default summary prompt
            
        Returns:
            RAG response with answer and source chunks
        """
        start_time = time.time()
        
        # Search for documents with matching keywords
        relevant_chunks = self.vector_store.search_by_keywords(
            keywords=keywords,
            filter_metadata=filter_metadata,
            top_k=top_k
        )
        
        # Extract content from chunks for context
        context = [chunk.content for chunk in relevant_chunks]
        
        # Generate response using Ollama
        # If query is not provided, summarize the documents
        if not query:
            query = f"Provide a summary of the information related to: {', '.join(keywords)}"
            
        result = self.ollama_client.generate_with_context(
            query=query,
            context=context,
            model=model
        )
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Create response
        response = RAGResponse(
            answer=result.get("response", "Failed to generate response"),
            source_chunks=relevant_chunks,
            modellm_used=model,
            processing_time_ms=processing_time_ms
        )
        
        return response