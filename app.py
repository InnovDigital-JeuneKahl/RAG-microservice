from fastapi import FastAPI, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional
import uvicorn
import time
import os
import json

from config import API_TITLE, API_DESCRIPTION, API_VERSION, DEFAULT_LLM, ALTERNATIVE_LLM, VECTOR_DB_PATH
from models.document import TranscriptionInput, RAGResponse, FileMetadata, QueryRequest, KeywordSearchRequest, GenerateRequest
from services.rag import RAGService
from services.ollama import OllamaClient


# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

# Initialize services
rag_service = RAGService(use_ollama_for_keywords=True)
ollama_client = OllamaClient()

# Track processed files for demo purposes
# File to store processed files data
PROCESSED_FILES_PATH = os.path.join(os.path.dirname(__file__), "data", "processed_files.json")
# Load existing processed files data
def load_processed_files():
    if os.path.exists(PROCESSED_FILES_PATH):
        try:
            with open(PROCESSED_FILES_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading processed files data: {e}")
    
    # If file doesn't exist or there's an error, return empty dict
    return {}

# Save processed files data to disk
def save_processed_files(data):
    # Ensure directory exists
    os.makedirs(os.path.dirname(PROCESSED_FILES_PATH), exist_ok=True)
    
    try:
        with open(PROCESSED_FILES_PATH, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving processed files data: {e}")

# Track processed files (now loaded from disk)
processed_files = load_processed_files()


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION
    }

@app.get("/models")
async def list_models():
    """List available models from Ollama."""
    models = ollama_client.list_models()
    return {"models": models}

@app.post("/process-transcription")
async def process_transcription(
    input_data: TranscriptionInput,
    background_tasks: BackgroundTasks,
    model: str = DEFAULT_LLM
):
    """Process and store transcription with metadata."""
    # For large files, process in background
    if len(input_data.transcription) > 10000:
        background_tasks.add_task(
            process_in_background,
            input_data=input_data,
            model=model
        )
        return {
            "status": "processing",
            "message": f"Processing {input_data.metadata.filename} in background",
            "file_id": input_data.metadata.filename
        }
    
    try:
        start_time = time.time()
        chunk_ids = rag_service.process_transcription(input_data, model)
        processing_time = time.time() - start_time
        
        # Store processing info
        processed_files[input_data.metadata.filename] = {
            "chunk_count": len(chunk_ids),
            "processing_time_seconds": processing_time,
            "modellm_used": model,
            "file_type": input_data.metadata.file_type,
            "processed_at": time.time()
        }
        
        # Save to disk
        save_processed_files(processed_files)
        
        return {
            "status": "success",
            "message": f"Processed {len(chunk_ids)} chunks",
            "chunk_ids": chunk_ids,
            "processing_time_seconds": processing_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing transcription: {str(e)}")

# New function to handle background processing
async def process_in_background(input_data: TranscriptionInput, model: str):
    try:
        start_time = time.time()
        chunk_ids = rag_service.process_transcription(input_data, model)
        processing_time = time.time() - start_time
        
        # Store processing info
        processed_files[input_data.metadata.filename] = {
            "chunk_count": len(chunk_ids),
            "processing_time_seconds": processing_time,
            "modellm_used": model,
            "file_type": input_data.metadata.file_type,
            "processed_at": time.time()
        }
        
        # Save to disk
        save_processed_files(processed_files)
        
        print(f"Background processing complete for {input_data.metadata.filename}")
    except Exception as e:
        print(f"Error in background processing: {e}")

@app.get("/processed-files")
async def get_processed_files():
    """Get list of processed files."""
    return {"files": processed_files}

@app.post("/query")
async def query(request: QueryRequest) -> RAGResponse:
    """Query the RAG system."""
    if request.model not in ollama_client.list_models():
        raise HTTPException(status_code=400, detail=f"Model {request.model} not available")
    try:
        print(f"Querying with model: {request.model}, top_k: {request.top_k}, filter_metadata: {request.filter_metadata}")
        response = rag_service.query(
            query=request.query,
            filter_metadata=request.filter_metadata,
            model=request.model,
            top_k=request.top_k
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying: {str(e)}")

@app.post("/keyword-search")
async def keyword_search(request: KeywordSearchRequest) -> RAGResponse:
    """Search by keywords and generate a response."""
    try:
        response = rag_service.keyword_search(
            keywords=request.keywords,
            filter_metadata=request.filter_metadata,
            model=request.model,
            top_k=request.top_k,
            query=request.query
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching by keywords: {str(e)}")

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    """Generate text using Ollama directly or with RAG system context.
    
    If use_rag=True is set, the system will retrieve relevant documents
    and include them as context for generation, with citations.
    """
    try:
        # Check if we should use RAG for enhanced generation with citations
        if hasattr(request, 'use_rag') and request.use_rag:
            # Get relevant documents first
            relevant_chunks = rag_service.vector_store.search(
                query=request.prompt,
                top_k=request.top_k if hasattr(request, 'top_k') else 3
            )
            
            # Format context from retrieved chunks
            context = "\n\n".join([
                f"Document {i+1}: {chunk.content}" 
                for i, chunk in enumerate(relevant_chunks)
            ])
            
            # Create enhanced prompt with context
            enhanced_prompt = f"""Please answer based on the following information:

{context}

Question: {request.prompt}

Please include citations in your response using [Doc 1], [Doc 2], etc."""

            # Generate with context
            result = ollama_client.generate(
                prompt=enhanced_prompt,
                model=request.model,
                system_prompt=request.system_prompt or "You are a helpful assistant that provides accurate information with citations to the relevant sources.",
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            # Add metadata about sources to response
            if "response" in result:
                sources = []
                for i, chunk in enumerate(relevant_chunks):
                    source = {
                        "id": i+1,
                        "chunk_id": chunk.chunk_id,
                        "metadata": chunk.metadata,
                        "content_snippet": chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content
                    }
                    sources.append(source)
                
                result["sources"] = sources
            
            return result
        else:
            # Standard generation without RAG context
            result = ollama_client.generate(
                prompt=request.prompt,
                model=request.model,
                system_prompt=request.system_prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")
    

@app.get("/documents")
async def list_documents():
    """Get a list of all document filenames in the RAG system."""
    try:
        # Extract just the filenames from processed_files dictionary
        filenames = list(processed_files.keys())
        
        # Optionally, sort the filenames alphabetically
        filenames.sort()
        
        return {
            "count": len(filenames),
            "filenames": filenames
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving document list: {str(e)}")
    

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document and its chunks from the system."""
    try:
        # Check if file exists in our tracking
        if filename not in processed_files:
            raise HTTPException(status_code=404, detail=f"Document {filename} not found")
        
        # Query for chunks with this filename
        filter_metadata = {"filename": filename}
        chunks = rag_service.vector_store.search(
            query="",  # Empty query to match based on filter only
            filter_metadata=filter_metadata,
            top_k=100  # Set high to get all chunks
        )
        
        # Delete chunks from vector store
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if chunk_ids:
            rag_service.vector_store.delete_by_ids(chunk_ids)
        
        # Remove from processed files
        del processed_files[filename]
        save_processed_files(processed_files)
        
        return {
            "status": "success",
            "message": f"Deleted document {filename} with {len(chunk_ids)} chunks"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")
    
@app.post("/system/reset")
async def reset_system(confirm: bool = False):
    """Reset the entire RAG system, deleting all documents and clearing the vector store."""
    print(confirm)
    if not confirm:
        raise HTTPException(
            status_code=400, 
            detail="Confirmation required. Set 'confirm=true' to proceed with system reset."
        )
    
    try:
        # Clear vector store
        rag_service.vector_store.clear()
        
        # Reset processed files tracking
        global processed_files
        processed_files = {}
        save_processed_files(processed_files)
        
        # Ensure vector store directory exists but is empty
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        
        return {
            "status": "success",
            "message": "RAG system reset successfully. All documents and indices have been removed."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting system: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)