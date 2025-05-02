import os
import uuid
from typing import List, Dict, Any, Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import VECTOR_DB_PATH, EMBEDDING_MODEL
from models.document import DocumentChunk
import datetime
from langchain_community.vectorstores.utils import filter_complex_metadata

class VectorStore:
    """Service for managing the vector database."""
    
    def __init__(self):
        """Initialize the vector store with the specified embedding model."""
        # Initialize embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # Initialize Chroma DB
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        self.vectordb = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=self.embeddings
        )
        
    def add_chunks(self, chunks: List[DocumentChunk]) -> List[str]:
        """Add document chunks to the vector store."""
        texts = [chunk.content for chunk in chunks]
        metadatas = []
        
        for chunk in chunks:
            # Prepare metadata dictionary with keywords as a special field
            metadata = chunk.metadata.copy()
            metadata["keywords"] = ", ".join(chunk.keywords)
            metadata["chunk_id"] = chunk.chunk_id
            
            # Convert complex types to strings and None to empty string
            for key, value in list(metadata.items()):
                if value is None:
                    metadata[key] = ""  # Convert None to empty string
                elif isinstance(value, datetime.datetime):
                    metadata[key] = value.isoformat()
                elif isinstance(value, dict):
                    metadata[key] = str(value)  # Convert dict to string
                elif not isinstance(value, (str, int, float, bool)):
                    metadata[key] = str(value)
            
            metadatas.append(metadata)
        
        # Add documents to vector store
        ids = [chunk.chunk_id for chunk in chunks]
        print(f"Adding {len(chunks)} chunks to vector store with IDs: {ids}")
        
        try:
            self.vectordb.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            return ids
        except ValueError as e:
            print(f"Error adding chunks to vector store: {e}")
            raise e
        
    def search(self, query: str, filter_metadata: Optional[Dict[str, Any]] = None, top_k: int = 5) -> List[DocumentChunk]:
        """Search for relevant document chunks based on query and optional filters."""
        # Format filter for ChromaDB
        chroma_filter = None
        if filter_metadata:
            if len(filter_metadata) == 1:
                # Single condition can be used as-is
                chroma_filter = filter_metadata
            else:
                # Multiple conditions need to be combined with $and
                chroma_filter = {
                    "$and": [
                        {key: value} for key, value in filter_metadata.items()
                    ]
                }
        
        results = self.vectordb.similarity_search(
            query=query,
            k=top_k,
            filter=chroma_filter
        )
        
        chunks = []
        for doc in results:
            # Extract keywords from metadata
            metadata = doc.metadata.copy()
            keywords_str = metadata.pop("keywords", "")
            keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]
            
            chunk = DocumentChunk(
                chunk_id=metadata.pop("chunk_id", str(uuid.uuid4())),
                content=doc.page_content,
                metadata=metadata,
                keywords=keywords
            )
            chunks.append(chunk)
            
        return chunks
    
    def search_by_keywords(self, keywords: List[str], filter_metadata: Optional[Dict[str, Any]] = None, top_k: int = 5) -> List[DocumentChunk]:
        """Search for documents that contain specific keywords."""
        # If no keywords, just use regular search
        if not keywords:
            return self.search(
                query="",
                filter_metadata=filter_metadata,
                top_k=top_k
            )
        
        # Create a simple query string from the keywords
        query_text = " ".join(keywords)
        
        # Just use the standard search method, which will perform semantic search
        # We won't use explicit keyword filtering since ChromaDB doesn't support
        # substring matching operators like $contains
        results = self.search(
            query=query_text,
            filter_metadata=filter_metadata,
            top_k=top_k
        )
        
        return results
    
    def delete_by_ids(self, chunk_ids: List[str]) -> None:
        """Delete chunks by their IDs."""
        try:
            self.vectordb.delete(ids=chunk_ids)
            self.vectordb.persist()
        except Exception as e:
            print(f"Error deleting chunks: {e}")
            raise e

    def clear(self) -> None:
        """Clear all documents from the vector store."""
        try:
            # This will delete all documents but keep the collection
            self.vectordb.delete_collection()
        except Exception as e:
            print(f"Error clearing vector store: {e}")
            raise e