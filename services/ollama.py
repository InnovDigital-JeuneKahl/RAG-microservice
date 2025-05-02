import requests
import json
import time
from typing import List, Dict, Any, Optional
from config import OLLAMA_BASE_URL, DEFAULT_LLM  # Changed from ..config

class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        """Initialize the Ollama client."""
        self.base_url = base_url
        
    def list_models(self) -> List[str]:
        """Get list of available models."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                return [model["name"] for model in response.json()["models"]]
            return []
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
    
    def generate(
        self, 
        prompt: str, 
        model: str = DEFAULT_LLM,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """Generate text using the specified model."""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        try:
            start_time = time.time()
            response = requests.post(url, json=payload)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                result["processing_time_ms"] = (end_time - start_time) * 1000
                return result
            else:
                return {
                    "error": f"Failed to generate: {response.status_code}",
                    "response": response.text,
                    "processing_time_ms": (end_time - start_time) * 1000
                }
        except Exception as e:
            return {
                "error": f"Exception during generation: {str(e)}",
                "processing_time_ms": 0
            }
            
    def generate_with_context(
        self,
        query: str,
        context: List[str],
        model: str = DEFAULT_LLM,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """Generate a response with provided context."""
        # Create a prompt with context
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context. 
        If you don't know the answer or it's not in the context, say so. Do not make up information."""
        
        context_text = "\n\n".join(context)
        prompt = f"""Context information is below.
        ---------------------
        {context_text}
        ---------------------
        
        Given the context information and not prior knowledge, answer the query.
        Query: {query}
        Answer:"""
        
        return self.generate(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )