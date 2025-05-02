import spacy
from keybert import KeyBERT
from typing import List, Dict, Any, Optional
import nltk
from nltk.corpus import stopwords
import yake
from services.ollama import OllamaClient  # Changed from .ollama
from config import DEFAULT_LLM  # Changed from relative import

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

class KeywordExtractor:
    """Service for extracting keywords from text with multiple methods including Ollama LLM."""
    
    def __init__(self, use_ollama: bool = True):
        """Initialize keyword extractor.
        
        Args:
            use_ollama: Whether to use Ollama LLM for extraction (True) or traditional methods (False)
        """
        self.use_ollama = use_ollama
        
        if use_ollama:
            # Initialize Ollama client
            self.ollama_client = OllamaClient()
        
        # Initialize traditional extractors regardless, as a fallback
        # Load spaCy model for NER and noun phrases
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize KeyBERT for keyword extraction
        self.keybert = KeyBERT()
        
        # Initialize YAKE for additional keyword extraction
        self.kw_extractor = yake.KeywordExtractor(
            lan="en", 
            n=3,  # ngram size
            dedupLim=0.9,
            dedupFunc='seqm',
            windowsSize=1,
            top=20
        )
        
        # Get stopwords
        self.stopwords = set(stopwords.words('english'))
    
    def extract_keywords(self, text: str, max_keywords: int = 15, model: Optional[str] = None) -> List[str]:
        """Extract keywords from text using Ollama LLM or traditional methods.
        
        Args:
            text: The text to extract keywords from
            max_keywords: Maximum number of keywords to extract
            model: The LLM model to use (only relevant if use_ollama=True)
        
        Returns:
            List of extracted keywords
        """
        if self.use_ollama and model:
            try:
                # Try Ollama-based extraction first
                ollama_keywords = self.extract_keywords_with_ollama(text, max_keywords, model)
                if ollama_keywords:
                    return ollama_keywords
            except Exception as e:
                print(f"Ollama keyword extraction failed: {e}. Falling back to traditional methods.")
        
        # Fall back to traditional methods if Ollama fails or is disabled
        return self.extract_keywords_traditional(text, max_keywords)
    
    def extract_keywords_with_ollama(self, text: str, max_keywords: int = 15, model: str = DEFAULT_LLM) -> List[str]:
        """Extract keywords from text using Ollama LLM."""
        system_prompt = f"""You are a keyword extraction specialist. Extract the most important keywords from the given text.
        - Return ONLY keywords and key phrases, nothing else
        - Return at most {max_keywords} keywords
        - Focus on substantive terms that capture the main topics
        - Include important named entities, technical terms, and domain-specific concepts
        - Format as a comma-separated list with no additional text"""
        
        # Truncate text if too long (prevent token limits)
        max_text_length = 8000  # Adjust based on model's context window
        if len(text) > max_text_length:
            text = text[:max_text_length] + "..."
        
        prompt = f"Extract the key terms and concepts from this text:\n\n{text}"
        
        result = self.ollama_client.generate(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            temperature=0.1  # Low temperature for consistent results
        )
        
        if "response" in result:
            # Parse the comma-separated response
            keywords = [
                kw.strip().lower() 
                for kw in result["response"].split(",") 
                if kw.strip()
            ]
            
            # Remove duplicates while preserving order
            seen = set()
            unique_keywords = [
                kw for kw in keywords 
                if not (kw in seen or seen.add(kw))
            ]
            
            return unique_keywords[:max_keywords]
        
        return []  # Return empty list if extraction failed
    
    def extract_keywords_traditional(self, text: str, max_keywords: int = 15) -> List[str]:
        """Extract keywords from text using multiple traditional methods."""
        keywords = set()
        
        # Extract using KeyBERT
        keybert_keywords = self.keybert.extract_keywords(
            text, 
            keyphrase_ngram_range=(1, 3),
            stop_words='english',
            use_mmr=True,
            diversity=0.7,
            top_n=max_keywords//2
        )
        for kw, _ in keybert_keywords:
            keywords.add(kw.lower())
        
        # Extract using YAKE
        yake_keywords = self.kw_extractor.extract_keywords(text)
        for kw, _ in yake_keywords[:max_keywords//2]:
            keywords.add(kw.lower())
        
        # Extract named entities and noun chunks using spaCy
        doc = self.nlp(text)
        
        # Add named entities
        for ent in doc.ents:
            if len(keywords) >= max_keywords:
                break
            keywords.add(ent.text.lower())
        
        # Add important noun chunks
        for chunk in doc.noun_chunks:
            if len(keywords) >= max_keywords:
                break
            # Filter out chunks that are just stopwords
            if not all(token.text.lower() in self.stopwords for token in chunk):
                keywords.add(chunk.text.lower())
        
        return list(keywords)[:max_keywords]
    
    def extract_metadata_keywords(self, metadata: Dict[str, Any], model: Optional[str] = None) -> List[str]:
        """Extract keywords from metadata.
        
        Args:
            metadata: The metadata dictionary
            model: Optional Ollama model to use for extraction (for complex metadata)
        
        Returns:
            List of extracted keywords
        """
        keywords = set()
        
        # Extract from filename (without extension)
        if 'filename' in metadata:
            filename = metadata['filename']
            if '.' in filename:
                filename = filename[:filename.rindex('.')]
            filename_parts = filename.replace('_', ' ').replace('-', ' ').split()
            keywords.update([part.lower() for part in filename_parts if len(part) > 2])
        
        # Extract from author
        if 'author' in metadata and metadata['author']:
            keywords.add(metadata['author'].lower())
            
        # Extract from file type
        if 'file_type' in metadata:
            keywords.add(metadata['file_type'].lower())
            
        # Extract from additional metadata
        if 'additional_metadata' in metadata:
            # If there's complex additional metadata and Ollama is enabled, use it
            complex_metadata = False
            metadata_text = ""
            
            for key, value in metadata['additional_metadata'].items():
                if isinstance(value, str):
                    if len(value) < 50:  # Simple metadata
                        keywords.add(value.lower())
                    else:  # Complex metadata
                        complex_metadata = True
                        metadata_text += f"{key}: {value}\n"
            
            # Extract keywords from complex metadata using Ollama if available
            if complex_metadata and self.use_ollama and model:
                try:
                    ollama_keywords = self.extract_keywords_with_ollama(metadata_text, 10, model)
                    keywords.update(ollama_keywords)
                except:
                    pass  # Silently fail and stick with basic extraction
                
        return list(keywords)