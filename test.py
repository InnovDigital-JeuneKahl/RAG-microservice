import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.document import TranscriptionInput, FileMetadata
from services.rag import RAGService
from datetime import datetime

# Initialize the RAG service
rag_service = RAGService(use_ollama_for_keywords=True)

# Create sample metadata
metadata = FileMetadata(
    filename="quarterly_meeting_2024Q1.mp3",
    file_type="audio",
    created_at=datetime.now(),
    author="John Smith",
    size_bytes=15000000,
    additional_metadata={
        "meeting_title": "Q1 2024 Financial Review",
        "department": "Finance",
        "participants": "John Smith, Jane Doe, Robert Johnson",
        "duration_minutes": 45,
        "transcription_quality": "high"
    }
)

# Sample transcription text
transcription = """
Welcome to our Q1 2024 financial review meeting. I'm John Smith and today we'll discuss our quarterly performance.

Revenue for Q1 2024 was $5.2 million, which represents a 15% increase over Q1 2023. Our e-commerce division saw the strongest growth at 22%, while our retail division grew by 8%.

Operating expenses were $3.7 million, up 10% from last year due to our expansion into the European market. The new Berlin office accounted for approximately $450,000 of these expenses.

Our gross profit margin improved to 42%, up from 38% last year, primarily due to improved supply chain efficiencies and better vendor negotiations.

R&D spending was $850,000, focused mainly on our new AI-powered recommendation engine, which we expect to launch in Q3.

Marketing spend was $650,000, with digital marketing accounting for 70% of that budget.

Jane, could you please discuss the customer acquisition metrics?

[Jane] Thank you, John. Our customer acquisition cost decreased by 12% to $38 per customer. We acquired 28,000 new customers this quarter, with 65% coming through organic channels.

Our customer retention rate remained strong at 78%, and our Net Promoter Score increased to 62, up from 58 last quarter.

The new loyalty program launched in February has already seen 45,000 sign-ups, exceeding our target by 50%.

[John] Great, thanks Jane. Robert, could you cover the cash position and outlook?

[Robert] Sure. Our cash balance at the end of Q1 was $12.4 million, up from $11.2 million at year-end. Operating cash flow was positive at $1.8 million.

We invested $600,000 in capital expenditures, primarily for the new warehouse management system.

Looking ahead to Q2, we forecast revenue growth of 16-18%, with operating margins expected to remain stable. We're monitoring inflationary pressures in our supply chain, but so far we've been able to mitigate these through our hedging strategy.

[John] Thank you, Robert. Any questions from the team?
"""

# Create input object
input_data = TranscriptionInput(
    transcription=transcription,
    metadata=metadata
)

# Process the transcription using Qwen
chunk_ids = rag_service.process_transcription(input_data, model="qwen2.5:3b-instruct")
print(f"Processed {len(chunk_ids)} chunks")

# Query the system
response = rag_service.query(
    query="What was the Q1 revenue and how does it compare to last year?",
    model="qwen2.5:3b-instruct"
)

print(f"\nQuery: What was the Q1 revenue and how does it compare to last year?")
print(f"Answer: {response.answer}")
print(f"Model: {response.modellm_used}")
print(f"Processing time: {response.processing_time_ms:.2f} ms")

# Try keyword search
keyword_response = rag_service.keyword_search(
    keywords=["customer acquisition", "marketing", "retention"],
    model="gemma3:4b"
)

print(f"\nKeyword search: customer acquisition, marketing, retention")
print(f"Answer: {keyword_response.answer}")
print(f"Model: {keyword_response.modellm_used}")
print(f"Processing time: {keyword_response.processing_time_ms:.2f} ms")