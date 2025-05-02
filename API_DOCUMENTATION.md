# RAG System API Documentation

This document provides an overview of the Retrieval Augmented Generation (RAG) System API endpoints, their parameters, and example usages. It also explains how input is validated.

## Base URL

```
http://localhost:8000
```

---

## Core Endpoints

### Get API Information

```
GET /
```

Returns basic information about the API.

**Response:**
```json
{
  "name": "RAG System API",
  "version": "1.0.0",
  "description": "API for transcription processing and retrieval-augmented generation"
}
```

---

### List Available Models

```
GET /models
```

Lists all available language models from Ollama.

**Response:**
```json
{
  "models": [
    {"name": "llama2", "size": 3.8, "modified_at": "2025-04-30T12:00:00Z"},
    {"name": "qwen2.5:3b-instruct", "size": 1.8, "modified_at": "2025-04-28T15:30:00Z"}
  ]
}
```

---

## Document Processing

### Process Transcription

```
POST /process-transcription
```

Processes a transcription with metadata and stores it in the vector database.

**Parameters:**
- `input_data` (TranscriptionInput, required): Contains transcription text and metadata
- `model` (string, optional): LLM model to use for keyword extraction, defaults to system default

**Request Body:**
```json
{
  "transcription": "The quarterly meeting reviewed our Q1 2024 financial results...",
  "metadata": {
    "filename": "quarterly_meeting_2024Q1.mp3",
    "file_type": "audio",
    "author": "John Smith",
    "size_bytes": 15000000,
    "additional_metadata": {
      "meeting_title": "Q1 2024 Financial Review",
      "department": "Finance",
      "participants": "John Smith, Jane Doe, Robert Johnson",
      "duration_minutes": 45,
      "transcription_quality": "high"
    }
  }
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Processed 3 chunks",
  "chunk_ids": ["adc4eeca-0b70-465c-834a-d554e4bcdef2", "..."],
  "processing_time_seconds": 2.35
}
```

For large files (>10,000 characters), processing happens in the background:

```json
{
  "status": "processing",
  "message": "Processing quarterly_meeting_2024Q1.mp3 in background",
  "file_id": "quarterly_meeting_2024Q1.mp3"
}
```

---

### List Processed Files

```
GET /processed-files
```

Returns information about all processed files.

**Response:**
```json
{
  "files": {
    "quarterly_meeting_2024Q1.mp3": {
      "chunk_count": 3,
      "processing_time_seconds": 2.35,
      "modellm_used": "qwen2.5:3b-instruct",
      "file_type": "audio",
      "processed_at": 1714639987.895848
    }
  }
}
```

---

## Document Management

### List All Documents

```
GET /documents
```

Returns a list of all unique filenames in the RAG system.

**Response:**
```json
{
  "count": 3,
  "filenames": [
    "financial_report_2024Q1.pdf",
    "meeting_minutes_apr2024.txt",
    "quarterly_meeting_2024Q1.mp3"
  ]
}
```

---

### Delete a Document

```
DELETE /documents/{filename}
```

Deletes a document and its associated chunks from the system.

**Path Parameter:**
- `filename` (string, required): The name of the file to delete

**Response:**
```json
{
  "status": "success",
  "message": "Deleted document quarterly_meeting_2024Q1.mp3 with 3 chunks"
}
```

---

### Reset the System

```
POST /system/reset?confirm=true
```

Resets the entire RAG system, deleting all documents and clearing the vector store.

**Query Parameter:**
- `confirm` (boolean, required): Must be set to `true` to confirm the reset

**Response:**
```json
{
  "status": "success",
  "message": "RAG system reset successfully. All documents and indices have been removed."
}
```

If `confirm=true` is not provided, the response will be:
```json
{
  "detail": "Confirmation required. Set 'confirm=true' to proceed with system reset."
}
```

---

## Querying and Generation

### Query the RAG System

```
POST /query
```

Queries the system with natural language and returns relevant document chunks with an AI-generated answer.

**Request Body:**
```json
{
  "query": "What were the Q1 2024 financial results?",
  "filter_metadata": {
    "file_type": "audio",
    "author": "John Smith"
  },
  "model": "qwen2.5:3b-instruct",
  "top_k": 3
}
```

**Response:**
```json
{
  "answer": "The Q1 2024 financial results showed a revenue of $5.2 million...",
  "relevant_chunks": [
    {
      "chunk_id": "adc4eeca-0b70-465c-834a-d554e4bcdef2",
      "content": "In our Q1 2024 financial review...",
      "metadata": {
        "filename": "quarterly_meeting_2024Q1.mp3",
        "file_type": "audio",
        "author": "John Smith"
      },
      "keywords": ["q1 2024", "financial", "revenue"]
    }
  ],
  "model_used": "qwen2.5:3b-instruct"
}
```

---

### Generate Text

```
POST /generate
```

Generates text directly using the Ollama LLM, optionally including RAG context and citations.

**Request Body:**
```json
{
  "prompt": "Write a summary of quarterly financial performance",
  "system_prompt": "You are a financial analyst with expertise in quarterly reports",
  "model": "qwen2.5:3b-instruct",
  "temperature": 0.3,
  "max_tokens": 1500,
  "use_rag": true,
  "top_k": 3
}
```

**Response:**
```json
{
  "response": "The quarterly financial performance showed a revenue of $5.2 million...",
  "sources": [
    {
      "id": 1,
      "chunk_id": "adc4eeca-0b70-465c-834a-d554e4bcdef2",
      "metadata": {
        "filename": "quarterly_meeting_2024Q1.mp3",
        "file_type": "audio",
        "author": "John Smith"
      },
      "content_snippet": "In our Q1 2024 financial review..."
    }
  ]
}
```

---

## Input Validation

### General Validation Rules
- **Required Fields**: Missing required fields will result in a `400 Bad Request` error.
- **Type Checking**: Fields are validated against their expected types (e.g., `string`, `integer`, `boolean`).
- **Custom Validation**: For example, the `/system/reset` endpoint requires `confirm=true` to proceed.

### Example Error Response
If a required field is missing:
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "query"],
      "msg": "Field required",
      "input": null
    }
  ]
}
```