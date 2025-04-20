---
title: Research Agent
emoji: 📉
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: apache-2.0
---

# Research Agent

A document-based Q&A application built with LangChain and Chainlit that allows users to upload documents and ask questions about their content.

## Features

- Upload PDF or text documents
- Ask questions about the uploaded documents
- Get AI-generated answers based on the document content
- Streaming responses for better user experience

## Technology Stack

- **LangChain**: Framework for developing applications powered by language models
- **Chainlit**: Frontend for creating chat-based applications
- **Qdrant**: Vector database for storing and retrieving document embeddings
- **OpenAI**: Provides the language model and embeddings

## How It Works

1. User uploads a PDF or text document
2. The application processes the document:
   - Splits it into manageable chunks
   - Creates embeddings using OpenAI
   - Stores these embeddings in Qdrant vector database
3. User asks questions about the document
4. The application:
   - Retrieves relevant chunks using semantic search
   - Uses a Retrieval-Augmented Generation (RAG) pipeline to generate answers
   - Returns streaming responses to the user

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd AIE6-ResearchAgent

# Install dependencies
pip install -r requirements.txt
# Or using uv
uv add langchain langchain-openai langchain-community langchain-core langchain-text-splitters langchain-qdrant qdrant-client chainlit
```

### Running the Application

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=your-api-key

# Start the application
chainlit run app.py
```

## Docker Deployment

The application can also be deployed using Docker:

```bash
docker build -t research-agent .
docker run -p 7860:7860 -e OPENAI_API_KEY=your-api-key research-agent
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
