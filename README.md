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

An advanced research assistant that combines web search, academic papers, and document analysis to help with comprehensive research tasks.

## Features

- **Multi-source Research**: Use web search (Tavily), arXiv papers, and uploaded documents
- **Upload PDF or text documents** for document-specific research
- **Research Process Transparency**: View the agent's research steps
- **Comprehensive Reports**: Receive structured reports with citations
- **Concept Breakdown**: Get clear explanations of key concepts
- **Streaming Responses**: Experience real-time answer generation

## Technology Stack

- **LangChain**: Framework for developing applications powered by language models
- **Chainlit**: Frontend for creating chat-based applications
- **Qdrant**: Vector database for storing and retrieving document embeddings
- **OpenAI**: Provides the language model and embeddings
- **Tavily**: Web search API for real-time information
- **arXiv**: Integration for academic paper search

## How It Works

1. **Ask a research question or upload a document**: Start with a question or upload a related document
2. **Multi-tool research process**: 
   - Searches the web using Tavily for current information
   - Queries arXiv for relevant academic papers
   - Analyzes uploaded documents using RAG (if provided)
3. **Comprehensive analysis**: 
   - Breaks down key concepts from research
   - Organizes information into a structured report
   - Provides proper citations for all sources
4. **Conclusion and action items**: Summarizes findings and suggests next steps

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key
- Tavily API key (for web search)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd AIE6-ResearchAgent

# Install dependencies using uv
uv add langchain langchain-openai langchain-community langchain-core langchain-text-splitters langchain-qdrant qdrant-client chainlit tavily-python arxiv langchain-experimental
```

### Environment Variables

```bash
# Set your API keys
export OPENAI_API_KEY=your-openai-api-key
export TAVILY_API_KEY=your-tavily-api-key
```

### Running the Application

```bash
# Start the application
chainlit run app.py
```

## Docker Deployment

The application can also be deployed using Docker:

```bash
docker build -t research-agent .
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=your-openai-api-key \
  -e TAVILY_API_KEY=your-tavily-api-key \
  research-agent
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
