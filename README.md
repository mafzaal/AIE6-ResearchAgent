---
title: Research Agent
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: apache-2.0
---

# Research Agent

An advanced research assistant that combines web search, academic papers, RSS feeds, and document analysis to help with comprehensive research tasks.

## Features

- **Multi-source Research**: Use web search (Tavily/DuckDuckGo), arXiv papers, RSS feeds, and uploaded documents
- **Upload PDF or text documents** for document-specific research
- **Research Process Transparency**: View the agent's step-by-step research process
- **Comprehensive Reports**: Receive structured reports with citations
- **Concept Breakdown**: Get clear explanations of key concepts
- **Streaming Responses**: Experience real-time answer generation
- **LangGraph Workflow**: Enhanced agent reasoning and tool orchestration

## Available Research Tools

The agent uses several specialized tools to conduct comprehensive research:

1. **Web Search (Tavily & DuckDuckGo)**: Searches the internet for current information and real-time data
   - Uses Tavily and DuckDuckGo APIs to perform semantic search across the web
   - Returns up to 5 relevant results by default
   - Ideal for current events, trends, and general information

2. **Academic Research (arXiv)**: Searches academic papers and scholarly articles
   - Connects to arXiv's database of scientific papers
   - Returns up to 5 relevant papers by default with titles, authors, and abstracts
   - Ideal for scientific research, technical topics, and academic information

3. **RSS Feed Reader**: Fetches and processes articles from RSS feeds
   - Retrieves content from multiple RSS feed URLs
   - Supports filtering by search query
   - Provides article metadata including title, source, link, and publication date
   - Optional NLP processing for summaries and keywords extraction
   - Ideal for staying updated on specific news sources and blogs

4. **Document Analysis (RAG)**: Analyzes user-uploaded documents
   - Uses Retrieval Augmented Generation (RAG) to answer questions about uploaded files
   - Supports PDF and text file formats
   - Breaks documents into chunks and creates vector embeddings for semantic search
   - Uses Qdrant as the vector database for document storage and retrieval

## Agent Architecture

The Research Agent uses LangGraph for workflow orchestration, providing a robust framework for reasoning and tool usage:

```
┌───────────────┐      ┌─────────────┐      ┌───────────────┐
│               │      │             │      │               │
│    Retrieve   ├─────►│    Agent    │◄─────┤     Tool      │
│     Node      │      │     Node    │      │     Node      │
│               │      │             │      │               │
└───────────────┘      └──────┬──────┘      └───────────────┘
                              │
                              │
                              ▼
                       ┌──────────────┐
                       │              │
                       │  End State   │
                       │              │
                       └──────────────┘
```

- **Retrieve Node**: Gathers context from uploaded documents
- **Agent Node**: Processes user requests and determines required actions
- **Tool Node**: Executes specific tools like web search, arXiv queries, or RSS feed retrieval
- **End State**: Delivers final response after gathering and synthesizing information

## Research Process

When you ask a question, the agent:
1. Determines which tools are most appropriate for your query
2. Executes searches across selected tools (web, academic, RSS feeds)
3. Retrieves relevant context from uploaded documents (if any)
4. Shows you each step of the research process for transparency
5. Analyzes and synthesizes information from all sources
6. Provides a comprehensive response with citations to sources

## Technology Stack

- **LangChain**: Framework for developing applications powered by language models
- **LangGraph**: Advanced workflow orchestration for LLM applications
- **Chainlit**: Frontend for creating chat-based applications
- **Qdrant**: Vector database for storing and retrieving document embeddings
- **OpenAI**: Provides the GPT-4o language model and embeddings
- **Tavily/DuckDuckGo**: Web search APIs for real-time information
- **arXiv**: Integration for academic paper search
- **Feedparser**: Library for parsing RSS feeds

## How It Works

1. **Ask a research question or upload a document**: Start with a question or upload a related document
2. **Multi-tool research process**: 
   - Searches the web using Tavily/DuckDuckGo for current information
   - Queries arXiv for relevant academic papers
   - Fetches articles from RSS feeds when specific news sources are needed
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
uv sync
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
