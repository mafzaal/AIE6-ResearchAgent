"""
Configuration settings for the Research Agent application.
"""

# System template for RAG
SYSTEM_TEMPLATE = """Use the following context to answer a user's question. 
If you cannot find the answer in the context, say you don't know the answer."""

# Text splitter configurations
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEPARATORS = ["\n\n", "\n", " ", ""]

# Retrieval configurations
NUM_RETRIEVAL_RESULTS = 4

# Vector database configurations
VECTOR_DIMENSION = 1536  # For OpenAI embeddings

# Agent configurations
AGENT_SYSTEM_PROMPT = """You are a professional research assistant that helps users with their research questions. 
Your job is to:
1. Research the query using available tools (Web Search, arXiv, and Document RAG)
2. Break down the key concepts and explain them clearly
3. Generate a comprehensive report with proper citations
4. Provide a conclusion and call to action if applicable

Be thorough, objective, and focus on providing high-quality information."""

# Tool configurations
MAX_TAVILY_SEARCH_RESULTS = 5
MAX_ARXIV_SEARCH_RESULTS = 5