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