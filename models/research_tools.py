"""
Research tools implementation for the agent.

This module implements input schemas and tools specifically for research purposes.
"""
from typing import List, Optional
from pydantic import BaseModel, Field

from langchain_core.tools import Tool

class ArxivQueryInput(BaseModel):
    """Input for arXiv query."""
    query: str = Field(..., description="The search query to find papers on arXiv")
    max_results: int = Field(default=5, description="The maximum number of results to return")

class RAGQueryInput(BaseModel):
    """Input for RAG query."""
    query: str = Field(..., description="The query to search in the uploaded document")

class WebSearchInput(BaseModel):
    """Input for web search."""
    query: str = Field(..., description="The search query for web search")
    max_results: int = Field(default=5, description="The maximum number of results to return")

class DocumentAnalysisInput(BaseModel):
    """Input for document analysis."""
    query: str = Field(..., description="The specific question to analyze in the document")
    include_citations: bool = Field(default=True, description="Whether to include citations in the response")