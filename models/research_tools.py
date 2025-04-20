"""
Research tools implementation for the agent.

This module implements the Tavily search, arXiv, and RAG tools
that will be used by the research agent.
"""
import os
from typing import List, Dict, Any, Optional
from langchain.agents import tool
from langchain_core.tools import Tool
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.arxiv import ArxivAPIWrapper

import config
from models.rag import LangChainRAG

class ArxivQueryInput(BaseModel):
    """Input for arXiv query."""
    query: str = Field(..., description="The search query to find papers on arXiv")
    max_results: int = Field(default=config.MAX_ARXIV_SEARCH_RESULTS, description="The maximum number of results to return")

class RAGQueryInput(BaseModel):
    """Input for RAG query."""
    query: str = Field(..., description="The query to search in the uploaded document")

def create_tavily_search_tool() -> Tool:
    """Create a Tavily search tool for the agent."""
    # Check if TAVILY_API_KEY is in environment variables
    if "TAVILY_API_KEY" not in os.environ:
        print("Warning: TAVILY_API_KEY environment variable not set. Web search functionality may be limited.")
    
    return TavilySearchResults(max_results=config.MAX_TAVILY_SEARCH_RESULTS)

@tool
def arxiv_search(query: str, max_results: int = config.MAX_ARXIV_SEARCH_RESULTS) -> str:
    """
    Search for papers on arXiv.
    
    Args:
        query: The search query string
        max_results: Maximum number of results to return
        
    Returns:
        A string summary of the search results
    """
    client = ArxivAPIWrapper(
        top_k_results=max_results,
        ARXIV_MAX_QUERY_LENGTH=300,
        load_max_docs=max_results,
        load_all_available_meta=True
    )
    
    try:
        results = client.run(query)
        if not results:
            return "No papers found on arXiv for this query."
        
        formatted_results = []
        for idx, result in enumerate(results.split("\n\n")):
            if result.strip():
                formatted_results.append(f"[{idx+1}] {result.strip()}")
        
        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"Error searching arXiv: {str(e)}"

class ResearchToolkit:
    """
    A toolkit of research tools for the agent.
    """
    def __init__(self, rag_chain: Optional[LangChainRAG] = None):
        """
        Initialize the research toolkit.
        
        Args:
            rag_chain: Optional RAG chain instance
        """
        self.rag_chain = rag_chain
        self.tools = self._create_tools()
    
    def _create_tools(self) -> List[Tool]:
        """
        Create the tools for the agent.
        
        Returns:
            List of tools
        """
        tools = [
            create_tavily_search_tool(),
            Tool(
                name="ArxivSearch",
                description="Search for scientific papers on arXiv. Use this tool when you need academic or scientific information.",
                func=arxiv_search,
                args_schema=ArxivQueryInput
            )
        ]
        
        # Add RAG tool if available
        if self.rag_chain:
            @tool
            def document_rag_search(query: str) -> str:
                """
                Search the uploaded document using RAG.
                
                Args:
                    query: The search query string
                    
                Returns:
                    The response from the RAG model
                """
                docs = self.rag_chain.retriever.invoke(query)
                context = "\n\n".join([doc.page_content for doc in docs])
                response = self.rag_chain.chain.invoke(query)
                
                return f"Based on the uploaded document: {response}"
            
            tools.append(
                Tool(
                    name="DocumentSearch",
                    description="Search within the user's uploaded document. Use this tool when you need information from the specific document that was uploaded.",
                    func=document_rag_search,
                    args_schema=RAGQueryInput
                )
            )
        
        return tools
    
    def get_tools(self) -> List[Tool]:
        """
        Get the list of tools.
        
        Returns:
            List of tools
        """
        return self.tools
    
    def set_rag_chain(self, rag_chain: LangChainRAG):
        """
        Update the RAG chain and rebuild tools.
        
        Args:
            rag_chain: New RAG chain instance
        """
        self.rag_chain = rag_chain
        self.tools = self._create_tools()