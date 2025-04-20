"""
Chainlit event handlers for the Research Agent.
"""
import os
import chainlit as cl
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.tools import Tool
from typing import Dict, Any, List, Optional
from langchain_core.documents import Document

from utils.file_processor import process_file
from models.rag import LangChainRAG
from models.research_tools import RAGQueryInput, create_rss_feed_tool
from models.search_tools import create_search_tools
from models.agent import build_agent_chain, parse_output
import config

@cl.on_chat_start
async def on_chat_start():
    """
    Handler for chat start event. Prompts user to upload a file
    and initializes the research agent with various tools.
    """
    # Display welcome message
    await cl.Message(
        content="Welcome to the Research Agent! I can help you research topics using web search, arXiv papers, RSS feeds, and documents you upload."
    ).send()
    
    # Create search tools
    tools = create_search_tools(max_results=config.MAX_TAVILY_SEARCH_RESULTS)
    
    # Add RSS Feed tool
    rss_feed_tool = create_rss_feed_tool()
    tools.append(rss_feed_tool)
    
    # Create the agent
    agent = build_agent_chain(tools)
    
    # Store agent in user session
    cl.user_session.set("agent", agent)
    
    # Initialize retriever as None (will be set when a file is uploaded)
    cl.user_session.set("retriever", None)
    
    # Prompt user to upload a file (optional)
    await cl.Message(
        content="You can start researching right away, or upload a document (PDF or text) to include in your research sources."
    ).send()

@cl.on_message
async def main(message):
    """
    Handler for user messages. Processes the query through the research agent
    and streams the response back to the user.
    
    Args:
        message: The user's message
    """
    agent_executor = cl.user_session.get("agent")
    
    # Create Chainlit message for streaming
    msg = cl.Message(content="")
    
    # Process the message as a file upload if there are attachments
    if message.elements:
        for element in message.elements:
            if isinstance(element, cl.File):
                await process_uploaded_file(element, msg)
                return
    
    # Create a parent step for the research process
    with cl.Step(name="Research Process", type="tool") as step:
        # Run the agent executor with callbacks to stream the response
        result = await agent_executor.ainvoke(
            {"question": message.content},
            config={
                "callbacks": [cl.AsyncLangchainCallbackHandler()],
                "configurable": {"session_id": message.id}  # Add session_id from message
            }
        )
        
        # Add steps from agent's intermediate steps
        for i, step_data in enumerate(result.get("intermediate_steps", [])):
            step_name = f"Using: {step_data[0].tool}"
            step_input = str(step_data[0].tool_input)
            step_output = str(step_data[1])
            
            # Create individual steps as children of the main step
            with cl.Step(name=step_name, type="tool") as substep:
                await cl.Message(
                    content=f"**Input:** {step_input}\n\n**Output:** {step_output}",
                ).send()
    
    # Get the final answer
    final_answer = parse_output(result)
    
    # Stream tokens from the final_answer
    await msg.stream_token(final_answer)
    await msg.send()

async def process_uploaded_file(file: cl.File, msg: cl.Message):
    """
    Process an uploaded file and update the agent with RAG capabilities.
    
    Args:
        file: The uploaded file
        msg: The message to update
    """
    await msg.stream_token(f"Processing `{file.name}`...")
    await msg.send()
    
    try:
        # Load and process the file
        texts = process_file(file)
        
        if not texts:
            await cl.Message(content=f"Could not extract text from `{file.name}`. Please try another file.").send()
            return
            
        print(f"Processing {len(texts)} text chunks")
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings()
        
        # Create a unique collection name based on the file name
        collection_name = f"collection_{file.name.replace('.', '_')}_{os.urandom(4).hex()}"
        
        # Initialize Qdrant client (using in-memory storage)
        client = QdrantClient(":memory:")
        
        # Create collection with proper vector dimensions for OpenAI embeddings
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=config.VECTOR_DIMENSION, distance=Distance.COSINE)
        )
        
        # Create vector store with QdrantVectorStore
        vector_store = QdrantVectorStore(
            client=client, 
            collection_name=collection_name,
            embedding=embeddings
        )
        
        # Add documents to the vector store
        vector_store.add_documents(texts)
        
        # Create a retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": config.NUM_RETRIEVAL_RESULTS})
        
        # Store the retriever in the user session
        cl.user_session.set("retriever", retriever)
        
        # Get the search tools
        tools = create_search_tools(max_results=config.MAX_TAVILY_SEARCH_RESULTS)
        
        # Add RSS Feed tool
        rss_feed_tool = create_rss_feed_tool()
        tools.append(rss_feed_tool)
        
        # Rebuild the agent with the retriever
        agent = build_agent_chain(tools, retriever)
        cl.user_session.set("agent", agent)
        
        # Let the user know that the file is processed
        await cl.Message(
            content=f"Successfully processed `{file.name}`. You can now ask questions about this document along with web search and academic research."
        ).send()
        
    except Exception as e:
        await cl.Message(
            content=f"Error processing file: {str(e)}"
        ).send()