"""
Chainlit event handlers for the Research Agent.
"""
import os
import chainlit as cl
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from utils.file_processor import process_file
from models.rag import LangChainRAG
import config

@cl.on_chat_start
async def on_chat_start():
    """
    Handler for chat start event. Prompts user to upload a file
    and initializes the RAG system.
    """
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a Text or PDF file to begin!",
            accept=["text/plain", "application/pdf"],
            max_size_mb=2,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Load and process the file
    texts = process_file(file)
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
    
    # Initialize language model
    llm = ChatOpenAI(streaming=True)

    # Create RAG chain
    rag_chain = LangChainRAG(retriever=retriever, llm=llm)
    
    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", rag_chain)

@cl.on_message
async def main(message):
    """
    Handler for user messages. Processes the query through the RAG chain
    and streams the response back to the user.
    
    Args:
        message: The user's message
    """
    chain = cl.user_session.get("chain")

    msg = cl.Message(content="")
    result = await chain.arun_pipeline(message.content)

    async for stream_resp in result["response"]:
        await msg.stream_token(stream_resp)

    await msg.send()