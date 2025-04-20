"""
Chainlit event handlers for the Research Agent.
"""
import os
import chainlit as cl
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor
# Update memory import to use the newer approach
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder

from utils.file_processor import process_file
from models.rag import LangChainRAG
from models.research_tools import ResearchToolkit, RAGQueryInput
import config
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from typing import TypedDict, Annotated, Dict, Any, Literal, Union, cast, List, Optional
from langgraph.graph.message import add_messages
import operator
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.documents import Document
from langchain_core.tools import Tool

tavily_tool = TavilySearchResults(max_results=5)
duckduckgo_tool = DuckDuckGoSearchResults(max_results=5)
arxiv_tool = ArxivQueryRun()

tool_belt = [
    tavily_tool,
    duckduckgo_tool,
    arxiv_tool,
]

model = ChatOpenAI(model="gpt-4o", temperature=0)
model = model.bind_tools(tool_belt)

class ResearchAgentState(TypedDict):
    """
    State definition for the Research Agent using LangGraph.
    
    Attributes:
        messages: List of messages in the conversation
        context: Additional context information from RAG retrievals
        documents: Optional list of Document objects from uploaded files
    """
    messages: Annotated[list[BaseMessage], add_messages]
    context: str
    documents: Optional[List[Document]]


from langgraph.prebuilt import ToolNode


def call_model(state: Dict[str, Any]) -> Dict[str, list[BaseMessage]]:
    """
    Process the current state through the language model.
    
    Args:
        state: Current state containing messages and context
        
    Returns:
        Updated state with model's response added to messages
    """
    try:
        messages = state["messages"]
        context = state.get("context", "")
        
        # Add context from documents if available
        if context:
            # Insert system message with context before the latest user message
            context_message = SystemMessage(content=f"Use the following information from uploaded documents to enhance your response if relevant:\n\n{context}")
            
            # Find the position of the last user message
            for i in range(len(messages)-1, -1, -1):
                if isinstance(messages[i], HumanMessage):
                    # Insert context right after the last user message
                    enhanced_messages = messages[:i+1] + [context_message] + messages[i+1:]
                    break
            else:
                # No user message found, just append context
                enhanced_messages = messages + [context_message]
        else:
            enhanced_messages = messages
        
        # Get response from the model
        response = model.invoke(enhanced_messages)
        return {"messages": [response]}
    except Exception as e:
        # Handle exceptions gracefully
        error_msg = f"Error calling model: {str(e)}"
        print(error_msg)  # Log the error
        # Return a fallback response
        return {"messages": [HumanMessage(content=error_msg)]}


def should_continue(state: Dict[str, Any]) -> Union[Literal["action"], Literal[END]]:
    """
    Determine if the agent should continue processing or end.
    
    Args:
        state: Current state containing messages and context
        
    Returns:
        "action" if tool calls are present, otherwise END
    """
    last_message = state["messages"][-1]
    
    if last_message.tool_calls:
        return "action"
    
    return END


def convert_inputs(input_object: Dict[str, str]) -> Dict[str, list[BaseMessage]]:
    """
    Convert user input into the format expected by the agent.
    
    Args:
        input_object: Dictionary containing the user's question
        
    Returns:
        Formatted input state for the agent
    """
    return {"messages": [HumanMessage(content=input_object["question"])]}


def parse_output(input_state: Dict[str, Any]) -> str:
    """
    Extract the final response from the agent's state.
    
    Args:
        input_state: The final state of the agent
        
    Returns:
        The content of the last message
    """
    try:
        return cast(str, input_state["messages"][-1].content)
    except (IndexError, KeyError, AttributeError) as e:
        # Handle potential errors when accessing the output
        error_msg = f"Error parsing output: {str(e)}"
        print(error_msg)  # Log the error
        return "I encountered an error while processing your request."


def build_agent_chain() -> Any:
    """
    Constructs and returns the research agent execution chain.
    
    The chain consists of:
    1. A retrieval node that gets context from documents
    2. An agent node that processes messages
    3. A tool node that executes tools when called
    
    Returns:
        Compiled agent chain ready for execution
    """
    # Create document search tool
    doc_search_tool = Tool(
        name="DocumentSearch",
        description="Search within the user's uploaded document. Use this tool when you need information from the specific document that was uploaded.",
        func=document_search_tool,
        args_schema=RAGQueryInput
    )
    
    # Add document search tool to the tool belt if we have upload capability
    tools = tool_belt.copy()
    tools.append(doc_search_tool)
    
    # Create a node for tool execution
    tool_node = ToolNode(tools)

    # Initialize the graph with our state type
    uncompiled_graph = StateGraph(ResearchAgentState)

    # Add nodes
    uncompiled_graph.add_node("retrieve", retrieve_from_documents)
    uncompiled_graph.add_node("agent", call_model)
    uncompiled_graph.add_node("action", tool_node)
    
    # Set the entry point to retrieve context first
    uncompiled_graph.set_entry_point("retrieve")
    
    # Add edges
    uncompiled_graph.add_edge("retrieve", "agent")
    
    # Add conditional edges from agent
    uncompiled_graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "action": "action",
            END: END
        }
    )

    # Complete the loop
    uncompiled_graph.add_edge("action", "agent")
    
    # Compile the graph
    compiled_graph = uncompiled_graph.compile()

    # Create the full chain
    agent_chain = convert_inputs | compiled_graph 
    return agent_chain


def retrieve_from_documents(state: Dict[str, Any]) -> Dict[str, str]:
    """
    Retrieve relevant context from uploaded documents based on the user query.
    
    Args:
        state: Current state containing messages and optional documents
        
    Returns:
        Updated state with context from document retrieval
    """
    # Get the last user message
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            query = message.content
            break
    else:
        # No user message found
        return {"context": ""}
    
    # Skip if no documents are uploaded
    retriever = cl.user_session.get("retriever")
    if not retriever:
        return {"context": ""}
    
    try:
        # Retrieve relevant documents
        docs = retriever.invoke(query)
        if not docs:
            return {"context": ""}
        
        # Extract text from documents
        context = "\n\n".join([f"Document excerpt: {doc.page_content}" for doc in docs])
        return {"context": context}
    except Exception as e:
        print(f"Error retrieving from documents: {str(e)}")
        return {"context": ""}


def document_search_tool(query: str) -> str:
    """
    Tool function to search within uploaded documents.
    
    Args:
        query: Search query string
        
    Returns:
        Information retrieved from the documents
    """
    retriever = cl.user_session.get("retriever")
    if not retriever:
        return "No documents have been uploaded yet. Please upload a document first."
    
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in the uploaded documents."
    
    # Format the results
    results = []
    for i, doc in enumerate(docs):
        results.append(f"[Document {i+1}] {doc.page_content}")
    
    return "\n\n".join(results)


@cl.on_chat_start
async def on_chat_start():
    """
    Handler for chat start event. Prompts user to upload a file
    and initializes the research agent with various tools.
    """
    # Display welcome message
    await cl.Message(
        content="Welcome to the Research Agent! I can help you research topics using web search, arXiv papers, and documents you upload."
    ).send()
    
    # Create the agent
    agent = build_agent_chain()
    
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
            {"question" : message.content},
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
    final_answer = parse_output(result) #result["messages"][-1].content
    
    # Fix: Replace cl.make_async_gen with proper token streaming in Chainlit 2.0.4
    # Instead of using make_async_gen, we'll manually stream tokens from the final_answer
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
        
        # Rebuild the agent chain with updated tools
        agent = build_agent_chain()
        cl.user_session.set("agent", agent)
        
        # Let the user know that the file is processed
        await cl.Message(
            content=f"Successfully processed `{file.name}`. You can now ask questions about this document along with web search and academic research."
        ).send()
        
    except Exception as e:
        await cl.Message(
            content=f"Error processing file: {str(e)}"
        ).send()