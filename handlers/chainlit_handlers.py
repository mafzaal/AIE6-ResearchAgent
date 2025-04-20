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
from models.research_tools import ResearchToolkit
import config
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from typing import TypedDict, Annotated, Dict, Any, Literal, Union, cast
from langgraph.graph.message import add_messages
import operator
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchResults

tavily_tool = TavilySearchResults(max_results=5)
duckduckgo_tool = DuckDuckGoSearchResults(max_results=5)

tool_belt = [
    tavily_tool,
    duckduckgo_tool,
    ArxivQueryRun(),
]

model = ChatOpenAI(model="gpt-4o", temperature=0)
model = model.bind_tools(tool_belt)

class ResearchAgentState(TypedDict):
    """
    State definition for the Research Agent using LangGraph.
    
    Attributes:
        messages: List of messages in the conversation
        context: Additional context information
    """
    messages: Annotated[list[BaseMessage], add_messages]
    context: str


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
        response = model.invoke(messages)
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
    1. An agent node that processes messages
    2. A tool node that executes tools when called
    
    Returns:
        Compiled agent chain ready for execution
    """
    # Create a node for tool execution
    tool_node = ToolNode(tool_belt)

    # Initialize the graph with our state type
    uncompiled_graph = StateGraph(ResearchAgentState)

    uncompiled_graph.add_node("agent", call_model)
    uncompiled_graph.add_node("action", tool_node)
    uncompiled_graph.set_entry_point("agent")
    uncompiled_graph.add_conditional_edges(
        "agent",
        should_continue
    )

    uncompiled_graph.add_edge("action", "agent")
    compiled_graph = uncompiled_graph.compile()

    agent_chain = convert_inputs | compiled_graph 
    return agent_chain



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
    
    # Initialize language model for the agent
    
    
    # Create the agent
    agent = build_agent_chain()
    
    
    # Store agent and toolkit in user session
    cl.user_session.set("agent", agent)
    
    
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
        
        # Initialize language model
        llm = ChatOpenAI(streaming=True)
        
        # Create RAG chain
        rag_chain = LangChainRAG(retriever=retriever, llm=llm)
        
        # Get toolkit and update it with the RAG chain
        toolkit = cl.user_session.get("toolkit")
        toolkit.set_rag_chain(rag_chain)
        
        # Re-create the agent with updated tools
        agent_executor = cl.user_session.get("agent")
        
        # Update the agent's tools
        agent_executor.tools = toolkit.get_tools()
        
        # Update the session
        cl.user_session.set("agent", agent_executor)
        
        # Let the user know that the file is processed
        await cl.Message(
            content=f"Successfully processed `{file.name}`. You can now ask questions about this document along with web search and academic research."
        ).send()
        
    except Exception as e:
        await cl.Message(
            content=f"Error processing file: {str(e)}"
        ).send()