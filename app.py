"""
Main entry point for the Research Agent application.

This file imports the necessary components from other modules and
serves as the entry point for the Chainlit application.
"""

# Import the handlers to register Chainlit event handlers
from handlers.chainlit_handlers import on_chat_start, main

# The Chainlit application will automatically
# discover and use the imported event handlers

if __name__ == "__main__":
    print("Research Agent started. Access the web interface to interact.")