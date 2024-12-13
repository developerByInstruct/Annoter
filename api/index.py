import streamlit as st
from streamlit.web.server.server import Server
import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your app
from app import main

# Create the Vercel handler
def handler(event, context):
    if Server.get_current():
        return {
            "statusCode": 200,
            "body": "Streamlit app is already running"
        }
    
    # Run the Streamlit app
    main()
    
    return {
        "statusCode": 200,
        "body": "Streamlit app started successfully"
    }

# For local development
if __name__ == "__main__":
    main()
