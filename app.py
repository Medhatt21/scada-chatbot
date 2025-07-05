#!/usr/bin/env python3
"""Main entry point for the SCADA RAG Chatbot."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.streamlit_app import main

if __name__ == "__main__":
    # Run the Streamlit app
    try:
        main()
    except KeyboardInterrupt:
        print("Application stopped by user")
    except Exception as e:
        print(f"Application failed: {e}")
        sys.exit(1) 