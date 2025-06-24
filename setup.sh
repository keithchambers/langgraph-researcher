#!/bin/bash

# LangGraph Research Orchestrator Setup Script
# Automates environment setup for local development

set -e  # Exit on any error

echo "Setting up LangGraph Research Orchestrator environment..."

# Check if running on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS system"
    
    # Install Ollama using Homebrew
    if ! command -v ollama &> /dev/null; then
        echo "Installing Ollama..."
        if ! command -v brew &> /dev/null; then
            echo "ERROR: Homebrew not found. Please install Homebrew first: https://brew.sh/"
            exit 1
        fi
        brew install ollama
    else
        echo "Ollama already installed"
    fi
else
    echo "Non-macOS system detected. Please install Ollama manually from https://ollama.ai/"
    echo "   For Linux: curl -fsSL https://ollama.ai/install.sh | sh"
    
    # Verify Ollama installation
    if ! command -v ollama &> /dev/null; then
        echo "ERROR: Please install Ollama first, then re-run this script"
        exit 1
    fi
fi

# Start Ollama service
echo "Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!

# Wait for service to start
sleep 3

# Pull the model
echo "Pulling qwen2.5:0.5b model (this may take a few minutes)..."
ollama pull qwen2.5:0.5b

# Setup Python virtual environment
echo "Setting up Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment. Please ensure Python 3 is installed."
        exit 1
    fi
fi

# Activate virtual environment
source .venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install Python dependencies"
    exit 1
fi

echo ""
echo "Setup complete! The environment is ready."
echo ""
echo "OPTIONAL: To use Brave Search API for web search, set your API key:"
echo "   export BRAVE_API_KEY=\"your_brave_api_key_here\""
echo "   (Get your free API key from: https://api.search.brave.com/)"
echo ""
echo "Note: If BRAVE_API_KEY is not set, the app will fall back to LLM knowledge."
echo ""
echo "To run the application:"
echo "   source .venv/bin/activate"
echo "   python supervisor_ollama.py \"Your question here\""
echo ""
echo "Example:"
echo "   python supervisor_ollama.py \"What are the benefits of renewable energy?\""
echo ""
echo "For performance testing:"
echo "   python performance_test.py -q 5 -i 3"
echo ""
echo "Ollama service is running in background (PID: $OLLAMA_PID)"
echo "   To stop: kill $OLLAMA_PID" 