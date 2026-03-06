#!/bin/bash

echo "🚀 Multimodal RAG Engine - Quick Start"
echo "======================================"

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  Creating .env file from template..."
    cp .env.example .env
    echo "📝 Please edit .env and add your OPENAI_API_KEY"
    exit 1
fi

# Check if OPENAI_API_KEY is set
if ! grep -q "OPENAI_API_KEY=sk-" .env; then
    echo "⚠️  Please set your OPENAI_API_KEY in .env file"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create uploads directory
mkdir -p uploads

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the API server:"
echo "  make run-api"
echo ""
echo "To start the Streamlit demo:"
echo "  make run-demo"
echo ""
echo "To run tests:"
echo "  make test"
