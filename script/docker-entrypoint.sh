#!/bin/sh
# arXiv Research Assistant - Entrypoint Script
# Auto-index on first run, then start the application

set -e

# Check if data exists and has not been indexed yet
if [ ! -f /app/.indexed ]; then
    echo "🚀 First run detected - checking if indexing is needed..."
    
    # Wait for Qdrant to be ready
    echo "⏳ Waiting for Qdrant..."
    until python -c "from qdrant_client import QdrantClient; QdrantClient('${QDRANT_URL:-http://qdrant:6333}').get_collections()" 2>/dev/null; do
        echo "   Qdrant not ready yet, retrying in 2s..."
        sleep 2
    done
    echo "✅ Qdrant is ready!"
    
    # Wait for Elasticsearch to be ready
    echo "⏳ Waiting for Elasticsearch..."
    sleep 20
    echo "✅ Elasticsearch should be ready!"
    
    # Download papers from arXiv if not already downloaded
    if [ ! -d /app/data/raw ] || [ -z "$(ls -A /app/data/raw/*.pdf 2>/dev/null)" ]; then
        echo "📥 Downloading papers from arXiv..."
        python script/download_paper.py || echo "⚠️ Download had issues"
    else
        echo "📦 Papers already downloaded, skipping..."
    fi
    
    # Process PDFs if raw PDFs exist but no processed data
    if [ -d /app/data/raw ] && [ -n "$(ls -A /app/data/raw/*.pdf 2>/dev/null)" ]; then
        if [ ! -f /app/data/processed/arxiv_documents.json ] || [ ! -s /app/data/processed/arxiv_documents.json ]; then
            echo "📄 Processing PDFs..."
            python script/process_pdf.py || echo "⚠️ PDF processing had issues"
        fi
    fi
    
    # Initialize Qdrant collection (auto recreate on first run)
    echo "🔧 Initializing Qdrant collection..."
    python script/qdrant.py --recreate || echo "⚠️ Qdrant init skipped"

    # Setup Elasticsearch index
    echo "🔧 Setting up Elasticsearch index..."
    python script/elasticsearch_index.py || echo "⚠️ ES setup skipped"

    # Index documents if processed data exists
    if [ -f /app/data/processed/arxiv_documents.json ] && [ -s /app/data/processed/arxiv_documents.json ]; then
        echo "📊 Indexing documents to Qdrant and Elasticsearch..."
        python script/qdrant.py recreate || echo "⚠️ Qdrant indexing had issues"
        python script/elasticsearch_index.py recreate || echo "⚠️ ES indexing had issues"
    else
        echo "⚠️ No processed data to index"
    fi
    
    # Mark as indexed
    touch /app/.indexed
    echo "✅ Indexing complete! Starting application..."
else
    if [ -f /app/.indexed ]; then
        echo "📦 Data already indexed, skipping..."
    else
        echo "⚠️ No processed data found at /app/data/processed/arxiv_documents.json"
        echo "   Run data preparation scripts first, or mount data volume."
    fi
    echo "🚀 Starting application..."
fi

# Execute the main command
exec "$@"
