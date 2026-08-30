#!/usr/bin/env bash

# ==============================================================================
# arXiv Research Assistant & 2×2 Evaluation Benchmark Runner
# ==============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_banner() {
    echo -e "${CYAN}================================================================${NC}"
    echo -e "${CYAN}     🤖 arXiv Research Assistant & RAG Evaluation Platform      ${NC}"
    echo -e "${CYAN}================================================================${NC}"
}

activate_venv() {
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    else
        echo -e "${RED}Error: Virtual environment .venv not found. Please set up Python venv first.${NC}"
        exit 1
    fi
}

start_docker() {
    echo -e "${BLUE}[1/2] Checking Docker databases (Qdrant & Elasticsearch)...${NC}"
    if ! docker info > /dev/null 2>&1; then
        echo -e "${YELLOW}Warning: Docker is not running. Please open Docker Desktop.${NC}"
    else
        docker compose up -d qdrant elasticsearch
        echo -e "${GREEN}✓ Qdrant (:6333) and Elasticsearch (:9200) are running.${NC}"
    fi
}

index_elasticsearch() {
    activate_venv
    start_docker
    echo -e "${BLUE}[2/2] Indexing processed arXiv documents into Elasticsearch...${NC}"
    python script/elasticsearch_index.py setup
    python script/elasticsearch_index.py index
    echo -e "${GREEN}✓ Elasticsearch indexing complete!${NC}"
}

run_eval() {
    activate_venv
    start_docker
    echo -e "${GREEN}▶ Running 2×2 RAG Evaluation Benchmark (2 Retrievers × 2 LLMs)...${NC}"
    python -m src.evaluation.runner "$@"
}

run_tests() {
    activate_venv
    echo -e "${BLUE}▶ Running full test suite with pytest...${NC}"
    pytest test/test_retrievers.py test/test_llm_clients.py test/test_evaluation.py test/test_workflow.py test/test_api.py -v
}

run_app() {
    activate_venv
    start_docker

    echo -e "\n${GREEN}🚀 Starting Full Application Stack...${NC}"
    echo -e "${CYAN} • Backend API:${NC}  http://localhost:8000 (Docs: http://localhost:8000/docs)"
    echo -e "${CYAN} • Chat UI:${NC}      http://localhost:3000"
    echo -e "${CYAN} • Evaluation:${NC}   http://localhost:3000/evaluation"
    echo -e "${YELLOW}Press Ctrl+C to stop all servers.${NC}\n"

    # Trap Ctrl+C to kill background processes
    trap 'echo -e "\n${YELLOW}Stopping servers...${NC}"; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0' SIGINT SIGTERM

    # Start FastAPI backend
    python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload &
    BACKEND_PID=$!

    # Start Next.js frontend
    (cd frontend && npm run dev) &
    FRONTEND_PID=$!

    # Wait for both processes
    wait $BACKEND_PID $FRONTEND_PID
}

show_help() {
    print_banner
    echo -e "\nUsage: ${GREEN}./run.sh${NC} ${YELLOW}[command]${NC} [options]\n"
    echo -e "Available Commands:"
    echo -e "  ${GREEN}app${NC} | ${GREEN}dev${NC}         Start full stack (FastAPI Backend :8000 + Next.js Frontend :3000)"
    echo -e "  ${GREEN}eval${NC}               Run automated 2×2 Evaluation Benchmark"
    echo -e "                       ${CYAN}Examples:${NC}"
    echo -e "                         ./run.sh eval"
    echo -e "                         ./run.sh eval --max-questions 5"
    echo -e "                         ./run.sh eval --retrieval hybrid --llm model_1"
    echo -e "  ${GREEN}docker${NC}             Start Qdrant & Elasticsearch Docker containers"
    echo -e "  ${GREEN}index${NC}              Index arXiv papers into Elasticsearch for Hybrid retrieval"
    echo -e "  ${GREEN}test${NC}               Run automated test suite"
    echo -e "  ${GREEN}help${NC}               Show this help message"
    echo ""
}

# Main command router
COMMAND="${1:-help}"
shift 1 2>/dev/null || true

case "$COMMAND" in
    app|dev|start)
        run_app
        ;;
    eval|evaluation|benchmark)
        run_eval "$@"
        ;;
    docker|db)
        start_docker
        ;;
    index|es-index)
        index_elasticsearch
        ;;
    test|tests|pytest)
        run_tests
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        show_help
        exit 1
        ;;
esac
