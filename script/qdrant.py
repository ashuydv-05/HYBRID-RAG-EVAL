import argparse
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from tqdm import tqdm

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "arxiv_papers")
DATA_FILE = project_root / "data" / "processed" / "arxiv_documents.jsonl"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_VECTOR_SIZE = 384
DENSE_VECTOR_NAME = "dense"


def load_documents() -> list[dict]:
    if not DATA_FILE.exists():
        print(f"Data file not found: {DATA_FILE}")
        print("Run data processing first.")
        sys.exit(1)
    documents = []
    with open(DATA_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    return documents


def setup_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        return
    print(f"Creating collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=DENSE_VECTOR_SIZE, distance=models.Distance.COSINE
            )
        },
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="paper_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="section",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="page_info",
        field_schema=models.PayloadSchemaType.INTEGER,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="year",
        field_schema=models.PayloadSchemaType.INTEGER,
    )
    print(f"Collection '{COLLECTION_NAME}' created with dense config:")
    print(f"  Dense: {DENSE_MODEL} ({DENSE_VECTOR_SIZE}d, cosine)")
    print(f"  Indexes: paper_id, section, page_info, year")


def index_documents(
    client: QdrantClient, documents: list[dict], batch_size: int = 32
) -> None:
    total = len(documents)
    print(f"Indexing {total} documents (batch_size={batch_size})...")
    for i in tqdm(range(0, total, batch_size), desc="Indexing"):
        batch = documents[i : i + batch_size]
        points = []
        for doc in batch:
            content = doc["content"]
            point = models.PointStruct(
                id=doc["id"],
                vector={
                    DENSE_VECTOR_NAME: models.Document(text=content, model=DENSE_MODEL)
                },
                payload={
                    "content": content,
                    "paper_id": doc["paper_id"],
                    "title": doc["title"],
                    "year": doc["year"],
                    "section": doc["section"],
                    "content_type": doc["content_type"],
                    "page_info": doc["page_info"],
                },
            )
            points.append(point)
        client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
    print(f"Indexed {total} documents successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qdrant operations for arXiv papers")
    parser.add_argument(
        "command", choices=["setup", "index", "recreate"], help="Command to run"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for indexing"
    )
    args = parser.parse_args()
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    if args.command == "setup":
        setup_collection(client)
    elif args.command == "index":
        if not client.collection_exists(COLLECTION_NAME):
            print(f"Collection '{COLLECTION_NAME}' not found. Run 'setup' first.")
            sys.exit(1)
        documents = load_documents()
        index_documents(client, documents, batch_size=args.batch_size)
    elif args.command == "recreate":
        if client.collection_exists(COLLECTION_NAME):
            print(f"Deleting collection '{COLLECTION_NAME}'...")
            client.delete_collection(COLLECTION_NAME)
        setup_collection(client)
        documents = load_documents()
        index_documents(client, documents, batch_size=args.batch_size)
    print("\nDone!")


if __name__ == "__main__":
    main()
