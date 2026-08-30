#!/usr/bin/env python3
"""
arXiv Research Assistant - Cloud Vector Database Seeder
======================================================
Utility script to initialize and seed Qdrant Cloud (or any remote Qdrant instance)
with processed arXiv paper documents and dense embeddings.

Usage:
    python script/seed_cloud.py
    python script/seed_cloud.py --url "https://your-cluster.qdrant.io:6333" --api-key "your_key"
    python script/seed_cloud.py --recreate --batch-size 32
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from tqdm import tqdm

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

load_dotenv()

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_VECTOR_SIZE = 384
DENSE_VECTOR_NAME = "dense"
DATA_FILE = project_root / "data" / "processed" / "arxiv_documents.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Seed Qdrant Cloud Vector Database for arXiv Assistant"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant Cloud URL or endpoint",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("QDRANT_API_KEY", None),
        help="Qdrant Cloud API Key",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=os.getenv("QDRANT_COLLECTION", "arxiv_papers"),
        help="Collection name (default: arxiv_papers)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for vector upsert (default: 32)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the collection if it already exists",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only test connection without uploading documents",
    )
    return parser.parse_args()


def load_documents() -> list[dict]:
    if not DATA_FILE.exists():
        print(f"❌ Error: Data file not found at {DATA_FILE}")
        print("   Please ensure data/processed/arxiv_documents.jsonl exists.")
        sys.exit(1)

    documents = []
    with open(DATA_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    documents.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return documents


def test_connection(client: QdrantClient, url: str) -> bool:
    print(f"🔍 Testing connection to Qdrant at: {url} ...")
    try:
        collections = client.get_collections().collections
        names = [c.name for c in collections]
        print("✅ Successfully connected to Qdrant!")
        print(f"   Existing collections: {names if names else '(None)'}")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant at {url}")
        print(f"   Error: {e}")
        return False


def setup_collection(client: QdrantClient, collection_name: str, recreate: bool = False):
    exists = client.collection_exists(collection_name)
    if exists:
        if recreate:
            print(f"🗑️ Recreating collection '{collection_name}'...")
            client.delete_collection(collection_name)
        else:
            print(f"ℹ️ Collection '{collection_name}' already exists. Using existing collection.")
            return

    print(f"🔧 Creating collection '{collection_name}' with dense cosine embeddings ({DENSE_VECTOR_SIZE}d)...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=DENSE_VECTOR_SIZE, distance=models.Distance.COSINE
            )
        },
    )

    # Create payload indexes for fast filtering
    indexes = ["paper_id", "section", "page_info", "year"]
    for idx in indexes:
        schema = models.PayloadSchemaType.KEYWORD if idx in ["paper_id", "section"] else models.PayloadSchemaType.INTEGER
        client.create_payload_index(
            collection_name=collection_name,
            field_name=idx,
            field_schema=schema,
        )
    print("✅ Collection and payload indexes created successfully!")


def index_documents(client: QdrantClient, collection_name: str, documents: list[dict], batch_size: int = 32):
    total = len(documents)
    print(f"🚀 Starting ingestion of {total} documents (batch_size={batch_size})...")
    start_time = time.time()
    
    for i in tqdm(range(0, total, batch_size), desc="Ingesting vectors"):
        batch = documents[i : i + batch_size]
        points = []
        for doc in batch:
            content = doc.get("content", "")
            point = models.PointStruct(
                id=doc["id"],
                vector={
                    DENSE_VECTOR_NAME: models.Document(text=content, model=DENSE_MODEL)
                },
                payload={
                    "content": content,
                    "paper_id": doc.get("paper_id", ""),
                    "title": doc.get("title", ""),
                    "year": doc.get("year", 2024),
                    "section": doc.get("section", ""),
                    "content_type": doc.get("content_type", "section"),
                    "page_info": doc.get("page_info", 1),
                },
            )
            points.append(point)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                client.upsert(collection_name=collection_name, points=points, wait=True)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ Batch upload error ({e}), retrying ({attempt+1}/{max_retries})...")
                    time.sleep(2)
                else:
                    print(f"❌ Failed batch starting at index {i}: {e}")

    elapsed = round(time.time() - start_time, 2)
    info = client.get_collection(collection_name)
    print("\n🎉 Ingestion Complete!")
    print(f"   • Total Points in Collection: {info.points_count}")
    print(f"   • Time Taken: {elapsed} seconds")


def main():
    args = parse_args()
    print("==================================================")
    print("      arXiv Assistant - Qdrant Cloud Seeder       ")
    print("==================================================")

    client = QdrantClient(
        url=args.url,
        api_key=args.api_key if args.api_key else None,
        check_compatibility=False,
    )

    if not test_connection(client, args.url):
        sys.exit(1)

    if args.test_only:
        print("Test connection successful. Exiting (--test-only).")
        return

    documents = load_documents()
    print(f"📄 Loaded {len(documents)} document chunks from {DATA_FILE.name}")

    setup_collection(client, args.collection, recreate=args.recreate)
    index_documents(client, args.collection, documents, batch_size=args.batch_size)
    print("\n✅ Ready for production search queries!")


if __name__ == "__main__":
    main()
