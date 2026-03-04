import argparse
import json
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm
from src.config.settings import settings

DATA_FILE = project_root / "data" / "processed" / "arxiv_documents.jsonl"
INDEX_SETTINGS = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            "paper_id": {"type": "keyword"},
            "section": {"type": "keyword"},
            "content_type": {"type": "keyword"},
            "page_info": {"type": "integer"},
            "year": {"type": "integer"},
            "content": {"type": "text"},
            "title": {"type": "text"},
        }
    },
}


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


def setup_index(es: Elasticsearch, index_name: str) -> None:
    if es.indices.exists(index=index_name):
        print(f"Index '{index_name}' already exists.")
        return
    print(f"Creating index '{index_name}'...")
    es.indices.create(index=index_name, body=INDEX_SETTINGS)
    mapping = es.indices.get_mapping(index=index_name)
    props = mapping[index_name]["mappings"]["properties"]
    print(
        f"Index '{index_name}' created with fields: {', '.join(sorted(props.keys()))}"
    )


def index_documents(es: Elasticsearch, documents: list[dict], index_name: str) -> None:
    actions = []
    for doc in documents:
        actions.append(
            {
                "_index": index_name,
                "_id": doc["id"],
                "_source": {
                    "content": doc["content"],
                    "paper_id": doc["paper_id"],
                    "title": doc["title"],
                    "year": doc["year"],
                    "section": doc["section"],
                    "content_type": doc["content_type"],
                    "page_info": doc["page_info"],
                },
            }
        )
    success, errors = bulk(
        es,
        tqdm(actions, desc="Indexing"),
        chunk_size=100,
        request_timeout=120,
        raise_on_error=False,
    )
    print(
        f"Indexed: {success}, Errors: {(len(errors) if isinstance(errors, list) else errors)}"
    )
    es.indices.refresh(index=index_name)
    count = es.count(index=index_name)["count"]
    print(f"Total documents in index: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Elasticsearch operations for arXiv papers"
    )
    parser.add_argument(
        "command", choices=["setup", "index", "recreate"], help="Command to run"
    )
    args = parser.parse_args()
    es = Elasticsearch(settings.elasticsearch.url, request_timeout=60)
    index_name = settings.elasticsearch.index
    info = es.info()
    print(f"Connected to Elasticsearch {info['version']['number']}")
    if args.command == "setup":
        setup_index(es, index_name)
    elif args.command == "index":
        if not es.indices.exists(index=index_name):
            print(f"Index '{index_name}' not found. Run 'setup' first.")
            sys.exit(1)
        documents = load_documents()
        print(f"Loaded {len(documents)} documents from {DATA_FILE}")
        index_documents(es, documents, index_name)
    elif args.command == "recreate":
        if es.indices.exists(index=index_name):
            print(f"Deleting index '{index_name}'...")
            es.indices.delete(index=index_name)
        setup_index(es, index_name)
        documents = load_documents()
        print(f"Loaded {len(documents)} documents from {DATA_FILE}")
        index_documents(es, documents, index_name)
    print("\nDone!")


if __name__ == "__main__":
    main()
