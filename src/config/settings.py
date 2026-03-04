import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
load_dotenv()

@dataclass(frozen=True)
class QdrantSettings:
    url: str = os.getenv('QDRANT_URL', 'http://localhost:6333')
    api_key: str | None = os.getenv('QDRANT_API_KEY', None)
    collection: str = os.getenv('QDRANT_COLLECTION', 'arxiv_papers')

@dataclass(frozen=True)
class ElasticsearchSettings:
    url: str = os.getenv('ES_URL', 'http://localhost:9200')
    index: str = os.getenv('ES_INDEX', 'arxiv_papers')

@dataclass(frozen=True)
class EmbeddingSettings:
    dense_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    dense_vector_name: str = 'dense'

@dataclass(frozen=True)
class RerankerSettings:
    model: str = os.getenv('RERANKER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
    top_k: int = int(os.getenv('RERANKER_TOP_K', '5'))

@dataclass(frozen=True)
class HybridRetrievalSettings:
    rrf_k: int = int(os.getenv('RRF_K', '60'))

@dataclass(frozen=True)
class Settings:
    qdrant: QdrantSettings = field(default_factory=QdrantSettings)
    elasticsearch: ElasticsearchSettings = field(default_factory=ElasticsearchSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = field(default_factory=RerankerSettings)
    hybrid_retrieval: HybridRetrievalSettings = field(default_factory=HybridRetrievalSettings)
settings = Settings()