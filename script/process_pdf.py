import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from loguru import logger
from src.data.pdf_processor import process_pdf, process_pdfs_batch

PDF_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/processed/arxiv_documents.jsonl")


def main() -> None:
    if not PDF_DIR.exists():
        logger.error(f"PDF directory not found: {PDF_DIR}")
        sys.exit(1)
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in {PDF_DIR}")
        sys.exit(1)
    logger.info(f"Found {len(pdf_files)} PDF files in {PDF_DIR}")

    process_pdfs_batch(PDF_DIR, OUTPUT_PATH)

    if not OUTPUT_PATH.exists():
        logger.error("No documents were created!")
        sys.exit(1)

    num_lines = sum(1 for _ in open(OUTPUT_PATH, encoding="utf-8"))
    file_size = OUTPUT_PATH.stat().st_size / 1024 / 1024
    logger.info(f"Output: {OUTPUT_PATH} ({file_size:.2f} MB, {num_lines} chunks)")


if __name__ == "__main__":
    main()
