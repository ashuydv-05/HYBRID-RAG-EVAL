import json
import logging
from pathlib import Path
from typing import Any, cast
from loguru import logger
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc.labels import DocItemLabel

TOKENIZER = "sentence-transformers/all-MiniLM-L6-v2"


def create_converter() -> DocumentConverter:
    pipeline = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=True,
        table_structure_options=TableStructureOptions(
            do_cell_matching=True, mode=TableFormerMode.ACCURATE
        ),
        ocr_options=EasyOcrOptions(lang=["en"]),
        accelerator_options=AcceleratorOptions(
            num_threads=2, device=AcceleratorDevice.CPU
        ),
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline, backend=PyPdfiumDocumentBackend
            )
        }
    )


def extract_title(doc) -> str:
    for item, _ in doc.iterate_items():
        if getattr(item, "label", None) == DocItemLabel.SECTION_HEADER:
            return item.text
    return ""


def extract_year(paper_id: str) -> int | None:
    try:
        yy = int(paper_id[:2])
        return 1900 + yy if yy >= 90 else 2000 + yy
    except (ValueError, TypeError, IndexError):
        return None


def extract_chunk_meta(chunk) -> dict[str, Any]:
    meta = {"headings": [], "page_info": None, "content_type": None}
    if not hasattr(chunk, "meta"):
        return meta
    if hasattr(chunk.meta, "headings") and chunk.meta.headings:
        meta["headings"] = chunk.meta.headings
    if hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
        for item in chunk.meta.doc_items:
            if hasattr(item, "label"):
                meta["content_type"] = str(item.label)
            if hasattr(item, "prov") and item.prov:
                for prov in item.prov:
                    if hasattr(prov, "page_no"):
                        meta["page_info"] = prov.page_no
    return meta


def process_pdf(
    pdf_path: str | Path, paper_id: str | None = None
) -> list[dict[str, Any]]:
    pdf_path = Path(pdf_path)
    paper_id = paper_id or pdf_path.stem
    logger.info(f"Processing: {pdf_path.name}")
    try:
        doc = create_converter().convert(str(pdf_path)).document
    except Exception as e:
        logger.error(f"Failed to convert {pdf_path}: {e}")
        return []
    title = extract_title(doc) or paper_id
    year = extract_year(paper_id)
    try:
        chunks = list(HybridChunker(tokenizer=cast(Any, TOKENIZER)).chunk(doc))
    except Exception as e:
        logger.error(f"Failed to chunk {pdf_path}: {e}")
        return []
    documents = []
    for i, chunk in enumerate(chunks):
        meta = extract_chunk_meta(chunk)
        headings = meta["headings"]
        documents.append(
            {
                "id": i,
                "content": chunk.text,
                "paper_id": paper_id,
                "title": title,
                "year": year,
                "section": headings[-1] if headings else "Unknown",
                "content_type": meta["content_type"],
                "page_info": meta["page_info"],
            }
        )
    logger.info(f"Created {len(documents)} chunks for '{title}'")
    return documents


def process_pdfs_batch(
    pdf_dir: str | Path, output_path: str | Path | None = None
) -> list[dict[str, Any]]:
    import gc

    pdf_dir = Path(pdf_dir)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir}")
        return []
    logger.info(f"Found {len(pdf_files)} PDFs in {pdf_dir}")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()
        logger.info(f"Output: {output_path} (JSONL format, incremental)")

    global_counter = 0
    total_chunks = 0
    for i, pdf_path in enumerate(pdf_files, 1):
        try:
            logger.info(f"[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
            docs = process_pdf(pdf_path)
            for doc in docs:
                doc["id"] = global_counter
                global_counter += 1

            if output_path and docs:
                with open(output_path, "a", encoding="utf-8") as f:
                    for doc in docs:
                        f.write(json.dumps(doc, ensure_ascii=False) + "\n")

            num_chunks = len(docs)
            total_chunks += num_chunks
            logger.info(
                f"[{i}/{len(pdf_files)}] ✓ Created {num_chunks} chunks from {pdf_path.name} (global_id: {global_counter - num_chunks} -> {global_counter - 1})"
            )
            del docs
            gc.collect()
        except Exception as e:
            logger.error(f"[{i}/{len(pdf_files)}] ✗ Failed {pdf_path.name}: {e}")

    logger.info(f"Total: {total_chunks} chunks (IDs: 0 -> {global_counter - 1})")
    return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    process_pdfs_batch("data/raw", "data/processed/arxiv_documents.jsonl")
