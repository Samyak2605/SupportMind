from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from supportmind.models import ChunkRecord


def build_combined_text(instruction: str, response: str) -> str:
    return f"Customer issue:\n{instruction.strip()}\n\nApproved support response:\n{response.strip()}"


def chunk_documents(
    rows: list[dict],
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkRecord]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[ChunkRecord] = []

    for row in rows:
        combined = build_combined_text(row["instruction"], row["response"])
        split_chunks = splitter.split_text(combined)
        for index, chunk_text in enumerate(split_chunks):
            chunk_id = f"row-{row['row_id']}-chunk-{index}"
            metadata = {
                "row_id": row["row_id"],
                "chunk_index": index,
                "category": row["category"],
                "intent": row["intent"],
                "instruction": row["instruction"],
            }
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    row_id=row["row_id"],
                    source_instruction=row["instruction"],
                    source_response=row["response"],
                    category=row["category"],
                    intent=row["intent"],
                    text=chunk_text,
                    metadata=metadata,
                )
            )
    return chunks
