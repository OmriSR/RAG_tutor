"""
Semantic Chunker - Split documents into meaningful chunks for RAG.

WHY CHUNKING MATTERS:
Chunking is arguably the most impactful decision in RAG system design.
Poor chunking = poor retrieval = wrong answers, no matter how good your LLM.

NAIVE VS SEMANTIC CHUNKING:
- Naive: Split every N characters/tokens (fast, but breaks sentences/concepts)
- Semantic: Respect document structure - split at section/paragraph boundaries

KEY PARAMETERS (interview favorites):
- Chunk size: 500-1000 tokens is typical. Too small = missing context.
                                          Too large = noise drowns signal.
- Overlap: 100-200 tokens. Ensures concepts at chunk boundaries aren't lost.

INTERVIEW INSIGHT:
"I chose semantic chunking because it preserves document structure. A chunk
about 'embeddings' contains the full explanation, not half of it cut off
mid-sentence. This dramatically improves retrieval relevance."
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .pdf_parser import PageContent, detect_heading


@dataclass
class ChunkMetadata:
    """
    Metadata attached to each chunk for retrieval and citation.

    page_numbers: Which pages this chunk spans (for citations)
    section_title: The heading this content falls under
    chunk_index: Position in the document (useful for ordering results)
    """
    page_numbers: list[int] = field(default_factory=list)
    section_title: str = ""
    chunk_index: int = 0


@dataclass
class Chunk:
    """
    A chunk of text ready for embedding and indexing.

    WHY METADATA:
    When the LLM answers a question, we want to cite sources.
    Metadata lets us say "According to Chapter 5, page 142..."
    """
    text: str
    metadata: ChunkMetadata


def compute_median_font_size(pages: list[PageContent]) -> float:
    """
    Compute median font size across all lines in the document.
    Used as baseline for heading detection.
    """
    all_sizes = []
    for page in pages:
        for line in page.lines:
            all_sizes.append(line.font_size)

    if not all_sizes:
        return 12.0  # Default assumption

    sorted_sizes = sorted(all_sizes)
    mid = len(sorted_sizes) // 2
    return sorted_sizes[mid]


def extract_sections(pages: list[PageContent]) -> list[dict[str, Any]]:
    """
    Extract sections from document based on heading detection.

    Returns a list of sections, each containing:
    - title: The section heading
    - content: All text under that heading
    - page_numbers: Pages this section spans

    APPROACH:
    1. Find median font size to calibrate heading detection
    2. Walk through all lines, detecting heading transitions
    3. Group content under each heading
    """
    if not pages:
        return []

    median_size = compute_median_font_size(pages)
    sections = []
    current_section = {
        'title': 'Introduction',  # Default for content before first heading
        'content': [],
        'page_numbers': set()
    }

    for page in pages:
        for line in page.lines:
            if detect_heading(line, median_size):
                # Save previous section if it has content
                if current_section['content']:
                    current_section['page_numbers'] = sorted(current_section['page_numbers'])
                    sections.append(current_section)

                # Start new section
                current_section = {
                    'title': line.text,
                    'content': [],
                    'page_numbers': set()
                }
            else:
                # Add content to current section
                if line.text.strip():
                    current_section['content'].append(line.text)
                    current_section['page_numbers'].add(page.page_number)

    # Don't forget the last section
    if current_section['content']:
        current_section['page_numbers'] = sorted(current_section['page_numbers'])
        sections.append(current_section)

    return sections


def split_into_paragraphs(text: str) -> list[str]:
    """
    Split text into paragraphs based on blank lines or sentence patterns.

    WHY PARAGRAPHS:
    Paragraphs are natural semantic units. A paragraph typically contains
    one complete idea, making it a good atomic unit for chunking.
    """
    # Split on multiple newlines (paragraph breaks)
    paragraphs = re.split(r'\n\s*\n', text)

    # Filter empty paragraphs and strip whitespace
    return [p.strip() for p in paragraphs if p.strip()]


def estimate_tokens(text: str) -> int:
    """
    Rough token count estimation.

    A common heuristic: ~4 characters per token for English.
    This is approximate but good enough for chunk size targeting.
    """
    return len(text) // 4


def create_chunks_with_overlap(
    paragraphs: list[str],
    section_title: str,
    page_numbers: list[int],
    target_size: int = 500,
    overlap: int = 100,
    start_index: int = 0
) -> list[Chunk]:
    """
    Create chunks from paragraphs with overlap between consecutive chunks.

    OVERLAP EXPLAINED:
    If chunk 1 ends with "...embeddings are vectors" and chunk 2 starts with
    "These vectors capture semantic meaning...", without overlap we might
    miss the connection. Overlap ensures boundary concepts appear in both chunks.

    Args:
        paragraphs: List of paragraph texts
        section_title: Heading for metadata
        page_numbers: Pages this section spans
        target_size: Target tokens per chunk (not a hard limit)
        overlap: Tokens to repeat at chunk boundaries
        start_index: Starting chunk index for metadata

    Returns:
        List of Chunk objects
    """
    if not paragraphs:
        return []

    chunks = []
    current_text = []
    current_tokens = 0
    chunk_index = start_index

    for para in paragraphs:
        para_tokens = estimate_tokens(para)

        # If adding this paragraph exceeds target, save current chunk
        if current_tokens + para_tokens > target_size and current_text:
            chunk_text = '\n\n'.join(current_text)
            chunks.append(Chunk(
                text=chunk_text,
                metadata=ChunkMetadata(
                    page_numbers=page_numbers,
                    section_title=section_title,
                    chunk_index=chunk_index
                )
            ))
            chunk_index += 1

            # Keep overlap: find paragraphs from end that fit in overlap budget
            overlap_text = []
            overlap_tokens = 0
            for prev_para in reversed(current_text):
                prev_tokens = estimate_tokens(prev_para)
                if overlap_tokens + prev_tokens <= overlap:
                    overlap_text.insert(0, prev_para)
                    overlap_tokens += prev_tokens
                else:
                    break

            current_text = overlap_text
            current_tokens = overlap_tokens

        current_text.append(para)
        current_tokens += para_tokens

    # Don't forget the last chunk
    if current_text:
        chunk_text = '\n\n'.join(current_text)
        chunks.append(Chunk(
            text=chunk_text,
            metadata=ChunkMetadata(
                page_numbers=page_numbers,
                section_title=section_title,
                chunk_index=chunk_index
            )
        ))

    return chunks


def semantic_chunk(
    pages: list[PageContent],
    target_chunk_size: int = 500,
    chunk_overlap: int = 100
) -> list[Chunk]:
    """
    Main entry point: convert parsed pages into semantic chunks.

    PIPELINE:
    1. Extract sections based on headings
    2. Split each section into paragraphs
    3. Combine paragraphs into chunks with target size and overlap
    4. Attach metadata for retrieval and citation

    Args:
        pages: List of PageContent from PDF parser
        target_chunk_size: Target tokens per chunk
        chunk_overlap: Tokens to overlap between chunks

    Returns:
        List of Chunk objects ready for embedding
    """
    sections = extract_sections(pages)
    all_chunks = []
    chunk_index = 0

    for section in sections:
        section_text = '\n'.join(section['content'])
        paragraphs = split_into_paragraphs(section_text)

        chunks = create_chunks_with_overlap(
            paragraphs=paragraphs,
            section_title=section['title'],
            page_numbers=section['page_numbers'],
            target_size=target_chunk_size,
            overlap=chunk_overlap,
            start_index=chunk_index
        )

        all_chunks.extend(chunks)
        chunk_index += len(chunks)

    return all_chunks
