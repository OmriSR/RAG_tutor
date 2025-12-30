"""
PDF Parser - Extract structured text from PDF documents.

WHY THIS MATTERS FOR RAG:
PDF parsing is the first step in any document-based RAG system. The quality of
your parsing directly impacts retrieval quality. Key challenges:
1. PDFs store text as positioned characters, not semantic structure
2. Headers, footers, page numbers create noise
3. Detecting headings helps us chunk by document structure (semantic chunking)

INTERVIEW INSIGHT:
"I extract font metadata during parsing to identify headings, which enables
semantic chunking that respects document structure rather than arbitrary splits."
"""

from __future__ import annotations

import pdfplumber
from pathlib import Path
from dataclasses import dataclass, field
from typing import Generator, Any


@dataclass
class LineInfo:
    """
    Represents a single line of text with formatting metadata.

    We track font_size and is_bold to identify headings later.
    This is crucial for semantic chunking - headings signal section boundaries.
    """
    text: str
    font_size: float
    top: float  # Y-position on page, useful for ordering
    is_bold: bool = False


@dataclass
class PageContent:
    """
    Represents extracted content from a single PDF page.

    We keep both raw text (for simple use cases) and structured lines
    (for semantic analysis). The page_number is essential metadata
    for source citations in RAG responses.
    """
    page_number: int
    text: str
    lines: list[LineInfo] = field(default_factory=list)


def extract_lines_with_metadata(page: Any) -> list[LineInfo]:
    """
    Extract text lines with font size and position metadata.

    WHY WE DO THIS:
    pdfplumber gives us individual characters with their properties.
    We group them into lines and aggregate font info to detect headings.

    Args:
        page: A pdfplumber page object

    Returns:
        List of LineInfo objects with text and formatting metadata
    """
    lines = []
    chars = page.chars

    if not chars:
        return lines

    # Group characters into lines based on y-position
    # Characters on the same line have similar 'top' values
    current_line = []
    current_top = None
    tolerance = 3.0  # Pixels - chars within this range are same line

    for char in sorted(chars, key=lambda c: (c['top'], c['x0'])):
        if current_top is None:
            current_top = char['top']
            current_line = [char]
        elif abs(char['top'] - current_top) <= tolerance:
            current_line.append(char)
        else:
            # New line detected - save current and start fresh
            if current_line:
                line_info = _create_line_info(current_line, current_top)
                if line_info.text:
                    lines.append(line_info)
            current_top = char['top']
            current_line = [char]

    # Don't forget the last line
    if current_line and current_top is not None:
        line_info = _create_line_info(current_line, current_top)
        if line_info.text:
            lines.append(line_info)

    return lines


def _create_line_info(chars: list[dict[str, Any]], top: float) -> LineInfo:
    """
    Create a LineInfo object from character data.

    We compute average font size (headings are typically larger) and
    check if any character is bold (another heading indicator).
    """
    line_text = ''.join(c['text'] for c in chars).strip()
    avg_size = sum(c.get('size', 12) for c in chars) / len(chars)
    is_bold = any('bold' in str(c.get('fontname', '')).lower() for c in chars)

    return LineInfo(
        text=line_text,
        font_size=avg_size,
        top=top,
        is_bold=is_bold,
    )


def detect_heading(line: LineInfo, median_font_size: float) -> bool:
    """
    Determine if a line is likely a heading.

    HEADING DETECTION HEURISTICS:
    1. Larger than median font size (titles are bigger)
    2. Bold formatting
    3. Short length (headings aren't paragraphs)

    This is imperfect - real-world PDFs vary widely. But it works
    well enough for most technical books and documents.

    Args:
        line: LineInfo object to check
        median_font_size: Median font size in document for comparison

    Returns:
        True if the line appears to be a heading
    """
    if not line.text:
        return False

    is_larger = line.font_size > median_font_size * 1.1
    is_bold = line.is_bold
    is_short = len(line.text) < 100

    return (is_larger or is_bold) and is_short


def parse_pdf(pdf_path: str | Path) -> Generator[PageContent, None, None]:
    """
    Parse a PDF file and yield structured content page by page.

    WHY A GENERATOR:
    For large PDFs (500+ pages), loading everything into memory is wasteful.
    Generators let us process page-by-page, reducing memory footprint.

    Args:
        pdf_path: Path to the PDF file

    Yields:
        PageContent objects with text and line metadata
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            lines = extract_lines_with_metadata(page)

            yield PageContent(
                page_number=page_num,
                text=text,
                lines=lines,
            )


def parse_pdf_to_list(pdf_path: str | Path) -> list[PageContent]:
    """
    Parse entire PDF and return as list.

    Use this when you need random access to pages or will iterate
    multiple times. For single-pass processing, prefer parse_pdf().
    """
    return list(parse_pdf(pdf_path))
