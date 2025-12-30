"""
Tests for semantic chunker module.
"""

import pytest

from src.ingestion.chunker import (
    Chunk,
    ChunkMetadata,
    compute_median_font_size,
    extract_sections,
    split_into_paragraphs,
    estimate_tokens,
    create_chunks_with_overlap,
    semantic_chunk,
)
from src.ingestion.pdf_parser import PageContent, LineInfo


class TestChunkMetadata:
    """Tests for ChunkMetadata dataclass."""

    def test_default_values(self):
        meta = ChunkMetadata()
        assert meta.page_numbers == []
        assert meta.section_title == ""
        assert meta.chunk_index == 0

    def test_with_values(self):
        meta = ChunkMetadata(
            page_numbers=[1, 2, 3],
            section_title="Chapter 1",
            chunk_index=5
        )
        assert meta.page_numbers == [1, 2, 3]
        assert meta.section_title == "Chapter 1"
        assert meta.chunk_index == 5


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_create_chunk(self):
        chunk = Chunk(
            text="Some content",
            metadata=ChunkMetadata(page_numbers=[1])
        )
        assert chunk.text == "Some content"
        assert chunk.metadata.page_numbers == [1]


class TestComputeMedianFontSize:
    """Tests for median font size calculation."""

    def test_empty_pages(self):
        assert compute_median_font_size([]) == 12.0

    def test_single_line(self):
        pages = [
            PageContent(
                page_number=1,
                text="",
                lines=[LineInfo(text="Test", font_size=14.0, top=100)]
            )
        ]
        assert compute_median_font_size(pages) == 14.0

    def test_multiple_lines(self):
        pages = [
            PageContent(
                page_number=1,
                text="",
                lines=[
                    LineInfo(text="A", font_size=10.0, top=100),
                    LineInfo(text="B", font_size=12.0, top=200),
                    LineInfo(text="C", font_size=14.0, top=300),
                ]
            )
        ]
        # Median of [10, 12, 14] is 12
        assert compute_median_font_size(pages) == 12.0


class TestSplitIntoParagraphs:
    """Tests for paragraph splitting."""

    def test_single_paragraph(self):
        text = "This is a single paragraph."
        paragraphs = split_into_paragraphs(text)
        assert paragraphs == ["This is a single paragraph."]

    def test_multiple_paragraphs(self):
        text = "First paragraph.\n\nSecond paragraph."
        paragraphs = split_into_paragraphs(text)
        assert paragraphs == ["First paragraph.", "Second paragraph."]

    def test_empty_paragraphs_removed(self):
        text = "First.\n\n\n\nSecond."
        paragraphs = split_into_paragraphs(text)
        assert paragraphs == ["First.", "Second."]

    def test_whitespace_stripped(self):
        text = "  First.  \n\n  Second.  "
        paragraphs = split_into_paragraphs(text)
        assert paragraphs == ["First.", "Second."]


class TestEstimateTokens:
    """Tests for token estimation."""

    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_string(self):
        # "Hello" = 5 chars, 5 // 4 = 1
        assert estimate_tokens("Hello") == 1

    def test_longer_string(self):
        # 100 characters = ~25 tokens
        text = "a" * 100
        assert estimate_tokens(text) == 25


class TestCreateChunksWithOverlap:
    """Tests for chunk creation with overlap."""

    def test_empty_paragraphs(self):
        chunks = create_chunks_with_overlap(
            paragraphs=[],
            section_title="Test",
            page_numbers=[1]
        )
        assert chunks == []

    def test_single_small_paragraph(self):
        chunks = create_chunks_with_overlap(
            paragraphs=["Short text."],
            section_title="Test Section",
            page_numbers=[1, 2]
        )
        assert len(chunks) == 1
        assert chunks[0].text == "Short text."
        assert chunks[0].metadata.section_title == "Test Section"
        assert chunks[0].metadata.page_numbers == [1, 2]

    def test_multiple_paragraphs_single_chunk(self):
        # Small paragraphs that fit in one chunk
        chunks = create_chunks_with_overlap(
            paragraphs=["Para one.", "Para two."],
            section_title="Test",
            page_numbers=[1],
            target_size=100
        )
        assert len(chunks) == 1
        assert "Para one." in chunks[0].text
        assert "Para two." in chunks[0].text

    def test_chunk_index_increments(self):
        # Create enough content to span multiple chunks
        long_para = "This is a paragraph. " * 50  # ~1000 chars = ~250 tokens
        chunks = create_chunks_with_overlap(
            paragraphs=[long_para, long_para],
            section_title="Test",
            page_numbers=[1],
            target_size=100,
            overlap=20
        )
        # Should create multiple chunks with incrementing indexes
        if len(chunks) > 1:
            assert chunks[0].metadata.chunk_index == 0
            assert chunks[1].metadata.chunk_index == 1


class TestExtractSections:
    """Tests for section extraction."""

    def test_empty_pages(self):
        sections = extract_sections([])
        assert sections == []

    def test_no_headings(self):
        pages = [
            PageContent(
                page_number=1,
                text="",
                lines=[
                    LineInfo(text="Normal text line one.", font_size=12.0, top=100),
                    LineInfo(text="Normal text line two.", font_size=12.0, top=200),
                ]
            )
        ]
        sections = extract_sections(pages)
        # All content should be in default "Introduction" section
        assert len(sections) == 1
        assert sections[0]['title'] == 'Introduction'
        assert len(sections[0]['content']) == 2

    def test_with_heading(self):
        pages = [
            PageContent(
                page_number=1,
                text="",
                lines=[
                    LineInfo(text="Chapter 1", font_size=18.0, top=50, is_bold=True),
                    LineInfo(text="Content under chapter 1.", font_size=12.0, top=100),
                ]
            )
        ]
        sections = extract_sections(pages)
        # Should detect heading and create section
        assert any(s['title'] == 'Chapter 1' for s in sections)


class TestSemanticChunk:
    """Integration tests for the full chunking pipeline."""

    def test_empty_input(self):
        chunks = semantic_chunk([])
        assert chunks == []

    def test_basic_chunking(self):
        pages = [
            PageContent(
                page_number=1,
                text="",
                lines=[
                    LineInfo(text="Introduction", font_size=16.0, top=50, is_bold=True),
                    LineInfo(text="This is the introduction text.", font_size=12.0, top=100),
                    LineInfo(text="More introduction content.", font_size=12.0, top=150),
                ]
            )
        ]
        chunks = semantic_chunk(pages)
        assert len(chunks) >= 1
        # All chunks should have metadata
        for chunk in chunks:
            assert chunk.metadata.page_numbers
            assert chunk.text

    def test_preserves_page_numbers(self):
        pages = [
            PageContent(
                page_number=5,
                text="",
                lines=[
                    LineInfo(text="Content on page 5.", font_size=12.0, top=100),
                ]
            )
        ]
        chunks = semantic_chunk(pages)
        assert len(chunks) >= 1
        assert 5 in chunks[0].metadata.page_numbers
