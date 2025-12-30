"""
Tests for PDF parser module.

These tests verify the core parsing functionality without requiring
an actual PDF file (using mocks).
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.ingestion.pdf_parser import (
    LineInfo,
    PageContent,
    extract_lines_with_metadata,
    _create_line_info,
    detect_heading,
    parse_pdf,
    parse_pdf_to_list,
)


class TestLineInfo:
    """Tests for LineInfo dataclass."""

    def test_create_line_info(self):
        line = LineInfo(text="Hello", font_size=12.0, top=100.0)
        assert line.text == "Hello"
        assert line.font_size == 12.0
        assert line.top == 100.0
        assert line.is_bold is False

    def test_create_bold_line(self):
        line = LineInfo(text="Title", font_size=16.0, top=50.0, is_bold=True)
        assert line.is_bold is True


class TestPageContent:
    """Tests for PageContent dataclass."""

    def test_create_page_content(self):
        page = PageContent(page_number=1, text="Sample text")
        assert page.page_number == 1
        assert page.text == "Sample text"
        assert page.lines == []

    def test_page_with_lines(self):
        lines = [LineInfo(text="Line 1", font_size=12.0, top=100.0)]
        page = PageContent(page_number=1, text="Line 1", lines=lines)
        assert len(page.lines) == 1


class TestCreateLineInfo:
    """Tests for _create_line_info helper function."""

    def test_basic_chars(self):
        chars = [
            {'text': 'H', 'size': 12, 'fontname': 'Arial'},
            {'text': 'i', 'size': 12, 'fontname': 'Arial'},
        ]
        line = _create_line_info(chars, top=100.0)
        assert line.text == "Hi"
        assert line.font_size == 12.0
        assert line.is_bold is False

    def test_bold_detection(self):
        chars = [
            {'text': 'B', 'size': 14, 'fontname': 'Arial-Bold'},
            {'text': 'old', 'size': 14, 'fontname': 'Arial-Bold'},
        ]
        line = _create_line_info(chars, top=50.0)
        assert line.is_bold is True

    def test_mixed_sizes_averaged(self):
        chars = [
            {'text': 'A', 'size': 10},
            {'text': 'B', 'size': 14},
        ]
        line = _create_line_info(chars, top=100.0)
        assert line.font_size == 12.0  # Average of 10 and 14


class TestDetectHeading:
    """Tests for heading detection logic."""

    def test_larger_font_is_heading(self):
        line = LineInfo(text="Chapter 1", font_size=16.0, top=100.0)
        assert detect_heading(line, median_font_size=12.0) is True

    def test_bold_is_heading(self):
        line = LineInfo(text="Section Title", font_size=12.0, top=100.0, is_bold=True)
        assert detect_heading(line, median_font_size=12.0) is True

    def test_normal_text_not_heading(self):
        line = LineInfo(text="This is normal paragraph text.", font_size=12.0, top=100.0)
        assert detect_heading(line, median_font_size=12.0) is False

    def test_long_text_not_heading(self):
        # Even with large font, very long text is probably not a heading
        long_text = "A" * 150
        line = LineInfo(text=long_text, font_size=16.0, top=100.0)
        assert detect_heading(line, median_font_size=12.0) is False

    def test_empty_text_not_heading(self):
        line = LineInfo(text="", font_size=16.0, top=100.0)
        assert detect_heading(line, median_font_size=12.0) is False


class TestExtractLinesWithMetadata:
    """Tests for line extraction from page."""

    def test_empty_page(self):
        page = MagicMock()
        page.chars = []
        lines = extract_lines_with_metadata(page)
        assert lines == []

    def test_single_line(self):
        page = MagicMock()
        page.chars = [
            {'text': 'H', 'top': 100, 'x0': 0, 'size': 12, 'fontname': 'Arial'},
            {'text': 'i', 'top': 100, 'x0': 10, 'size': 12, 'fontname': 'Arial'},
        ]
        lines = extract_lines_with_metadata(page)
        assert len(lines) == 1
        assert lines[0].text == "Hi"

    def test_multiple_lines(self):
        page = MagicMock()
        page.chars = [
            {'text': 'A', 'top': 100, 'x0': 0, 'size': 12, 'fontname': 'Arial'},
            {'text': 'B', 'top': 200, 'x0': 0, 'size': 12, 'fontname': 'Arial'},
        ]
        lines = extract_lines_with_metadata(page)
        assert len(lines) == 2


class TestParsePdf:
    """Tests for PDF parsing functions."""

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            list(parse_pdf("/nonexistent/path.pdf"))

    @patch('src.ingestion.pdf_parser.pdfplumber')
    def test_parse_pdf_yields_pages(self, mock_pdfplumber):
        # Setup mock
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page content"
        mock_page.chars = []

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        mock_pdfplumber.open.return_value = mock_pdf

        # Create a temp file path that "exists"
        with patch.object(Path, 'exists', return_value=True):
            pages = list(parse_pdf("/fake/path.pdf"))

        assert len(pages) == 1
        assert pages[0].page_number == 1
        assert pages[0].text == "Page content"

    @patch('src.ingestion.pdf_parser.parse_pdf')
    def test_parse_pdf_to_list(self, mock_parse_pdf):
        mock_parse_pdf.return_value = iter([
            PageContent(page_number=1, text="Page 1"),
            PageContent(page_number=2, text="Page 2"),
        ])

        pages = parse_pdf_to_list("/fake/path.pdf")

        assert len(pages) == 2
        assert pages[0].page_number == 1
        assert pages[1].page_number == 2
