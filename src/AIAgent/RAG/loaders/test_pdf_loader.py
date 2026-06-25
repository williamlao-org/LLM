from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

RAG_DIR = Path(__file__).resolve().parents[1]
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

from config import config
from loaders.pdf_loader import PdfLoader


def text_block(text, bbox=(0, 0, 100, 20)):
    return {
        "type": 0,
        "bbox": bbox,
        "lines": [{"spans": [{"text": text}]}],
    }


def image_block(bbox=(0, 40, 100, 140)):
    return {"type": 1, "bbox": bbox}


class FakePage:
    def __init__(self, blocks=None):
        self.blocks = blocks or []

    def get_text(self, fmt, sort=True):
        assert fmt == "dict"
        return {"blocks": self.blocks}

    def find_tables(self):
        return SimpleNamespace(tables=[])

    def get_pixmap(self, matrix, alpha=False):
        return SimpleNamespace(tobytes=lambda fmt: b"fake-png")


class FakePdf:
    is_encrypted = False
    page_count = 1

    def __getitem__(self, index):
        return FakePage()

    def authenticate(self, password):
        return True

    def close(self):
        pass


class FakeCompletions:
    def __init__(self, content="OCR markdown"):
        self.content = content
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_image_placeholder_is_kept_without_caption():
    loader = PdfLoader()
    page = FakePage(
        [
            text_block("This page has enough native text to avoid OCR.", (0, 0, 200, 20)),
            image_block((0, 40, 100, 140)),
        ]
    )

    text, needs_ocr = loader._extract_pdf_page(page, 1)

    assert needs_ocr is False
    assert "[Image page=1 index=1]" in text


def test_image_caption_is_kept_when_present():
    loader = PdfLoader()
    page = FakePage(
        [
            text_block("Enough native text before the image to avoid OCR.", (0, 0, 200, 20)),
            image_block((0, 40, 100, 140)),
            text_block("图 1 系统架构图", (0, 145, 100, 165)),
        ]
    )

    text, needs_ocr = loader._extract_pdf_page(page, 1)

    assert needs_ocr is False
    assert "[Image page=1 index=1]" in text
    assert "Caption: 图 1 系统架构图" in text


def test_short_text_with_image_needs_ocr():
    loader = PdfLoader()

    assert loader._page_needs_ocr(
        text_blocks=[{"text": "短文本"}],
        image_blocks=[{"bbox": (0, 0, 10, 10)}],
        tables=[],
    )


def test_multiple_images_need_ocr():
    loader = PdfLoader()

    assert loader._page_needs_ocr(
        text_blocks=[{"text": "This page has enough native text to avoid short text."}],
        image_blocks=[
            {"bbox": (0, 0, 10, 10)},
            {"bbox": (20, 0, 30, 10)},
        ],
        tables=[],
    )


def test_multiple_tables_need_ocr():
    loader = PdfLoader()

    assert loader._page_needs_ocr(
        text_blocks=[{"text": "This page has enough native text to avoid short text."}],
        image_blocks=[],
        tables=[{"bbox": (0, 0, 10, 10)}, {"bbox": (20, 0, 30, 10)}],
    )


def test_normal_long_text_does_not_need_ocr():
    loader = PdfLoader()

    assert not loader._page_needs_ocr(
        text_blocks=[
            {
                "text": (
                    "这是一段正常的 PDF 原生文本，包含中文、English words, "
                    "and punctuation."
                )
            }
        ],
        image_blocks=[],
        tables=[],
    )


def test_garbled_native_text_needs_ocr():
    loader = PdfLoader()

    assert loader._page_needs_ocr(
        text_blocks=[{"text": "\ufffd" * 40}],
        image_blocks=[],
        tables=[],
    )


def test_ocr_success_uses_configured_prompt(monkeypatch):
    loader = PdfLoader()
    completions = FakeCompletions("# OCR result")
    loader._pdf_ocr_client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    monkeypatch.setattr(config, "pdf_ocr_prompt", "Generic OCR prompt")
    monkeypatch.setattr(config, "pdf_ocr_model", "provider/any-vlm")

    text = loader._parse_pdf_page_with_ocr(FakePage(), 3)

    assert text == "# Page 3\n\n# OCR result"
    assert completions.last_kwargs["model"] == "provider/any-vlm"
    content = completions.last_kwargs["messages"][0]["content"]
    assert content[1]["text"] == "Generic OCR prompt"
    assert "<|grounding|>" not in content[1]["text"]
    assert "<image>" not in content[1]["text"]


def test_ocr_failure_falls_back_to_native_text(monkeypatch, capsys):
    import fitz

    loader = PdfLoader()
    native_text = "# Page 1\n\nThis is enough native text to use as fallback."

    monkeypatch.setattr(fitz, "open", lambda filepath: FakePdf())
    monkeypatch.setattr(
        loader,
        "_extract_pdf_page",
        lambda page, number: (native_text, True),
    )
    monkeypatch.setattr(
        loader,
        "_parse_pdf_page_with_ocr",
        lambda page, number: (_ for _ in ()).throw(ValueError("boom")),
    )

    parts = loader.load(Path("fake.pdf"))

    assert len(parts) == 1
    assert parts[0].content == native_text
    assert parts[0].metadata == {"part_type": "page", "page": 1}
    assert "远程 PDF OCR/VLM 解析失败" in capsys.readouterr().out


def test_missing_api_key_error_is_not_paddle_specific(monkeypatch):
    loader = PdfLoader()
    monkeypatch.setattr(config, "pdf_ocr_api_key", "")

    with pytest.raises(ValueError) as exc_info:
        loader._get_pdf_ocr_client()

    message = str(exc_info.value)
    assert "远程 PDF OCR/VLM" in message
    assert "Paddle" not in message
