"""
PDF 加载器

职责：把单个 PDF 文件解析成适合 RAG 使用的文本。

处理策略：
  - 普通文本 PDF：优先用 PyMuPDF 直接提取文本和表格
  - 扫描页 / 复杂版式：把页面渲染成图片，调用 SiliconFlow 的
    OpenAI 兼容接口进行 OCR / Markdown 解析
"""

import base64
import re
from pathlib import Path
from typing import Any

try:
    # pyrefly: ignore [missing-import]
    from config import config
    from loaders.table_formatter import rows_to_markdown
    from loaders.types import DocumentPart
except ImportError:
    from ..config import config
    from .table_formatter import rows_to_markdown
    from .types import DocumentPart


class PdfLoader:
    """加载 PDF 文件并输出纯文本 / Markdown-ish 文本"""

    def __init__(self):
        self._pdf_ocr_client = None

    def load(self, filepath: Path) -> list[DocumentPart]:
        try:
            import fitz
        except ImportError as e:
            raise ImportError("PDF 加载需要安装 PyMuPDF: uv add pymupdf") from e

        try:
            pdf = fitz.open(filepath)
        except Exception as e:
            raise ValueError(f"无法打开 PDF 文件 {filepath.name}: {e}") from e

        try:
            if pdf.is_encrypted and not pdf.authenticate(""):
                raise ValueError(f"PDF 文件已加密，无法读取: {filepath.name}")

            parts: list[DocumentPart] = []
            for page_index in range(pdf.page_count):
                page_number = page_index + 1
                page = pdf[page_index]
                try:
                    page_text, page_needs_ocr = self._extract_pdf_page(
                        page,
                        page_number,
                    )
                except Exception as e:
                    raise ValueError(
                        f"解析 PDF 文件 {filepath.name} 第 {page_number} 页失败: {e}"
                    ) from e

                if page_needs_ocr:
                    try:
                        page_text = self._parse_pdf_page_with_ocr(page, page_number)
                    except Exception as e:
                        if not (
                            config.pdf_ocr_fallback_to_native
                            and self._has_meaningful_page_text(page_text)
                        ):
                            raise
                        print(
                            "  ⚠️ 远程 PDF OCR/VLM 解析失败，"
                            f"已回退到第 {page_number} 页本地文本: {e}"
                        )

                if page_text.strip():
                    parts.append(
                        DocumentPart(
                            content=page_text,
                            metadata={
                                "part_type": "page",
                                "page": page_number,
                            },
                        )
                    )

            return parts
        finally:
            pdf.close()

    def _extract_pdf_page(self, page: Any, page_number: int) -> tuple[str, bool]:
        tables = self._extract_pdf_tables(page, page_number)
        text_blocks, image_blocks = self._extract_pdf_blocks(
            page,
            tables,
            page_number,
        )

        page_parts = [f"# Page {page_number}"]
        page_parts.extend(
            item["text"]
            for item in sorted(text_blocks + tables, key=self._pdf_item_sort_key)
        )

        for image_index, image_item in enumerate(image_blocks, start=1):
            caption = self._find_image_caption(text_blocks, image_item["bbox"])
            image_parts = [f"[Image page={page_number} index={image_index}]"]
            if caption:
                image_parts.append(f"Caption: {caption}")
            page_parts.append("\n".join(image_parts))

        needs_ocr = self._page_needs_ocr(
            text_blocks=text_blocks,
            image_blocks=image_blocks,
            tables=tables,
        )

        return "\n\n".join(part for part in page_parts if part.strip()), needs_ocr

    def _extract_pdf_tables(self, page: Any, page_number: int) -> list[dict[str, Any]]:
        try:
            finder = page.find_tables()
        except Exception:
            return []

        table_items = []
        for table_index, table in enumerate(getattr(finder, "tables", []), start=1):
            try:
                table_text = table.to_markdown(clean=True)
            except Exception:
                try:
                    rows = table.extract()
                except Exception:
                    continue
                table_text = rows_to_markdown(rows)

            if not table_text or not table_text.strip():
                continue

            table_items.append(
                {
                    "bbox": table.bbox,
                    "text": (
                        f"[Table page={page_number} index={table_index}]\n"
                        f"{table_text.strip()}"
                    ),
                }
            )
        return table_items

    def _extract_pdf_blocks(
        self,
        page: Any,
        tables: list[dict[str, Any]],
        page_number: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        try:
            page_dict = page.get_text("dict", sort=True)
        except Exception as e:
            raise ValueError(f"无法提取页面文本块: {e}") from e

        text_blocks = []
        image_blocks = []
        table_bboxes = [item["bbox"] for item in tables]

        for block in page_dict.get("blocks", []):
            bbox = block.get("bbox")
            if bbox is None:
                continue

            block_type = block.get("type")
            if block_type == 0:
                if self._bbox_inside_any(bbox, table_bboxes):
                    continue
                text = self._extract_text_from_block(block)
                if text:
                    text_blocks.append({"bbox": bbox, "text": text})
            elif block_type == 1:
                image_blocks.append(
                    {"bbox": bbox, "text": f"[Image page={page_number}]"}
                )

        return text_blocks, image_blocks

    def _parse_pdf_page_with_ocr(self, page: Any, page_number: int) -> str:
        try:
            import fitz
        except ImportError as e:
            raise ImportError("PDF OCR 需要安装 PyMuPDF: uv add pymupdf") from e

        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        data_url = (
            "data:image/png;base64,"
            + base64.b64encode(pixmap.tobytes("png")).decode("utf-8")
        )

        try:
            response = self._get_pdf_ocr_client().chat.completions.create(
                model=config.pdf_ocr_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url,
                                    "detail": "high",
                                },
                            },
                            {
                                "type": "text",
                                "text": config.pdf_ocr_prompt,
                            },
                        ],
                    }
                ],
            )
        except Exception as e:
            raise ValueError(
                f"远程 PDF OCR/VLM 解析第 {page_number} 页失败: {e}"
            ) from e

        text = response.choices[0].message.content or ""
        if not text.strip():
            raise ValueError(f"远程 PDF OCR/VLM 未返回第 {page_number} 页有效文本")
        return f"# Page {page_number}\n\n{text.strip()}"

    def _get_pdf_ocr_client(self):
        if self._pdf_ocr_client is not None:
            return self._pdf_ocr_client

        if not config.pdf_ocr_api_key:
            raise ValueError(
                "需要设置 SILICONFLOW_API_KEY 才能调用远程 PDF OCR/VLM"
            )
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("PDF OCR 需要安装 OpenAI SDK: uv add openai") from e

        self._pdf_ocr_client = OpenAI(
            api_key=config.pdf_ocr_api_key,
            base_url=config.pdf_ocr_base_url,
        )
        return self._pdf_ocr_client

    def _page_needs_ocr(
        self,
        text_blocks: list[dict[str, Any]],
        image_blocks: list[dict[str, Any]],
        tables: list[dict[str, Any]],
    ) -> bool:
        native_text = "\n".join(item["text"] for item in text_blocks)
        native_text_length = len(native_text)
        if (
            native_text_length < config.pdf_ocr_min_native_text_length
            and image_blocks
        ):
            return True
        if len(image_blocks) >= config.pdf_ocr_image_count_threshold:
            return True
        if len(tables) >= config.pdf_ocr_table_count_threshold:
            return True
        if (
            native_text_length >= config.pdf_ocr_min_native_text_length
            and self._native_text_quality_score(native_text)
            < config.pdf_ocr_min_text_quality
        ):
            return True
        return False

    def _native_text_quality_score(self, text: str) -> float:
        if not text:
            return 1.0

        visible_chars = [char for char in text if not char.isspace()]
        if not visible_chars:
            return 1.0

        good_chars = 0
        bad_chars = 0
        for char in visible_chars:
            codepoint = ord(char)
            if char.isalnum() or self._is_cjk_char(char) or char in (
                "，。！？；：、,.!?;:()[]{}<>《》“”\"'`-_/\\|+*=#%&@$~^"
            ):
                good_chars += 1
            elif (
                char == "\ufffd"
                or codepoint < 32
                or 0xE000 <= codepoint <= 0xF8FF
            ):
                bad_chars += 1

        score = good_chars / len(visible_chars)
        bad_ratio = bad_chars / len(visible_chars)
        return max(0.0, score - bad_ratio)

    def _is_cjk_char(self, char: str) -> bool:
        codepoint = ord(char)
        return (
            0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
            or 0x20000 <= codepoint <= 0x2A6DF
            or 0x2A700 <= codepoint <= 0x2B73F
            or 0x2B740 <= codepoint <= 0x2B81F
            or 0x2B820 <= codepoint <= 0x2CEAF
        )

    def _has_meaningful_page_text(self, text: str) -> bool:
        meaningful_text = re.sub(r"^# Page \d+\s*", "", text.strip())
        return len(meaningful_text.strip()) >= config.pdf_ocr_min_native_text_length

    def _extract_text_from_block(self, block: dict[str, Any]) -> str:
        lines = []
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(span.get("text", "") for span in spans).strip()
            if line_text:
                lines.append(line_text)
        return "\n".join(lines).strip()

    def _find_image_caption(
        self,
        text_blocks: list[dict[str, Any]],
        image_bbox: Any,
    ) -> str:
        image_rect = self._as_rect_tuple(image_bbox)
        caption_pattern = re.compile(
            r"^\s*(图\s*\d+|figure\s*\d+|fig\.\s*\d+)",
            re.I,
        )

        candidates = []
        for block in text_blocks:
            text = block["text"].strip()
            if not caption_pattern.search(text):
                continue

            rect = self._as_rect_tuple(block["bbox"])
            vertical_gap = min(
                abs(rect[1] - image_rect[3]),
                abs(image_rect[1] - rect[3]),
            )
            horizontal_overlap = min(rect[2], image_rect[2]) - max(
                rect[0], image_rect[0]
            )
            if vertical_gap <= 80 and horizontal_overlap > 0:
                candidates.append((vertical_gap, text))

        if not candidates:
            return ""
        return sorted(candidates, key=lambda item: item[0])[0][1]

    def _bbox_inside_any(self, bbox: Any, containers: list[Any]) -> bool:
        rect = self._as_rect_tuple(bbox)
        for container in containers:
            container_rect = self._as_rect_tuple(container)
            if (
                rect[0] >= container_rect[0]
                and rect[1] >= container_rect[1]
                and rect[2] <= container_rect[2]
                and rect[3] <= container_rect[3]
            ):
                return True
        return False

    def _pdf_item_sort_key(self, item: dict[str, Any]) -> tuple[float, float]:
        x0, y0, _, _ = self._as_rect_tuple(item["bbox"])
        return y0, x0

    def _as_rect_tuple(self, bbox: Any) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = bbox
        return float(x0), float(y0), float(x1), float(y1)
