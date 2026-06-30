"""
RAG 系统配置文件

使用兼容 OpenAI 格式的 API（DeepSeek、通义千问等）
"""

from pathlib import Path

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class RAGConfig:
    """RAG 系统的所有配置集中管理"""

    # ===== LLM 配置 =====
    # DeepSeek 作为 LLM 服务商
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    )
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "deepseek-chat")
    )

    # ===== Embedding 配置 =====
    # SiliconFlow 的 BGE-M3 作为 Embedding 模型
    # LLM 和 Embedding 用不同的服务商是很常见的做法
    embedding_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_api_key: str = field(
        default_factory=lambda: os.getenv("SILICONFLOW_API_KEY", "")
    )
    embedding_model: str = "Pro/BAAI/bge-m3"
    embedding_dim: int = 1024  # BGE-M3 输出 1024 维向量

    # ===== Reranker 配置 =====
    # Cross-encoder 重排序，SiliconFlow 的 /rerank endpoint
    reranker_base_url: str = "https://api.siliconflow.cn/v1"
    reranker_api_key: str = field(
        default_factory=lambda: os.getenv("SILICONFLOW_API_KEY", "")
    )
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # ===== RAGAS 评估配置 =====
    # RAGAS 内部用 LLM 作为评判（LLM-as-Judge），复用 DeepSeek 凭据
    ragas_llm_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    )
    ragas_llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    ragas_llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "deepseek-chat")
    )
    # RAGAS 评估时同样需要 Embedding 来计算 Answer Relevancy，复用已有配置
    ragas_embedding_base_url: str = "https://api.siliconflow.cn/v1"
    ragas_embedding_api_key: str = field(
        default_factory=lambda: os.getenv("SILICONFLOW_API_KEY", "")
    )
    ragas_embedding_model: str = "Pro/BAAI/bge-m3"

    # ===== PDF OCR 配置 =====
    # 使用 SiliconFlow OpenAI 兼容接口处理扫描版 PDF 和复杂版式
    pdf_ocr_base_url: str = "https://api.siliconflow.cn/v1"
    pdf_ocr_api_key: str = field(
        default_factory=lambda: os.getenv("SILICONFLOW_API_KEY", "")
    )
    pdf_ocr_model: str = field(
        default_factory=lambda: os.getenv(
            "PDF_OCR_MODEL", "PaddlePaddle/PaddleOCR-VL-1.5"
        )
    )
    pdf_ocr_prompt: str = (
        "Convert this PDF page image to clean Markdown. Preserve headings, "
        "paragraphs, tables, lists, and visible text. Do not include bounding-box "
        "tokens or internal model tags."
    )
    pdf_ocr_min_native_text_length: int = 30
    pdf_ocr_image_count_threshold: int = 2
    pdf_ocr_table_count_threshold: int = 2
    pdf_ocr_min_text_quality: float = 0.65
    pdf_ocr_fallback_to_native: bool = True

    # ===== 分块配置 =====
    chunk_size: int = 500  # 每个 chunk 的最大字符数
    chunk_overlap: int = 50  # 相邻 chunk 的重叠字符数

    # ===== 检索配置 =====
    top_k: int = 3  # 检索返回的最相关 chunk 数量

    # ===== 路径配置 =====
    docs_dir: str | Path = Path(__file__).resolve().parent / "docs"  # 知识库文档目录
    db_dir: str | Path = (
        Path(__file__).resolve().parent / "chroma_db"
    )  # 向量数据库持久化目录
    simple_index_file: str | Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "SIMPLE_INDEX_FILE",
                Path(__file__).resolve().parent / "simple_index.json",
            )
        )
    )  # SimpleVectorStore 的本地索引缓存文件


# 全局配置实例
config = RAGConfig()
