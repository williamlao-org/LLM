"""
文本分块器

职责：把长文档切成适合检索的小块（chunk）

本模块实现了三种分块策略：
1. 固定大小分块 (fixed) —— 最简单，按固定长度切分
2. 句子分块 (sentence) —— 按标点符号等句子边界切分，并进行合并
3. 递归字符分块 (recursive) —— 更智能，尽量保持语义完整

为什么需要分块？
- LLM 上下文窗口有限，不能把整个知识库塞进去
- 检索粒度：整篇文档太大，匹配到但大部分内容无关
- 小块检索更精准，只注入真正相关的内容
"""
import re
from pathlib import Path
from dataclasses import dataclass, field
from document_loader import Document


@dataclass
class Chunk:
    """表示文档的一个分块"""
    content: str  # 分块文本内容
    metadata: dict = field(default_factory=dict)  # 元数据（继承自文档 + 分块信息）

    def __repr__(self):
        preview = self.content[:60].replace("\n", " ")
        source = self.metadata.get("source", "?")
        idx = self.metadata.get("chunk_index", "?")
        return f"Chunk(source={source}, idx={idx}, len={len(self.content)}, '{preview}...')"


# ========== 策略 1：固定大小分块 ==========

def fixed_size_chunk(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """
    固定大小分块

    原理很简单：
    - 从头开始，每次取 chunk_size 个字符作为一个块
    - 下一个块从 (当前起点 + chunk_size - overlap) 开始
    - overlap 保证块与块之间有重叠，不会在边界处丢信息

    示例 (chunk_size=10, overlap=3):
        文本: "ABCDEFGHIJKLMNOPQRST"
        Chunk 1: "ABCDEFGHIJ"   (0-9)
        Chunk 2: "HIJKLMNOPQ"   (7-16)  ← 从第7个开始，和上一块重叠 HIJ
        Chunk 3: "OPQRST"       (14-19) ← 最后一块可能不满

    Args:
        text: 要分块的文本
        chunk_size: 每块的最大字符数
        chunk_overlap: 相邻块的重叠字符数

    Returns:
        分块后的文本列表
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(f"overlap ({chunk_overlap}) 必须小于 chunk_size ({chunk_size})")

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        # 取一块
        end = start + chunk_size
        chunk = text[start:end]

        # 去掉首尾空白（但保留中间的）
        if chunk.strip():
            chunks.append(chunk.strip())

        # 移动起始位置（步长 = chunk_size - overlap）
        start += chunk_size - chunk_overlap

    return chunks


# ========== 策略 2：句子分块 ==========

def sentence_chunk(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """
    句子分块

    核心思想：按句子边界（句号、问号、叹号、换行等）切分，并进行合并，保证每个 chunk 尽量接近 chunk_size。

    Args:
        text: 要分块的文本
        chunk_size: 每块的最大字符数
        chunk_overlap: 相邻块的重叠字符数

    Returns:
        分块后的文本列表
    """
    if not text:
        return []

    # 按标点符号切分，利用 lookbehind 保留分隔符
    # 匹配：中文句号、问号、叹号、换行，或者英文句号、问号、叹号
    sentence_delimiters = r'(?<=[。！？\n])|(?<=[.!?])'
    sentences = re.split(sentence_delimiters, text)
    # 过滤掉空串
    sentences = [s for s in sentences if s]

    chunks = []
    current_chunk_parts = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # 如果单句就超过 chunk_size，单独处理
        if sentence_len > chunk_size:
            # 先保存已积累的 chunk
            if current_chunk_parts:
                chunks.append("".join(current_chunk_parts).strip())
                current_chunk_parts = []
                current_length = 0

            # 对这个超长单句进行硬切分（固定大小）
            start = 0
            while start < sentence_len:
                end = start + chunk_size
                chunk = sentence[start:end]
                if chunk.strip():
                    chunks.append(chunk.strip())
                start += chunk_size - chunk_overlap
            continue

        # 如果合并后超过 chunk_size，保存当前积累，并处理 overlap
        if current_length + sentence_len > chunk_size:
            chunks.append("".join(current_chunk_parts).strip())

            # 处理 overlap
            overlap_parts = []
            overlap_len = 0
            for part in reversed(current_chunk_parts):
                if overlap_len + len(part) > chunk_overlap:
                    break
                overlap_parts.insert(0, part)
                overlap_len += len(part)

            current_chunk_parts = overlap_parts
            current_length = sum(len(p) for p in current_chunk_parts)

        current_chunk_parts.append(sentence)
        current_length += sentence_len

    if current_chunk_parts:
        chunks.append("".join(current_chunk_parts).strip())

    return chunks


# ========== 策略 3：递归字符分块 ==========

def recursive_chunk(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: list[str] | None = None,
) -> list[str]:
    """
    递归字符分块

    核心思想：优先在"语义边界"处切割
    按照分隔符的优先级依次尝试：
    1. 先按 "\\n\\n"（段落）分
    2. 段落太大 → 按 "\\n"（行）分
    3. 行太大 → 按 "。" 或 ". "（句子）分
    4. 句子太大 → 按 " "（空格/词）分
    5. 词太大 → 按字符分

    这样尽量保证切割发生在自然的语义边界上。

    Args:
        text: 要分块的文本
        chunk_size: 每块的最大字符数
        chunk_overlap: 相邻块的重叠字符数
        separators: 分隔符优先级列表

    Returns:
        分块后的文本列表
    """
    if separators is None:
        # 默认分隔符优先级（从高到低）
        separators = ["\n\n", "\n", "。", ". ", " ", ""]

    chunks = []

    # 取当前最高优先级的分隔符
    separator = separators[0]
    remaining_separators = separators[1:]

    # 用当前分隔符切割文本
    if separator == "":
        # 最后一级：按字符切割
        splits = list(text)
    else:
        splits = text.split(separator)

    # 把切割后的小段合并成 chunk_size 以内的块
    current_chunk_parts = []
    current_length = 0

    for split in splits:
        split_len = len(split)

        # 如果单个 split 就超过 chunk_size，需要用更细的分隔符继续切
        if split_len > chunk_size and remaining_separators:
            # 先把已积累的部分存下来
            if current_chunk_parts:
                chunk_text = separator.join(current_chunk_parts).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                current_chunk_parts = []
                current_length = 0

            # 递归：用下一级分隔符切割这个大段
            sub_chunks = recursive_chunk(
                split, chunk_size, chunk_overlap, remaining_separators
            )
            chunks.extend(sub_chunks)
            continue

        # 如果加上这个 split 会超过 chunk_size，先把当前积累的存下来
        separator_len = len(separator) if current_chunk_parts else 0
        if current_length + separator_len + split_len > chunk_size and current_chunk_parts:
            chunk_text = separator.join(current_chunk_parts).strip()
            if chunk_text:
                chunks.append(chunk_text)

            # 处理 overlap：保留尾部一些 parts
            # 简化实现：从后往前保留，直到总长度不超过 overlap
            overlap_parts = []
            overlap_len = 0
            for part in reversed(current_chunk_parts):
                if overlap_len + len(part) > chunk_overlap:
                    break
                overlap_parts.insert(0, part)
                overlap_len += len(part) + len(separator)

            current_chunk_parts = overlap_parts
            current_length = sum(len(p) for p in current_chunk_parts) + len(separator) * max(0, len(current_chunk_parts) - 1)

        # 把当前 split 加入积累
        current_chunk_parts.append(split)
        current_length += separator_len + split_len

    # 别忘了最后一块
    if current_chunk_parts:
        chunk_text = separator.join(current_chunk_parts).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


# ========== 统一接口 ==========

def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    strategy: str = "recursive",
) -> list[Chunk]:
    """
    对文档列表进行分块

    Args:
        documents: Document 对象列表
        chunk_size: 每块的最大字符数
        chunk_overlap: 相邻块的重叠字符数
        strategy: 分块策略，"fixed"、"sentence" 或 "recursive"

    Returns:
        Chunk 对象列表
    """
    all_chunks = []

    # 选择分块函数
    if strategy == "fixed":
        chunk_fn = lambda text: fixed_size_chunk(text, chunk_size, chunk_overlap)
    elif strategy == "sentence":
        chunk_fn = lambda text: sentence_chunk(text, chunk_size, chunk_overlap)
    elif strategy == "recursive":
        chunk_fn = lambda text: recursive_chunk(text, chunk_size, chunk_overlap)
    else:
        raise ValueError(f"未知的分块策略: {strategy}")

    for doc in documents:
        # 对每个文档进行分块
        text_chunks = chunk_fn(doc.content)

        for i, text in enumerate(text_chunks):
            chunk = Chunk(
                content=text,
                metadata={
                    **doc.metadata,  # 继承文档的元数据
                    "chunk_index": i,
                    "chunk_total": len(text_chunks),
                    "chunk_strategy": strategy,
                },
            )
            all_chunks.append(chunk)

        print(f"  📄 {doc.metadata.get('source', '?')} → {len(text_chunks)} 个 chunks")

    print(f"\n🔪 共生成 {len(all_chunks)} 个 chunks (策略: {strategy}, "
          f"chunk_size: {chunk_size}, overlap: {chunk_overlap})")
    return all_chunks


# ===== 测试 =====
if __name__ == "__main__":
    # 测试固定大小分块
    test_text = "这是第一段话。" * 20 + "\n\n" + "这是第二段话。" * 20

    print("=" * 50)
    print("固定大小分块:")
    print("=" * 50)
    chunks = fixed_size_chunk(test_text, chunk_size=100, chunk_overlap=20)
    for i, c in enumerate(chunks):
        print(f"  Chunk {i}: ({len(c)} chars) {c[:50]}...")

    print("\n" + "=" * 50)
    print("句子分块:")
    print("=" * 50)
    chunks = sentence_chunk(test_text, chunk_size=100, chunk_overlap=20)
    for i, c in enumerate(chunks):
        print(f"  Chunk {i}: ({len(c)} chars) {c[:50]}...")

    print("\n" + "=" * 50)
    print("递归字符分块:")
    print("=" * 50)
    chunks = recursive_chunk(test_text, chunk_size=100, chunk_overlap=20)
    for i, c in enumerate(chunks):
        print(f"  Chunk {i}: ({len(c)} chars) {c[:50]}...")

    # 测试完整文档分块流程
    print("\n" + "=" * 50)
    print("完整文档分块:")
    print("=" * 50)
    from document_loader import load_documents
    docs_dir = Path(__file__).resolve().parent / "docs"
    docs = load_documents(docs_dir)

    # 默认用 recursive，也可以改为 sentence 或 fixed
    all_chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50, strategy="recursive")
    print(f"\n前 3 个 chunks:")
    for chunk in all_chunks[:3]:
        print(f"  {chunk}")
