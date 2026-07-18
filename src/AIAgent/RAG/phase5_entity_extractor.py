"""
Phase 5 — 实体/关系抽取器

职责：用 LLM 从文本 chunk 中自动提取实体（Entity）和关系（Relationship）

这是 GraphRAG 和传统知识图谱最大的区别：
  传统方式：NER（命名实体识别）+ 规则模板 → 费力、覆盖率低
  GraphRAG：直接让 LLM 一步到位抽取 → 质量高、灵活

核心流程：
  1. 给 LLM 一段文本 + 实体类型列表
  2. LLM 返回 JSON：实体列表 + 关系列表
  3. Gleaning（二次收割）：再问一次 LLM 补漏

为什么需要 Gleaning？
  LLM 一次抽取经常遗漏实体（尤其是 chunk 末尾的、隐含的）。
  Gleaning 就是抽取完后再追问一次"你是不是漏了什么？"，通常能多捞出 10-20% 的实体。
  做 1-2 轮 gleaning 就够了，再多收益递减。

本模块产出的原始实体/关系还没去重合并——那是下一步 KnowledgeGraph 的职责。
"""

import json
from dataclasses import dataclass, field

from openai import OpenAI
from config import config
from phase1_chunker import Chunk


# ========== 数据模型 ==========


@dataclass
class Entity:
    """
    知识图谱中的一个实体（节点）

    对应现实世界中的"东西"：人、组织、技术、概念、模型、论文...

    示例：
      Entity(name="TRANSFORMER", entity_type="TECHNOLOGY",
             description="Google 在 2017 年提出的神经网络架构，完全基于注意力机制",
             source_chunk_ids=["chunk_0", "chunk_1"])
    """

    name: str  # 实体名称（归一化后：大写、去多余空格）
    entity_type: str  # 实体类型：PERSON / ORGANIZATION / TECHNOLOGY / MODEL / CONCEPT ...
    description: str  # LLM 生成的实体描述
    source_chunk_ids: list[str] = field(default_factory=list)  # 来自哪些 chunk（溯源用）


@dataclass
class Relationship:
    """
    知识图谱中的一条关系（边）

    连接两个实体，描述它们之间的联系。

    示例：
      Relationship(source="GOOGLE", target="TRANSFORMER",
                   description="Google 在 2017 年提出了 Transformer 架构",
                   strength=9, source_chunk_ids=["chunk_0"])
    """

    source: str  # 源实体名称（归一化后）
    target: str  # 目标实体名称（归一化后）
    description: str  # 关系描述
    strength: float  # 关系强度 1-10
    source_chunk_ids: list[str] = field(default_factory=list)


# ========== 抽取结果容器 ==========


@dataclass
class ExtractionResult:
    """一个 chunk 的抽取结果"""

    chunk_id: str
    entities: list[Entity]
    relationships: list[Relationship]


# ========== 抽取 Prompt ==========

# 实体类型列表——根据你的文档领域调整
# 这里针对 AI/技术文档设计
DEFAULT_ENTITY_TYPES = [
    "PERSON",  # 人物（研究者、CEO 等）
    "ORGANIZATION",  # 组织/公司（Google、OpenAI、DeepSeek 等）
    "TECHNOLOGY",  # 技术/方法（Transformer、Attention、RAG 等）
    "MODEL",  # 具体模型（GPT-4、BERT、LLaMA 等）
    "CONCEPT",  # 抽象概念（幻觉、知识蒸馏、梯度消失等）
    "PAPER",  # 论文
    "DATASET",  # 数据集
    "TOOL",  # 工具/框架（LangChain、PyTorch 等）
]

_EXTRACT_SYSTEM_PROMPT = """你是一个专业的知识图谱构建助手。你的任务是从文本中提取实体和关系。

## 任务说明

给定一段文本和一个实体类型列表，你需要：

1. **识别所有实体**。对于每个实体，提取：
   - entity_name: 实体名称（使用原文中最完整的称呼）
   - entity_type: 实体类型，必须从给定列表中选择
   - entity_description: 对该实体的全面描述（基于文本中的信息）

2. **识别所有关系**。从已识别的实体中，找出所有明确相关的实体对，提取：
   - source_entity: 源实体名称
   - target_entity: 目标实体名称
   - relationship_description: 解释为什么这两个实体相关
   - relationship_strength: 1-10 的整数，表示关系的紧密程度

## 重要规则

- 实体名称使用原文中的表述，不要翻译或缩写
- 只提取文本中**明确提到或可以直接推断**的实体和关系
- 不要凭空编造文本中没有的实体或关系
- 一段文本中可能有 0 个、1 个或多个实体/关系，都是正常的
- 关系是有向的：source → target，选择逻辑上自然的方向

## 输出格式

严格输出 JSON，不要加任何其他内容：
{
  "entities": [
    {"entity_name": "...", "entity_type": "...", "entity_description": "..."}
  ],
  "relationships": [
    {"source_entity": "...", "target_entity": "...", "relationship_description": "...", "relationship_strength": 8}
  ]
}

如果文本中没有可提取的实体，返回空列表：
{"entities": [], "relationships": []}
"""

_EXTRACT_USER_PROMPT = """实体类型列表：{entity_types}

文本：
---
{text}
---

请从以上文本中提取所有实体和关系，以 JSON 格式输出。"""


# ========== Gleaning Prompt ==========

_GLEANING_USER_PROMPT = """上一轮你从同一段文本中提取了以下实体和关系：

{previous_extraction}

请仔细重新检查原文，看是否有遗漏的实体或关系。特别注意：
- 文本末尾提到的实体
- 隐含但可推断的关系
- 作为修饰语或定语出现的实体（如"基于 XXX 的方法"中的 XXX）

如果发现遗漏，请**只输出新增的**实体和关系（不要重复上一轮已提取的）。
如果没有遗漏，返回：{{"entities": [], "relationships": []}}

原文：
---
{text}
---

实体类型列表：{entity_types}

以 JSON 格式输出新增的实体和关系。"""


# ========== 抽取器 ==========


class EntityExtractor:
    """
    LLM 实体/关系抽取器

    从文本 chunk 中提取知识图谱的原材料——实体和关系。

    使用方式：
        extractor = EntityExtractor()
        results = extractor.extract_from_chunks(chunks)
        # results: list[ExtractionResult]

    特点：
        - 使用 LLM (DeepSeek) 做抽取，不依赖 NER 工具
        - 支持 Gleaning（二次收割），提高抽取覆盖率
        - 实体名称自动归一化（大写 + 去多余空格）
        - 每个实体/关系记录来源 chunk_id，方便溯源
    """

    def __init__(
        self,
        llm_client: OpenAI | None = None,
        model: str | None = None,
        entity_types: list[str] | None = None,
        gleaning_rounds: int = 1,
    ):
        """
        Args:
            llm_client: OpenAI 兼容客户端，默认用 config 中的 DeepSeek
            model: 模型名称
            entity_types: 要抽取的实体类型列表
            gleaning_rounds: Gleaning 轮数（0=不做，1=推荐，2=充分）
        """
        self.client = llm_client or OpenAI(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
        )
        self.model = model or config.llm_model
        self.entity_types = entity_types or DEFAULT_ENTITY_TYPES
        self.gleaning_rounds = gleaning_rounds

    # ---------- 公开方法 ----------

    def extract_from_chunks(
        self,
        chunks: list[Chunk],
        verbose: bool = True,
    ) -> list[ExtractionResult]:
        """
        批量从 chunk 列表中抽取实体和关系。

        Args:
            chunks: Phase 1 分块器产出的 Chunk 列表
            verbose: 是否打印进度

        Returns:
            每个 chunk 对应一个 ExtractionResult
        """
        results = []
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            chunk_id = chunk.metadata.get("chunk_id", f"chunk_{i}")

            if verbose:
                print(f"\n{'='*60}")
                print(f"[{i+1}/{total}] 正在抽取 chunk: {chunk_id}")
                print(f"  文本预览: {chunk.content[:80]}...")

            result = self._extract_single_chunk(chunk.content, chunk_id, verbose)
            results.append(result)

            if verbose:
                print(f"  ✅ 抽取到 {len(result.entities)} 个实体, "
                      f"{len(result.relationships)} 条关系")

        if verbose:
            total_entities = sum(len(r.entities) for r in results)
            total_rels = sum(len(r.relationships) for r in results)
            print(f"\n{'='*60}")
            print(f"📊 抽取完成: {total} 个 chunk → "
                  f"{total_entities} 个实体, {total_rels} 条关系")

        return results

    def extract_from_text(
        self,
        text: str,
        chunk_id: str = "manual",
        verbose: bool = True,
    ) -> ExtractionResult:
        """
        从单段文本中抽取实体和关系（方便调试用）。

        Args:
            text: 要抽取的文本
            chunk_id: 自定义 chunk ID
            verbose: 是否打印过程

        Returns:
            ExtractionResult
        """
        return self._extract_single_chunk(text, chunk_id, verbose)

    # ---------- 内部方法 ----------

    def _extract_single_chunk(
        self,
        text: str,
        chunk_id: str,
        verbose: bool,
    ) -> ExtractionResult:
        """对单个 chunk 做抽取 + gleaning"""

        # ---- 第一轮：主抽取 ----
        entity_types_str = ", ".join(self.entity_types)
        user_prompt = _EXTRACT_USER_PROMPT.format(
            entity_types=entity_types_str,
            text=text,
        )

        if verbose:
            print("  🔍 第 1 轮抽取...")

        raw = self._call_llm(_EXTRACT_SYSTEM_PROMPT, user_prompt)
        entities, relationships = self._parse_extraction(raw, chunk_id)

        if verbose:
            print(f"     → {len(entities)} 个实体, {len(relationships)} 条关系")

        # ---- Gleaning：二次收割 ----
        for round_num in range(self.gleaning_rounds):
            if verbose:
                print(f"  🔍 Gleaning 第 {round_num + 1} 轮...")

            # 把上一轮的结果告诉 LLM，让它检查遗漏
            previous = json.dumps(
                {
                    "entities": [
                        {"entity_name": e.name, "entity_type": e.entity_type,
                         "entity_description": e.description}
                        for e in entities
                    ],
                    "relationships": [
                        {"source_entity": r.source, "target_entity": r.target,
                         "relationship_description": r.description,
                         "relationship_strength": r.strength}
                        for r in relationships
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )

            gleaning_prompt = _GLEANING_USER_PROMPT.format(
                previous_extraction=previous,
                text=text,
                entity_types=entity_types_str,
            )

            raw_new = self._call_llm(_EXTRACT_SYSTEM_PROMPT, gleaning_prompt)
            new_entities, new_relationships = self._parse_extraction(raw_new, chunk_id)

            if verbose:
                print(f"     → 新增 {len(new_entities)} 个实体, "
                      f"{len(new_relationships)} 条关系")

            # 合并新增的
            entities.extend(new_entities)
            relationships.extend(new_relationships)

        return ExtractionResult(
            chunk_id=chunk_id,
            entities=entities,
            relationships=relationships,
        )

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """调用 LLM，返回原始文本响应"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # 低温度保证抽取的确定性
            response_format={"type": "json_object"},  # 强制 JSON 输出
        )
        return response.choices[0].message.content

    def _parse_extraction(
        self,
        raw_json: str,
        chunk_id: str,
    ) -> tuple[list[Entity], list[Relationship]]:
        """
        解析 LLM 返回的 JSON，转为 Entity 和 Relationship 对象。

        健壮性处理：
          - JSON 解析失败 → 返回空列表
          - 字段缺失 → 跳过该条目
          - 实体名称归一化：大写 + 去多余空格
        """
        entities = []
        relationships = []

        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            print(f"  ⚠️  JSON 解析失败，跳过。原始输出: {raw_json[:200]}...")
            return entities, relationships

        # 解析实体
        for item in data.get("entities", []):
            name = item.get("entity_name", "").strip()
            if not name:
                continue  # 跳过空名称

            entity = Entity(
                name=self._normalize_name(name),
                entity_type=item.get("entity_type", "UNKNOWN").upper(),
                description=item.get("entity_description", ""),
                source_chunk_ids=[chunk_id],
            )
            entities.append(entity)

        # 解析关系
        for item in data.get("relationships", []):
            source = item.get("source_entity", "").strip()
            target = item.get("target_entity", "").strip()
            if not source or not target:
                continue  # 跳过无效关系

            # strength 做个安全转换
            try:
                strength = float(item.get("relationship_strength", 5))
                strength = max(1.0, min(10.0, strength))  # 钳位到 1-10
            except (ValueError, TypeError):
                strength = 5.0

            rel = Relationship(
                source=self._normalize_name(source),
                target=self._normalize_name(target),
                description=item.get("relationship_description", ""),
                strength=strength,
                source_chunk_ids=[chunk_id],
            )
            relationships.append(rel)

        return entities, relationships

    @staticmethod
    def _normalize_name(name: str) -> str:
        """
        实体名称归一化

        规则：
          1. 转大写（"Transformer" → "TRANSFORMER"）
          2. 去除首尾空白
          3. 多个空格合并为一个（"Deep  Seek" → "DEEP SEEK"）

        为什么要归一化？
          不同 chunk 对同一个实体的称呼可能略有不同：
          "Transformer" / "transformer" / " Transformer " → 统一成 "TRANSFORMER"
          这样下一步去重合并才能正确匹配。
        """
        return " ".join(name.upper().split())


# ========== 独立运行 Demo ==========


def demo():
    """
    独立运行 demo：对一段示例文本做实体/关系抽取

    用法: python phase5_entity_extractor.py
    """
    print("=" * 60)
    print("Phase 5 — 实体/关系抽取器 Demo")
    print("=" * 60)

    # 用一段 docs/ 中的真实文本来测试
    test_text = """
    Transformer 是 Google 在 2017 年论文《Attention Is All You Need》中提出的一种神经网络架构。
    它彻底抛弃了传统的循环神经网络（RNN）和卷积神经网络（CNN），完全基于注意力机制（Attention Mechanism）来处理序列数据。
    Transformer 的出现是自然语言处理领域的一个里程碑事件。
    它不仅在机器翻译任务上取得了当时的最佳成绩，还成为了后续几乎所有大语言模型（如 GPT、BERT、LLaMA 等）的基础架构。
    自注意力机制是 Transformer 最核心的创新。它允许序列中的每个位置都能直接"关注"到序列中其他所有位置的信息。
    """

    extractor = EntityExtractor(gleaning_rounds=1)

    print("\n📝 输入文本:")
    print(f"  {test_text.strip()[:100]}...")

    result = extractor.extract_from_text(test_text, chunk_id="demo_chunk")

    print(f"\n{'='*60}")
    print("📦 抽取结果:")
    print(f"{'='*60}")

    print("\n🔵 实体:")
    for e in result.entities:
        print(f"  [{e.entity_type}] {e.name}")
        print(f"    描述: {e.description[:80]}...")

    print("\n🔗 关系:")
    for r in result.relationships:
        print(f"  {r.source} —[{r.description[:40]}]→ {r.target}  (强度: {r.strength})")

    print(f"\n✅ Demo 完成")


if __name__ == "__main__":
    demo()
