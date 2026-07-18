"""
Phase 5 — 知识图谱构建

职责：把 Step 1 抽取的原始实体/关系合并去重，形成一张干净的知识图谱

为什么需要这一步？
  Step 1 是逐 chunk 独立抽取的，同一个实体会在多个 chunk 中被提到：
    Chunk 1: Entity("TRANSFORMER", "一种神经网络架构")
    Chunk 5: Entity("TRANSFORMER", "Google 提出的注意力模型，2017 年发布")
    Chunk 9: Entity("TRANSFORMER模型", "基础架构")  ← 名称还有细微差异

  合并策略：
    1. 名称归一化：大写 + 去空格 → 匹配到同一个 key
    2. 描述合并：用 LLM 把多段描述综合成一段（比简单拼接好得多）
    3. 关系去重：相同 (source, target) 的关系合并描述 + 取最大强度
    4. source_chunk_ids 取并集（溯源完整性）

数据结构：
  KnowledgeGraph
    ├── entities: dict[str, Entity]     # name → Entity（去重后）
    ├── relationships: list[Relationship] # 去重后的关系列表
    └── save/load JSON 持久化

和向量库的本质区别：
  向量库存的是"文本块 + 向量"，查询靠相似度
  知识图谱存的是"实体节点 + 关系边"，查询靠图遍历
  两者互补，最终会在 Hybrid 中融合
"""

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import defaultdict

from openai import OpenAI
from config import config
from phase5_entity_extractor import Entity, Relationship, ExtractionResult


# ========== 描述合并 Prompt ==========

_MERGE_DESC_PROMPT = """你是一个知识库编辑。下面是对同一个实体"{entity_name}"的多段描述，来自不同的文本片段。

请把它们合并成一段全面、准确、不重复的综合描述。
保留所有有价值的信息，去除重复内容，使描述读起来连贯流畅。

多段描述：
{descriptions}

输出合并后的描述（只输出描述文本，不要加任何前缀或解释）："""


# ========== 知识图谱 ==========


class KnowledgeGraph:
    """
    知识图谱：实体 + 关系的图结构

    核心职责：
      1. 接收 Step 1 的原始抽取结果
      2. 合并去重 → 干净的图
      3. JSON 持久化 → 不用每次重新抽取

    使用方式：
        # 从抽取结果构建
        kg = KnowledgeGraph()
        kg.build_from_extractions(extraction_results)
        kg.save_to_json("phase5_knowledge_graph.json")

        # 从文件加载
        kg = KnowledgeGraph.load_from_json("phase5_knowledge_graph.json")
        print(kg.stats())

    数据结构类比：
        向量库  ≈  一个大箱子，里面装了很多文本片段，检索靠相似度
        知识图谱 ≈  一张关系网，节点是实体，边是关系，检索靠沿边遍历
    """

    def __init__(self):
        self.entities: dict[str, Entity] = {}  # name → Entity
        self.relationships: list[Relationship] = []

        # LLM 客户端（用于描述合并）
        self._llm_client = OpenAI(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
        )
        self._llm_model = config.llm_model

    # ========== 构建 ==========

    def build_from_extractions(
        self,
        results: list[ExtractionResult],
        use_llm_merge: bool = True,
        verbose: bool = True,
    ):
        """
        从 Step 1 的抽取结果构建知识图谱。

        流程：
          1. 收集所有实体 → 按归一化名称分组
          2. 每组合并成一个实体（描述用 LLM 合并或简单拼接）
          3. 收集所有关系 → 按 (source, target) 分组
          4. 每组合并成一条关系

        Args:
            results: EntityExtractor.extract_from_chunks() 的输出
            use_llm_merge: 是否用 LLM 合并多段描述（否则简单拼接）
            verbose: 是否打印进度
        """
        if verbose:
            print("🏗️  开始构建知识图谱...")

        # ---- 第一步：收集和分组实体 ----
        entity_groups: dict[str, list[Entity]] = defaultdict(list)
        for result in results:
            for entity in result.entities:
                entity_groups[entity.name].append(entity)

        if verbose:
            print(f"  📥 收集到 {sum(len(v) for v in entity_groups.values())} 个原始实体 "
                  f"→ {len(entity_groups)} 个唯一名称")

        # ---- 第二步：合并实体 ----
        self.entities = {}
        for name, group in entity_groups.items():
            merged = self._merge_entities(name, group, use_llm_merge, verbose)
            self.entities[name] = merged

        if verbose:
            print(f"  ✅ 合并后: {len(self.entities)} 个实体")

        # ---- 第三步：收集和分组关系 ----
        rel_groups: dict[tuple[str, str], list[Relationship]] = defaultdict(list)
        for result in results:
            for rel in result.relationships:
                # 确保关系两端的实体都存在于图中
                if rel.source in self.entities and rel.target in self.entities:
                    key = (rel.source, rel.target)
                    rel_groups[key].append(rel)
                elif verbose:
                    # 一端不存在——可能是名称归一化问题，尝试修复
                    fixed_source = self._find_closest_entity(rel.source)
                    fixed_target = self._find_closest_entity(rel.target)
                    if fixed_source and fixed_target:
                        rel.source = fixed_source
                        rel.target = fixed_target
                        key = (rel.source, rel.target)
                        rel_groups[key].append(rel)
                    else:
                        print(f"  ⚠️  关系 {rel.source} → {rel.target} 的端点不在实体表中，跳过")

        # ---- 第四步：合并关系 ----
        self.relationships = []
        for (source, target), group in rel_groups.items():
            merged_rel = self._merge_relationships(source, target, group)
            self.relationships.append(merged_rel)

        if verbose:
            print(f"  ✅ 合并后: {len(self.relationships)} 条关系")
            print(f"\n{self.stats()}")

    # ========== 合并逻辑 ==========

    def _merge_entities(
        self,
        name: str,
        group: list[Entity],
        use_llm_merge: bool,
        verbose: bool,
    ) -> Entity:
        """
        合并同名实体组。

        策略：
          - 类型：取出现次数最多的类型
          - 描述：用 LLM 综合，或简单拼接
          - source_chunk_ids：取并集
        """
        # 类型：投票取众数
        type_counts: dict[str, int] = defaultdict(int)
        for e in group:
            type_counts[e.entity_type] += 1
        best_type = max(type_counts, key=type_counts.get)

        # source_chunk_ids：并集
        all_chunk_ids = []
        for e in group:
            all_chunk_ids.extend(e.source_chunk_ids)
        unique_chunk_ids = list(set(all_chunk_ids))

        # 描述：合并
        descriptions = [e.description for e in group if e.description.strip()]
        # 去除完全重复的描述
        unique_descriptions = list(dict.fromkeys(descriptions))

        if len(unique_descriptions) == 0:
            merged_description = ""
        elif len(unique_descriptions) == 1:
            merged_description = unique_descriptions[0]
        elif use_llm_merge:
            # 多段不同的描述 → 用 LLM 合并
            if verbose:
                print(f"    🔀 LLM 合并实体描述: {name} ({len(unique_descriptions)} 段)")
            merged_description = self._llm_merge_descriptions(name, unique_descriptions)
        else:
            # 简单拼接
            merged_description = " | ".join(unique_descriptions)

        return Entity(
            name=name,
            entity_type=best_type,
            description=merged_description,
            source_chunk_ids=unique_chunk_ids,
        )

    def _merge_relationships(
        self,
        source: str,
        target: str,
        group: list[Relationship],
    ) -> Relationship:
        """
        合并同 (source, target) 的关系组。

        策略：
          - 描述：取最长的那条（通常信息最丰富）
          - 强度：取最大值
          - source_chunk_ids：取并集
        """
        # 描述：取最长
        best_desc = max(group, key=lambda r: len(r.description)).description

        # 强度：取最大
        max_strength = max(r.strength for r in group)

        # source_chunk_ids：并集
        all_chunk_ids = []
        for r in group:
            all_chunk_ids.extend(r.source_chunk_ids)
        unique_chunk_ids = list(set(all_chunk_ids))

        return Relationship(
            source=source,
            target=target,
            description=best_desc,
            strength=max_strength,
            source_chunk_ids=unique_chunk_ids,
        )

    def _llm_merge_descriptions(self, entity_name: str, descriptions: list[str]) -> str:
        """用 LLM 合并多段实体描述"""
        desc_text = "\n".join(f"  - {d}" for d in descriptions)
        prompt = _MERGE_DESC_PROMPT.format(
            entity_name=entity_name,
            descriptions=desc_text,
        )

        try:
            response = self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"    ⚠️  LLM 合并失败 ({e})，退回简单拼接")
            return " | ".join(descriptions)

    def _find_closest_entity(self, name: str) -> str | None:
        """
        尝试模糊匹配实体名称。

        处理常见的归一化差异：
          "TRANSFORMER模型" → 包含 "TRANSFORMER" → 匹配到 "TRANSFORMER"
          "SELF-ATTENTION" → 包含关系匹配
        """
        # 精确匹配
        if name in self.entities:
            return name

        # 包含匹配：A 包含 B 或 B 包含 A
        for existing_name in self.entities:
            if name in existing_name or existing_name in name:
                return existing_name

        return None

    # ========== 查询接口 ==========

    def get_entity(self, name: str) -> Entity | None:
        """按名称获取实体"""
        normalized = " ".join(name.upper().split())
        return self.entities.get(normalized)

    def get_neighbors(self, entity_name: str, hops: int = 1) -> dict:
        """
        获取实体的邻居（1 跳或 2 跳）

        返回:
            {"entities": [Entity, ...], "relationships": [Relationship, ...]}

        这是 Local Search 的核心操作：从一个实体出发，沿关系边走 N 跳，
        收集路径上的所有实体和关系。
        """
        visited_entities = set()
        collected_rels = []
        frontier = {entity_name}

        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                for rel in self.relationships:
                    if rel.source == node and rel.target not in visited_entities:
                        next_frontier.add(rel.target)
                        collected_rels.append(rel)
                    elif rel.target == node and rel.source not in visited_entities:
                        next_frontier.add(rel.source)
                        collected_rels.append(rel)
            visited_entities.update(frontier)
            frontier = next_frontier

        visited_entities.update(frontier)
        # 去掉起始节点自身
        visited_entities.discard(entity_name)

        neighbor_entities = [
            self.entities[name]
            for name in visited_entities
            if name in self.entities
        ]

        return {
            "entities": neighbor_entities,
            "relationships": collected_rels,
        }

    def get_entity_relationships(self, entity_name: str) -> list[Relationship]:
        """获取与某个实体相关的所有关系"""
        return [
            rel for rel in self.relationships
            if rel.source == entity_name or rel.target == entity_name
        ]

    # ========== 统计 ==========

    def stats(self) -> str:
        """返回图的基础统计信息"""
        # 度分布
        degree: dict[str, int] = defaultdict(int)
        for rel in self.relationships:
            degree[rel.source] += 1
            degree[rel.target] += 1

        degrees = list(degree.values()) if degree else [0]

        # 实体类型分布
        type_counts: dict[str, int] = defaultdict(int)
        for entity in self.entities.values():
            type_counts[entity.entity_type] += 1

        lines = [
            "📊 知识图谱统计:",
            f"  节点数: {len(self.entities)}",
            f"  边数:   {len(self.relationships)}",
            f"  平均度: {sum(degrees)/len(degrees):.1f}",
            f"  最大度: {max(degrees)}",
            f"  实体类型分布:",
        ]
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            lines.append(f"    {t}: {c}")

        return "\n".join(lines)

    # ========== 持久化 ==========

    def save_to_json(self, filepath: str | Path):
        """
        保存知识图谱到 JSON 文件。

        文件结构：
          {
            "entities": { "NAME": {name, entity_type, description, source_chunk_ids}, ... },
            "relationships": [ {source, target, description, strength, source_chunk_ids}, ... ]
          }
        """
        filepath = Path(filepath)
        data = {
            "entities": {
                name: asdict(entity)
                for name, entity in self.entities.items()
            },
            "relationships": [
                asdict(rel) for rel in self.relationships
            ],
        }
        filepath.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"💾 知识图谱已保存到 {filepath} ({filepath.stat().st_size / 1024:.1f} KB)")

    @classmethod
    def load_from_json(cls, filepath: str | Path) -> "KnowledgeGraph":
        """
        从 JSON 文件加载知识图谱。

        这样就不需要每次都重新做 LLM 抽取 + 合并了——
        和 Phase 1 的 simple_index.json 是同一个思路。
        """
        filepath = Path(filepath)
        data = json.loads(filepath.read_text(encoding="utf-8"))

        kg = cls()

        # 加载实体
        for name, entity_data in data["entities"].items():
            kg.entities[name] = Entity(**entity_data)

        # 加载关系
        for rel_data in data["relationships"]:
            kg.relationships.append(Relationship(**rel_data))

        print(f"📂 知识图谱已从 {filepath} 加载: "
              f"{len(kg.entities)} 个实体, {len(kg.relationships)} 条关系")

        return kg


# ========== 独立运行 Demo ==========


def demo():
    """
    独立运行 demo：做一次完整的 抽取 → 建图 流程

    只用 docs/ 下的 .md 文件（3 篇技术文档，约 9 个 chunk）。
    学习阶段不需要处理 PDF/docx 大文档——那是生产环境的事。

    用法: uv run python phase5_knowledge_graph.py
    """
    from pathlib import Path
    from phase1_document_loader import Document
    from phase1_chunker import chunk_documents
    from phase5_entity_extractor import EntityExtractor

    print("=" * 60)
    print("Phase 5 — 知识图谱构建 Demo")
    print("=" * 60)

    # Step 1: 直接读取 .md 文件（绕过 load_documents 避免加载大 PDF）
    print("\n📚 加载文档（仅 .md 技术文档）...")
    docs_dir = Path(__file__).parent / "docs"
    md_files = sorted(docs_dir.glob("*.md"))
    print(f"  找到 {len(md_files)} 个 .md 文件: {[f.name for f in md_files]}")

    md_docs = []
    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        md_docs.append(Document(content=text, metadata={"source": md_file.name}))
        print(f"  ✅ {md_file.name} ({len(text)} 字符)")

    # 用稍大的 chunk_size，给 LLM 更多上下文做实体抽取
    chunks = chunk_documents(md_docs, chunk_size=1200, chunk_overlap=100)
    print(f"  → {len(chunks)} 个 chunk")


    # Step 2: 实体/关系抽取
    print("\n🔍 开始实体/关系抽取...")
    extractor = EntityExtractor(gleaning_rounds=1)
    results = extractor.extract_from_chunks(chunks)

    # Step 3: 构建知识图谱
    print("\n🏗️  构建知识图谱（合并去重）...")
    kg = KnowledgeGraph()
    kg.build_from_extractions(results, use_llm_merge=True)

    # 保存
    save_path = Path(__file__).parent / "phase5_knowledge_graph.json"
    kg.save_to_json(save_path)

    # 打印一些实体详情
    print(f"\n{'='*60}")
    print("📦 部分实体详情:")
    for i, (name, entity) in enumerate(list(kg.entities.items())[:5]):
        print(f"\n  🔵 [{entity.entity_type}] {entity.name}")
        print(f"     描述: {entity.description[:100]}...")
        print(f"     来源: {entity.source_chunk_ids}")

    # 打印邻居查询示例
    if kg.entities:
        sample_entity = list(kg.entities.keys())[0]
        neighbors = kg.get_neighbors(sample_entity, hops=1)
        print(f"\n{'='*60}")
        print(f"🔗 实体 [{sample_entity}] 的 1 跳邻居:")
        for e in neighbors["entities"]:
            print(f"  → {e.name} ({e.entity_type})")
        for r in neighbors["relationships"]:
            print(f"  边: {r.source} —[{r.description[:30]}]→ {r.target}")

    print(f"\n✅ Demo 完成")


if __name__ == "__main__":
    demo()
