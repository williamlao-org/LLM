"""
Phase 5 — Local Search（局部搜索 / 实体中心检索）

职责：回答"有明确实体"的具体问题

工作原理（类比查字典）：
  1. 用户问："Transformer 的核心创新是什么？"
  2. 从问题中识别出实体 → "TRANSFORMER"
  3. 在知识图谱中找到 TRANSFORMER 节点
  4. 沿关系边走 1-2 跳，收集相关实体和关系
     → TRANSFORMER —[核心创新]→ SELF-ATTENTION
     → TRANSFORMER —[基础架构]→ GPT, BERT, LLAMA
     → TRANSFORMER 所在社区的摘要
  5. 把收集到的信息组装成上下文，交给 LLM 生成回答

和向量 RAG 的对比：
  向量 RAG：搜 "Transformer 核心创新" → 找语义最像的 chunk → 碰运气
  Local Search：直接走图 → 精准找到关系链 → 结构化信息

Local Search 擅长：
  ✅ "A 和 B 有什么关系？"
  ✅ "X 用了什么技术？"
  ✅ "谁发明了 Y？"

Local Search 不擅长：
  ❌ "这些文档的主要趋势是什么？"（没有具体实体 → 用 Global Search）
"""

import json
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from config import config
from phase5_entity_extractor import Entity, Relationship
from phase5_knowledge_graph import KnowledgeGraph
from phase5_community import Community, CommunityDetector


# ========== 上下文组装结果 ==========


@dataclass
class LocalSearchContext:
    """Local Search 组装出的上下文"""

    matched_entities: list[Entity]  # 匹配到的实体
    neighbor_entities: list[Entity]  # 邻居实体
    relationships: list[Relationship]  # 相关关系
    community_summaries: list[str]  # 相关社区摘要
    formatted_context: str  # 最终拼好的上下文文本


# ========== 实体匹配 Prompt ==========

_ENTITY_EXTRACT_PROMPT = """从以下用户问题中提取关键实体名称（人名、组织、技术、模型、概念等）。

用户问题：{query}

请以 JSON 格式输出实体名称列表：
{{"entities": ["实体1", "实体2", ...]}}

如果问题中没有明确实体，返回空列表：{{"entities": []}}
只输出 JSON。"""


# ========== 生成回答 Prompt ==========

_LOCAL_ANSWER_PROMPT = """你是一个知识库问答助手。请基于以下从知识图谱中检索到的结构化信息来回答用户的问题。

## 知识图谱上下文

{context}

---

用户问题：{query}

请基于以上信息回答问题。如果信息不足，请明确说明。"""


# ========== Local Search ==========


class LocalSearch:
    """
    Local Search：从实体出发的图检索

    流程：
      问题 → 提取实体 → 图中匹配 → 沿边扩展 → 组装上下文 → LLM 回答

    使用方式：
        kg = KnowledgeGraph.load_from_json("phase5_knowledge_graph.json")
        communities = CommunityDetector.load_communities("phase5_communities.json")
        searcher = LocalSearch(kg, communities)
        answer = searcher.search("Transformer 的核心创新是什么？")
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        communities: list[Community],
        llm_client: OpenAI | None = None,
        model: str | None = None,
        hops: int = 1,
        max_context_tokens: int = 3000,
    ):
        """
        Args:
            kg: 知识图谱
            communities: 社区列表
            llm_client: LLM 客户端
            model: 模型名称
            hops: 沿关系边走几跳（1 或 2）
            max_context_tokens: 上下文 token 预算（粗略估计，1 中文字 ≈ 1.5 token）
        """
        self.kg = kg
        self.communities = communities
        self.client = llm_client or OpenAI(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
        )
        self.model = model or config.llm_model
        self.hops = hops
        self.max_context_tokens = max_context_tokens

        # 预建：实体名 → 所属社区 的映射
        self._entity_to_community: dict[str, Community] = {}
        for community in communities:
            for name in community.entity_names:
                self._entity_to_community[name] = community

    # ========== 主流程 ==========

    def search(
        self,
        query: str,
        verbose: bool = True,
    ) -> str:
        """
        完整的 Local Search 流程：问题 → 回答

        Args:
            query: 用户问题
            verbose: 是否打印过程

        Returns:
            LLM 生成的回答
        """
        if verbose:
            print(f"\n🔍 Local Search: {query}")

        # Step 1: 提取实体
        query_entities = self._extract_query_entities(query, verbose)

        if not query_entities:
            if verbose:
                print("  ⚠️  未从问题中识别到实体，Local Search 无法进行")
            return "无法从问题中识别到明确的实体，建议使用向量检索或 Global Search。"

        # Step 2: 匹配实体
        matched = self._match_entities(query_entities, verbose)

        if not matched:
            if verbose:
                print("  ⚠️  图中未找到匹配的实体")
            return "知识图谱中未找到与问题相关的实体。"

        # Step 3: 收集上下文
        context = self._collect_context(matched, verbose)

        # Step 4: LLM 生成回答
        answer = self._generate_answer(query, context, verbose)

        return answer

    def search_context_only(
        self,
        query: str,
        verbose: bool = True,
    ) -> LocalSearchContext:
        """
        只做检索，不生成回答（方便调试和对比）

        Returns:
            LocalSearchContext 包含所有收集到的信息
        """
        query_entities = self._extract_query_entities(query, verbose)
        matched = self._match_entities(query_entities, verbose)
        return self._collect_context(matched, verbose)

    # ========== Step 1: 从问题中提取实体 ==========

    def _extract_query_entities(
        self,
        query: str,
        verbose: bool,
    ) -> list[str]:
        """
        从用户问题中提取实体名称。

        两种策略：
          1. LLM 提取（更准确，但多一次 API 调用）
          2. 简单字符串匹配（免费，但只能匹配已知实体名）

        这里用 LLM 提取 + 图中已知实体名匹配 的混合策略。
        """
        if verbose:
            print("  ① 提取问题中的实体...")

        # 策略 A: LLM 提取
        prompt = _ENTITY_EXTRACT_PROMPT.format(query=query)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            llm_entities = result.get("entities", [])
        except Exception:
            llm_entities = []

        # 策略 B: 直接匹配图中已知的实体名
        query_upper = query.upper()
        graph_matches = [
            name for name in self.kg.entities
            if name in query_upper
        ]

        # 合并两种策略的结果
        all_entities = list(set(
            [e.upper().strip() for e in llm_entities] + graph_matches
        ))

        if verbose:
            print(f"     LLM 提取: {llm_entities}")
            print(f"     图匹配: {graph_matches}")
            print(f"     合并: {all_entities}")

        return all_entities

    # ========== Step 2: 在图中匹配实体 ==========

    def _match_entities(
        self,
        query_entities: list[str],
        verbose: bool,
    ) -> list[Entity]:
        """
        把提取的实体名称匹配到图中的节点。

        匹配策略（按优先级）：
          1. 精确匹配：名称完全一致
          2. 包含匹配：一个名称包含另一个
          3. 如果都匹配不到 → 放弃（可以扩展为 embedding 匹配）
        """
        if verbose:
            print("  ② 在图中匹配实体...")

        matched = []
        for qe in query_entities:
            normalized = " ".join(qe.upper().split())

            # 精确匹配
            if normalized in self.kg.entities:
                matched.append(self.kg.entities[normalized])
                if verbose:
                    print(f"     ✅ 精确匹配: {normalized}")
                continue

            # 包含匹配
            found = False
            for name, entity in self.kg.entities.items():
                if normalized in name or name in normalized:
                    matched.append(entity)
                    if verbose:
                        print(f"     🔄 包含匹配: {normalized} → {name}")
                    found = True
                    break

            if not found and verbose:
                print(f"     ❌ 未匹配: {normalized}")

        return matched

    # ========== Step 3: 收集上下文 ==========

    def _collect_context(
        self,
        matched_entities: list[Entity],
        verbose: bool,
    ) -> LocalSearchContext:
        """
        从匹配的实体出发，沿关系边收集上下文。

        上下文优先级（从高到低）：
          1. 匹配实体自身的描述（最直接相关）
          2. 直接关系的描述（一跳关系）
          3. 匹配实体所在社区的摘要（提供宏观背景）
          4. 邻居实体的描述（补充信息）

        总 token 超过预算时从低优先级开始截断。
        """
        if verbose:
            print("  ③ 收集图上下文...")

        # 收集邻居实体和关系
        all_neighbor_entities = []
        all_relationships = []

        for entity in matched_entities:
            neighbors = self.kg.get_neighbors(entity.name, hops=self.hops)
            all_neighbor_entities.extend(neighbors["entities"])
            all_relationships.extend(neighbors["relationships"])

        # 去重
        seen_entities = set()
        unique_neighbors = []
        for e in all_neighbor_entities:
            if e.name not in seen_entities and e.name not in {m.name for m in matched_entities}:
                seen_entities.add(e.name)
                unique_neighbors.append(e)

        seen_rels = set()
        unique_rels = []
        for r in all_relationships:
            key = (r.source, r.target)
            if key not in seen_rels:
                seen_rels.add(key)
                unique_rels.append(r)

        # 按关系强度排序
        unique_rels.sort(key=lambda r: r.strength, reverse=True)

        # 收集相关社区摘要
        community_summaries = []
        seen_communities = set()
        for entity in matched_entities:
            community = self._entity_to_community.get(entity.name)
            if community and community.community_id not in seen_communities:
                seen_communities.add(community.community_id)
                community_summaries.append(
                    f"[{community.title}] {community.summary}"
                )

        if verbose:
            print(f"     匹配实体: {len(matched_entities)}")
            print(f"     邻居实体: {len(unique_neighbors)}")
            print(f"     关系: {len(unique_rels)}")
            print(f"     社区摘要: {len(community_summaries)}")

        # 组装上下文文本（按优先级）
        formatted = self._format_context(
            matched_entities, unique_neighbors, unique_rels, community_summaries
        )

        return LocalSearchContext(
            matched_entities=matched_entities,
            neighbor_entities=unique_neighbors,
            relationships=unique_rels,
            community_summaries=community_summaries,
            formatted_context=formatted,
        )

    def _format_context(
        self,
        matched: list[Entity],
        neighbors: list[Entity],
        relationships: list[Relationship],
        community_summaries: list[str],
    ) -> str:
        """
        把收集到的信息格式化为文本上下文。

        格式设计要点：
          - 结构化展示，让 LLM 容易理解
          - 优先级从高到低排列
          - 超预算时截断低优先级部分
        """
        parts = []
        char_budget = self.max_context_tokens  # 粗略用字符数估计

        # 优先级 1：匹配实体描述
        entity_section = "### 核心实体\n"
        for e in matched:
            entity_section += f"- **{e.name}** ({e.entity_type}): {e.description}\n"
        parts.append(entity_section)
        char_budget -= len(entity_section)

        # 优先级 2：关系描述
        if relationships and char_budget > 0:
            rel_section = "\n### 关系\n"
            for r in relationships:
                line = f"- {r.source} → {r.target}: {r.description} (强度: {r.strength})\n"
                if char_budget - len(line) < 0:
                    break
                rel_section += line
                char_budget -= len(line)
            parts.append(rel_section)

        # 优先级 3：社区摘要
        if community_summaries and char_budget > 0:
            comm_section = "\n### 所属社区背景\n"
            for summary in community_summaries:
                if char_budget - len(summary) < 0:
                    break
                comm_section += f"- {summary}\n"
                char_budget -= len(summary)
            parts.append(comm_section)

        # 优先级 4：邻居实体
        if neighbors and char_budget > 0:
            neighbor_section = "\n### 相关实体\n"
            for e in neighbors[:10]:  # 最多展示 10 个邻居
                line = f"- {e.name} ({e.entity_type}): {e.description[:100]}\n"
                if char_budget - len(line) < 0:
                    break
                neighbor_section += line
                char_budget -= len(line)
            parts.append(neighbor_section)

        return "".join(parts)

    # ========== Step 4: LLM 生成回答 ==========

    def _generate_answer(
        self,
        query: str,
        context: LocalSearchContext,
        verbose: bool,
    ) -> str:
        """基于图上下文让 LLM 生成回答"""
        if verbose:
            print("  ④ LLM 生成回答...")

        prompt = _LOCAL_ANSWER_PROMPT.format(
            context=context.formatted_context,
            query=query,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        answer = response.choices[0].message.content
        return answer


# ========== 独立运行 Demo ==========


def demo():
    """
    Local Search demo：用图检索回答关系类问题

    前提：先运行过 phase5_knowledge_graph.py 和 phase5_community.py

    用法: uv run python phase5_local_search.py
    """
    print("=" * 60)
    print("Phase 5 — Local Search Demo")
    print("=" * 60)

    # 加载知识图谱和社区
    kg_path = Path(__file__).parent / "phase5_knowledge_graph.json"
    comm_path = Path(__file__).parent / "phase5_communities.json"

    if not kg_path.exists() or not comm_path.exists():
        print("❌ 请先运行 phase5_knowledge_graph.py 和 phase5_community.py！")
        return

    kg = KnowledgeGraph.load_from_json(kg_path)
    communities = CommunityDetector.load_communities(comm_path)

    # 创建 Local Search
    searcher = LocalSearch(kg, communities, hops=1)

    # 测试问题（关系类问题——Local Search 的强项）
    test_queries = [
        "Transformer 的核心创新是什么？",
        "ReAct 和 Agent 有什么关系？",
        "RAG 解决了 LLM 的什么问题？",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        answer = searcher.search(query)
        print(f"\n📝 回答:\n{answer}")

    print(f"\n{'='*60}")
    print("✅ Demo 完成")


if __name__ == "__main__":
    demo()
