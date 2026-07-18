"""
Phase 5 — Global Search（全局搜索 / 社区摘要 Map-Reduce）

职责：回答"没有具体实体"的宏观问题

为什么需要 Global Search？
  有些问题没有明确实体，Local Search 找不到入口：
    "这些文档的核心主题有哪些？"
    "AI 领域的主要技术趋势是什么？"
    "这些技术之间有哪些共同特点？"

  向量 RAG 对这类问题也无能——它只返回 top-k 个局部片段，覆盖不了全局。

做法：Map-Reduce
  1. Map：对每个社区摘要，独立问 LLM"这个社区和用户问题相关吗？"
     → 输出：部分回答 + 相关度评分
  2. Filter：过滤掉相关度低的社区
  3. Reduce：把保留的部分回答汇总，让 LLM 综合成最终答案

代价：
  ⚠️ Global Search 很贵！每次查询对所有社区调一次 LLM（Map 阶段）。
  优化：用粗粒度社区、先 embedding 过滤、异步并发等。
"""

import json
from pathlib import Path
from dataclasses import dataclass

from openai import OpenAI
from config import config
from phase5_community import Community, CommunityDetector


# ========== Map 阶段结果 ==========


@dataclass
class MapResult:
    """Map 阶段每个社区的评估结果"""

    community_id: int
    community_title: str
    relevance_score: float  # 0-100
    partial_answer: str  # 该社区贡献的部分回答


# ========== Map Prompt ==========

_MAP_PROMPT = """根据以下社区摘要，评估它与用户问题的相关性，并生成部分回答。

## 社区摘要

标题：{title}
内容：{summary}

## 用户问题

{query}

## 任务

1. 判断这个社区的内容与用户问题的相关度（0-100 分）
   - 0 分 = 完全无关
   - 50 分 = 有一些关联
   - 100 分 = 高度相关
2. 如果相关度 > 0，根据社区内容生成一段针对用户问题的部分回答
3. 如果完全无关，partial_answer 留空

请以 JSON 格式输出：
{{"relevance_score": 85, "partial_answer": "..."}}

只输出 JSON。"""


# ========== Reduce Prompt ==========

_REDUCE_PROMPT = """你是一个分析师。以下是从不同主题社区收集的部分回答，每个回答代表知识图谱中一个社区对用户问题的贡献。

请综合这些信息，生成一个全面、连贯、有条理的最终回答。

## 用户问题

{query}

## 各社区的部分回答

{partial_answers}

## 要求

1. 综合所有部分回答，不要简单罗列
2. 按主题分类组织答案
3. 如果不同社区有互补信息，整合在一起
4. 保持答案全面但不冗余"""


# ========== Global Search ==========


class GlobalSearch:
    """
    Global Search：基于社区摘要的 Map-Reduce 全局检索

    适用场景：
      - 全局性问题："主要技术趋势是什么？"
      - 聚合性问题："有哪些共同特点？"
      - 概览性问题："总结一下这些文档的内容"

    流程：
      社区摘要 × N → Map（独立评估）→ Filter → Reduce（综合回答）

    使用方式：
        communities = CommunityDetector.load_communities("phase5_communities.json")
        searcher = GlobalSearch(communities)
        answer = searcher.search("这些文档的核心主题有哪些？")
    """

    def __init__(
        self,
        communities: list[Community],
        llm_client: OpenAI | None = None,
        model: str | None = None,
        relevance_threshold: float = 20,  # 相关度阈值
    ):
        """
        Args:
            communities: 社区列表
            llm_client: LLM 客户端
            model: 模型名称
            relevance_threshold: 相关度过滤阈值（0-100），低于此分数的社区被过滤
        """
        self.communities = communities
        self.client = llm_client or OpenAI(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
        )
        self.model = model or config.llm_model
        self.relevance_threshold = relevance_threshold

    # ========== 主流程 ==========

    def search(
        self,
        query: str,
        verbose: bool = True,
    ) -> str:
        """
        完整的 Global Search 流程。

        Args:
            query: 用户问题
            verbose: 是否打印过程

        Returns:
            LLM 综合生成的全局回答
        """
        if verbose:
            print(f"\n🌐 Global Search: {query}")

        # Step 1: Map — 对每个社区独立评估
        map_results = self._map_phase(query, verbose)

        # Step 2: Filter — 过滤低相关度
        filtered = self._filter_phase(map_results, verbose)

        if not filtered:
            if verbose:
                print("  ⚠️  没有社区与问题相关")
            return "根据现有知识图谱的社区结构，没有找到与该问题直接相关的内容。"

        # Step 3: Reduce — 综合回答
        answer = self._reduce_phase(query, filtered, verbose)

        return answer

    # ========== Map 阶段 ==========

    def _map_phase(
        self,
        query: str,
        verbose: bool,
    ) -> list[MapResult]:
        """
        Map 阶段：对每个社区摘要，LLM 独立评估相关度和生成部分回答。

        这是 Global Search 最贵的阶段——每个社区一次 LLM 调用。
        """
        if verbose:
            print(f"  📤 Map 阶段: 评估 {len(self.communities)} 个社区...")

        results = []
        for community in self.communities:
            # 跳过空摘要的社区
            if not community.summary.strip():
                continue

            prompt = _MAP_PROMPT.format(
                title=community.title,
                summary=community.summary,
                query=query,
            )

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                data = json.loads(response.choices[0].message.content)

                result = MapResult(
                    community_id=community.community_id,
                    community_title=community.title,
                    relevance_score=float(data.get("relevance_score", 0)),
                    partial_answer=data.get("partial_answer", ""),
                )
                results.append(result)

                if verbose and result.relevance_score > 0:
                    print(f"     社区 {community.community_id} "
                          f"[{community.title}]: "
                          f"相关度={result.relevance_score}")

            except Exception as e:
                if verbose:
                    print(f"     ⚠️  社区 {community.community_id} 评估失败: {e}")

        return results

    # ========== Filter 阶段 ==========

    def _filter_phase(
        self,
        results: list[MapResult],
        verbose: bool,
    ) -> list[MapResult]:
        """
        Filter 阶段：过滤掉相关度低于阈值的社区。

        排序：按相关度从高到低，优先保留最相关的。
        """
        filtered = [
            r for r in results
            if r.relevance_score >= self.relevance_threshold
        ]
        filtered.sort(key=lambda r: r.relevance_score, reverse=True)

        if verbose:
            print(f"  🔽 Filter: {len(results)} → {len(filtered)} 个社区 "
                  f"(阈值={self.relevance_threshold})")

        return filtered

    # ========== Reduce 阶段 ==========

    def _reduce_phase(
        self,
        query: str,
        filtered_results: list[MapResult],
        verbose: bool,
    ) -> str:
        """
        Reduce 阶段：把所有部分回答综合成最终答案。

        只需要一次 LLM 调用。
        """
        if verbose:
            print(f"  📥 Reduce 阶段: 综合 {len(filtered_results)} 个部分回答...")

        # 拼接部分回答
        partial_answers_text = "\n\n".join(
            f"### [{r.community_title}] (相关度: {r.relevance_score})\n{r.partial_answer}"
            for r in filtered_results
            if r.partial_answer.strip()
        )

        prompt = _REDUCE_PROMPT.format(
            query=query,
            partial_answers=partial_answers_text,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return response.choices[0].message.content


# ========== 独立运行 Demo ==========


def demo():
    """
    Global Search demo：用社区摘要回答全局问题

    前提：先运行过 phase5_community.py 生成了 communities JSON

    用法: uv run python phase5_global_search.py
    """
    print("=" * 60)
    print("Phase 5 — Global Search Demo")
    print("=" * 60)

    # 加载社区
    comm_path = Path(__file__).parent / "phase5_communities.json"
    if not comm_path.exists():
        print("❌ 请先运行 phase5_community.py 生成社区！")
        return

    communities = CommunityDetector.load_communities(comm_path)

    # 创建 Global Search
    searcher = GlobalSearch(communities, relevance_threshold=20)

    # 测试问题（全局问题——Global Search 的强项）
    test_queries = [
        "这些文档涵盖了哪些核心 AI 技术主题？",
        "这些技术之间有哪些共同的设计思路？",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        answer = searcher.search(query)
        print(f"\n📝 回答:\n{answer}")

    print(f"\n{'='*60}")
    print("✅ Demo 完成")


if __name__ == "__main__":
    demo()
