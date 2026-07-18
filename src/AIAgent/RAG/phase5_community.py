"""
Phase 5 — 社区检测 + 社区摘要

职责：
  1. 把知识图谱中"关系紧密的实体"自动分组成"社区"
  2. 让 LLM 为每个社区写一段摘要

为什么要做社区检测？
  一张大图里有些实体彼此联系很紧密，形成"小圈子"。
  比如 Transformer、Self-Attention、Multi-Head Attention 自然成一组，
  RAG、Vector Store、Embedding 自然成另一组。

  社区检测就是自动找出这些小圈子，好处是：
    - Local Search：知道实体属于哪个社区，可以附带社区摘要提供背景
    - Global Search：直接用社区摘要做 Map-Reduce，回答全局问题

社区摘要是什么？
  对每个社区内的实体、关系、源文本，让 LLM 生成一段"这群实体讲了什么主题"的概括。
  相当于给知识图谱的一个子图写了一个"目录条目"。

算法选择：
  - Leiden（首选）：保证社区内部连通，质量最好，需要 graspologic
  - Louvain（退路）：networkx 内置，经典方法，质量略差但够用
  学习阶段用 Louvain 完全可以，原理一样
"""

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import defaultdict

import networkx as nx
from openai import OpenAI
from config import config
from phase5_entity_extractor import Entity, Relationship
from phase5_knowledge_graph import KnowledgeGraph


# ========== 数据模型 ==========


@dataclass
class Community:
    """
    知识图谱中的一个社区

    社区 = 一组关系紧密的实体。你可以把它类比成：
      - 一本书的一个章节
      - 一个研究领域的子方向
      - 一群紧密合作的人

    示例：
      Community(
          community_id=0,
          level=0,
          title="Transformer 核心架构",
          entity_names=["TRANSFORMER", "SELF-ATTENTION", "MULTI-HEAD ATTENTION", ...],
          summary="此社区围绕 Transformer 的核心架构组件...",
      )
    """

    community_id: int  # 社区编号
    level: int  # 层级（0=最细粒度）
    title: str  # 社区标题（LLM 生成）
    entity_names: list[str]  # 社区内的实体名称列表
    relationship_descriptions: list[str]  # 社区内的关系描述列表
    summary: str  # LLM 生成的社区摘要
    source_chunk_ids: list[str] = field(default_factory=list)  # 相关的源 chunk


# ========== 社区摘要 Prompt ==========

_COMMUNITY_SUMMARY_PROMPT = """你是一个知识图谱分析师。下面是知识图谱中一个社区的信息。
这个社区包含一组紧密相关的实体和关系。

请为这个社区生成：
1. 一个简短的标题（不超过 15 个字）
2. 一段全面的摘要（100-200 字），说明这个社区的核心主题、关键实体和重要关系

## 社区中的实体

{entities_text}

## 社区中的关系

{relationships_text}

## 输出格式

请严格以 JSON 格式输出：
{{"title": "社区标题", "summary": "社区摘要..."}}

只输出 JSON，不加任何其他内容。"""


# ========== 社区检测器 ==========


class CommunityDetector:
    """
    社区检测 + 摘要生成

    两步走：
      1. 用图算法把实体分成社区
      2. 用 LLM 为每个社区写摘要

    使用方式：
        kg = KnowledgeGraph.load_from_json("phase5_knowledge_graph.json")
        detector = CommunityDetector()
        communities = detector.detect_and_summarize(kg)
        detector.save_communities(communities, "phase5_communities.json")

    算法：优先用 Leiden（需 graspologic），退路用 Louvain（networkx 内置）
    """

    def __init__(
        self,
        llm_client: OpenAI | None = None,
        model: str | None = None,
        algorithm: str = "auto",  # "auto" / "louvain" / "leiden"
        resolution: float = 1.0,  # 分辨率参数：越大社区越多越小
    ):
        """
        Args:
            llm_client: LLM 客户端
            model: 模型名称
            algorithm: 社区检测算法
            resolution: 分辨率，越大社区越细粒度
        """
        self.client = llm_client or OpenAI(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
        )
        self.model = model or config.llm_model
        self.algorithm = algorithm
        self.resolution = resolution

    # ========== 主流程 ==========

    def detect_and_summarize(
        self,
        kg: KnowledgeGraph,
        verbose: bool = True,
    ) -> list[Community]:
        """
        完整流程：构建 NetworkX 图 → 社区检测 → LLM 摘要

        Args:
            kg: 知识图谱
            verbose: 是否打印过程

        Returns:
            社区列表
        """
        if verbose:
            print("🔬 开始社区检测...")

        # Step 1: 构建 NetworkX 图
        G = self._build_nx_graph(kg, verbose)

        # Step 2: 社区检测
        node_to_community = self._detect_communities(G, verbose)

        # Step 3: 组装社区数据
        communities = self._build_communities(kg, node_to_community, verbose)

        # Step 4: LLM 生成摘要
        communities = self._generate_summaries(communities, verbose)

        if verbose:
            print(f"\n✅ 社区检测完成: {len(communities)} 个社区")
            for c in communities:
                print(f"  📁 社区 {c.community_id}: {c.title} "
                      f"({len(c.entity_names)} 个实体)")

        return communities

    # ========== Step 1: 构建 NetworkX 图 ==========

    def _build_nx_graph(self, kg: KnowledgeGraph, verbose: bool) -> nx.Graph:
        """
        从 KnowledgeGraph 转为 NetworkX Graph。

        为什么要转？
          KnowledgeGraph 是我们自定义的数据结构（dict + list），
          NetworkX 是专业的图分析库，提供了社区检测、最短路径等算法。
          转换很简单：实体 → 节点，关系 → 边。
        """
        G = nx.Graph()

        # 添加节点
        for name, entity in kg.entities.items():
            G.add_node(
                name,
                entity_type=entity.entity_type,
                description=entity.description[:200],  # 截断太长的描述
            )

        # 添加边
        for rel in kg.relationships:
            if rel.source in kg.entities and rel.target in kg.entities:
                G.add_edge(
                    rel.source,
                    rel.target,
                    weight=rel.strength,
                    description=rel.description,
                )

        if verbose:
            print(f"  📊 NetworkX 图: {G.number_of_nodes()} 节点, "
                  f"{G.number_of_edges()} 边")
            # 连通分量
            components = list(nx.connected_components(G))
            print(f"  📊 连通分量: {len(components)} 个 "
                  f"(最大: {max(len(c) for c in components)} 节点)")

        return G

    # ========== Step 2: 社区检测 ==========

    def _detect_communities(
        self,
        G: nx.Graph,
        verbose: bool,
    ) -> dict[str, int]:
        """
        运行社区检测算法，返回 {节点名: 社区ID} 映射。

        算法选择逻辑：
          auto → 先尝试 Leiden（需要 graspologic），失败就用 Louvain
          louvain → 直接用 networkx 内置的 Louvain
          leiden → 必须有 graspologic
        """
        algorithm = self.algorithm

        if algorithm == "auto":
            try:
                from graspologic.partition import hierarchical_leiden  # noqa: F401
                algorithm = "leiden"
            except ImportError:
                algorithm = "louvain"

        if verbose:
            print(f"  🧮 使用算法: {algorithm}")

        if algorithm == "leiden":
            return self._leiden_communities(G, verbose)
        else:
            return self._louvain_communities(G, verbose)

    def _louvain_communities(self, G: nx.Graph, verbose: bool) -> dict[str, int]:
        """
        Louvain 社区检测（networkx 内置）。

        Louvain 是经典的模块度优化算法：
          1. 每个节点初始为独立社区
          2. 迭代把节点移动到能最大化提升模块度的邻居社区
          3. 收敛后把每个社区压缩成一个超级节点，重复
          4. 直到模块度不再提升

        resolution 参数：
          > 1.0 → 更多更小的社区（精细分组）
          < 1.0 → 更少更大的社区（粗略分组）
          = 1.0 → 默认
        """
        communities_gen = nx.community.louvain_communities(
            G,
            resolution=self.resolution,
            seed=42,  # 固定种子保证可复现
        )

        node_to_community = {}
        for i, community_nodes in enumerate(communities_gen):
            for node in community_nodes:
                node_to_community[node] = i

        if verbose:
            n_communities = len(set(node_to_community.values()))
            print(f"  📊 检测到 {n_communities} 个社区")

        return node_to_community

    def _leiden_communities(self, G: nx.Graph, verbose: bool) -> dict[str, int]:
        """
        Leiden 社区检测（需要 graspologic）。

        Leiden 是 Louvain 的改进版：
          - 保证社区内部连通（Louvain 可能产生断裂的社区）
          - 质量更好
          - 速度也不差
        """
        from graspologic.partition import hierarchical_leiden

        community_mapping = hierarchical_leiden(
            G,
            max_cluster_size=10,
            random_seed=42,
        )

        # hierarchical_leiden 返回 list of HierarchicalCluster(node, cluster, parent_cluster, level, is_final_cluster)
        # 我们先用 level 0（最细粒度）
        node_to_community = {}
        for item in community_mapping:
            node = getattr(item, "node", None)
            cluster_id = getattr(item, "cluster", None)
            level = getattr(item, "level", 0)
            if node is not None and level == 0:
                node_to_community[node] = cluster_id

        # 处理没分到社区的孤立节点
        for node in G.nodes():
            if node not in node_to_community:
                # 分配到一个新社区
                max_id = max(node_to_community.values(), default=-1)
                node_to_community[node] = max_id + 1

        if verbose:
            n_communities = len(set(node_to_community.values()))
            print(f"  📊 检测到 {n_communities} 个社区 (Leiden)")

        return node_to_community

    # ========== Step 3: 组装社区数据 ==========

    def _build_communities(
        self,
        kg: KnowledgeGraph,
        node_to_community: dict[str, int],
        verbose: bool,
    ) -> list[Community]:
        """
        根据社区分配结果，组装每个社区的详细信息。

        每个社区包含：
          - 属于它的所有实体
          - 两端都在社区内的关系
          - 实体对应的源 chunk IDs
        """
        # 按社区 ID 分组实体
        community_entities: dict[int, list[str]] = defaultdict(list)
        for node, cid in node_to_community.items():
            community_entities[cid].append(node)

        communities = []
        for cid in sorted(community_entities.keys()):
            entity_names = community_entities[cid]

            # 找社区内部的关系（两端都在同一社区）
            entity_set = set(entity_names)
            internal_rels = [
                rel for rel in kg.relationships
                if rel.source in entity_set and rel.target in entity_set
            ]
            rel_descriptions = [
                f"{rel.source} → {rel.target}: {rel.description}"
                for rel in internal_rels
            ]

            # 收集源 chunk IDs
            all_chunk_ids = set()
            for name in entity_names:
                if name in kg.entities:
                    all_chunk_ids.update(kg.entities[name].source_chunk_ids)

            community = Community(
                community_id=cid,
                level=0,
                title="",  # 后面由 LLM 生成
                entity_names=entity_names,
                relationship_descriptions=rel_descriptions,
                summary="",  # 后面由 LLM 生成
                source_chunk_ids=list(all_chunk_ids),
            )
            communities.append(community)

        if verbose:
            sizes = [len(c.entity_names) for c in communities]
            print(f"  📊 社区规模分布: 最小={min(sizes)}, "
                  f"最大={max(sizes)}, 平均={sum(sizes)/len(sizes):.1f}")

        return communities

    # ========== Step 4: LLM 社区摘要 ==========

    def _generate_summaries(
        self,
        communities: list[Community],
        verbose: bool,
    ) -> list[Community]:
        """
        用 LLM 为每个社区生成标题和摘要。

        输入：社区内的实体描述 + 关系描述
        输出：一个简短标题 + 一段全面的摘要
        """
        for i, community in enumerate(communities):
            if verbose:
                print(f"\n  📝 生成社区 {community.community_id} 的摘要 "
                      f"({len(community.entity_names)} 个实体)...")

            # 组装实体信息
            entities_text = "\n".join(
                f"- {name}" for name in community.entity_names
            )

            # 组装关系信息
            if community.relationship_descriptions:
                relationships_text = "\n".join(
                    f"- {desc}" for desc in community.relationship_descriptions
                )
            else:
                relationships_text = "（此社区内部没有直接关系）"

            # 调用 LLM
            prompt = _COMMUNITY_SUMMARY_PROMPT.format(
                entities_text=entities_text,
                relationships_text=relationships_text,
            )

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                result = json.loads(response.choices[0].message.content)
                community.title = result.get("title", f"社区 {community.community_id}")
                community.summary = result.get("summary", "")

                if verbose:
                    print(f"     标题: {community.title}")
                    print(f"     摘要: {community.summary[:80]}...")

            except Exception as e:
                print(f"  ⚠️  社区 {community.community_id} 摘要生成失败: {e}")
                community.title = f"社区 {community.community_id}"
                community.summary = f"包含实体: {', '.join(community.entity_names[:5])}..."

        return communities

    # ========== 持久化 ==========

    @staticmethod
    def save_communities(communities: list[Community], filepath: str | Path):
        """保存社区到 JSON"""
        filepath = Path(filepath)
        data = [asdict(c) for c in communities]
        filepath.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"💾 社区已保存到 {filepath} ({filepath.stat().st_size / 1024:.1f} KB)")

    @staticmethod
    def load_communities(filepath: str | Path) -> list[Community]:
        """从 JSON 加载社区"""
        filepath = Path(filepath)
        data = json.loads(filepath.read_text(encoding="utf-8"))
        communities = [Community(**item) for item in data]
        print(f"📂 已加载 {len(communities)} 个社区")
        return communities


# ========== 独立运行 Demo ==========


def demo():
    """
    独立运行 demo：从已有的知识图谱做社区检测 + 摘要

    前提：先运行过 phase5_knowledge_graph.py 生成了 JSON

    用法: uv run python phase5_community.py
    """
    print("=" * 60)
    print("Phase 5 — 社区检测 + 摘要 Demo")
    print("=" * 60)

    # 加载已有的知识图谱
    kg_path = Path(__file__).parent / "phase5_knowledge_graph.json"
    if not kg_path.exists():
        print("❌ 请先运行 phase5_knowledge_graph.py 构建知识图谱！")
        return

    kg = KnowledgeGraph.load_from_json(kg_path)

    # 社区检测 + 摘要
    detector = CommunityDetector(resolution=1.0)
    communities = detector.detect_and_summarize(kg)

    # 保存
    save_path = Path(__file__).parent / "phase5_communities.json"
    detector.save_communities(communities, save_path)

    # 打印详情
    print(f"\n{'='*60}")
    print("📦 社区详情:")
    for c in communities:
        print(f"\n  📁 社区 {c.community_id}: {c.title}")
        print(f"     实体数: {len(c.entity_names)}")
        print(f"     成员: {', '.join(c.entity_names[:8])}"
              f"{'...' if len(c.entity_names) > 8 else ''}")
        print(f"     内部关系: {len(c.relationship_descriptions)} 条")
        print(f"     摘要: {c.summary[:120]}...")

    print(f"\n✅ Demo 完成")


if __name__ == "__main__":
    demo()
