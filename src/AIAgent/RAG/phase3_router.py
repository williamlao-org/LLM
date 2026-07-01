"""
知识库路由器 —— 多知识库 LLM 路由

当你有多个知识库时（比如技术文档库、FAQ 库、产品手册库），
不应该每次都全搜一遍——那样噪声太多、延迟也高。

路由器的作用：根据用户问题的类型，选择最合适的知识库去检索。

  用户问题
      │
      ▼
  LLM Router（路由器）
      │
      ├── "Transformer 注意力怎么算？" → 技术文档库
      │
      ├── "如何实现数字化转型？"       → 通用知识库
      │
      └── "ReAct Agent 和 RAG 有什么关系？" → 技术文档库
                                              + 通用知识库（多库）

实现方式：
  把每个知识库的名称和描述告诉 LLM，让它选择。
  这是最灵活的路由方式——不需要训练分类器，新增知识库只要写个描述就行。

为什么不用 embedding 相似度做路由？
  - 知识库的「主题」是高层语义概念，embedding 擅长的是段落级相似度
  - LLM 理解「这个问题应该查什么库」比 embedding 准得多
  - 代价只是一次轻量级 LLM 调用（不需要生成长文本）
"""

import json
from dataclasses import dataclass, field
from openai import OpenAI
from phase1_dense_retriever import SearchResult


# ========== 数据结构 ==========


@dataclass
class KnowledgeBase:
    """一个命名的知识库"""

    name: str  # 知识库名称，如 "tech_docs"
    description: str  # 描述（给 LLM 看的），如 "技术文档库，包含 AI、深度学习等技术文档"
    retriever: object  # 实现了 search(query, top_k) -> list[SearchResult] 接口的检索器
    # 可选：知识库包含的文件列表（帮助 LLM 做出更准确的路由）
    file_list: list[str] = field(default_factory=list)


@dataclass
class RouteDecision:
    """路由决策结果"""

    selected_kbs: list[str]  # 选中的知识库名称列表
    reason: str  # 路由理由


# ========== 路由 Prompt ==========

_ROUTER_SYSTEM = """你是一个智能查询路由器。你的任务是根据用户的问题，选择最合适的知识库进行检索。

可用的知识库：
{kb_descriptions}

## 路由规则
1. 根据问题的主题和意图，选择最相关的知识库
2. 如果问题涉及多个领域，可以选择多个知识库
3. 如果问题是纯常识/闲聊，选择空列表（不需要检索）

请以 JSON 格式输出：
{{
    "selected": ["知识库名称1", "知识库名称2"],
    "reason": "选择理由（一句话）"
}}

只输出 JSON，不加任何其他内容。"""


# ========== 路由器 ==========


class KnowledgeRouter:
    """
    多知识库 LLM 路由器

    使用 LLM 判断用户问题应该路由到哪个知识库检索。

    使用示例：
        # 注册知识库
        router = KnowledgeRouter(llm_client, model="deepseek-chat")
        router.add_kb(KnowledgeBase(
            name="tech_docs",
            description="AI/ML 技术文档，包括 Transformer、RAG、Agent 等",
            retriever=hybrid_retriever_1,
        ))
        router.add_kb(KnowledgeBase(
            name="general",
            description="通用知识库，包括商业、个人成长等书籍",
            retriever=hybrid_retriever_2,
        ))

        # 路由并检索
        results = router.route_and_search("Transformer 注意力机制", top_k=3)
    """

    def __init__(self, llm_client: OpenAI, model: str):
        self.llm_client = llm_client
        self.model = model
        self.knowledge_bases: dict[str, KnowledgeBase] = {}

    def add_kb(self, kb: KnowledgeBase):
        """注册一个知识库"""
        self.knowledge_bases[kb.name] = kb

    def route(self, question: str, verbose: bool = True) -> RouteDecision:
        """
        根据用户问题决定路由到哪些知识库。

        Args:
            question: 用户问题
            verbose: 是否打印路由过程

        Returns:
            RouteDecision 路由决策
        """
        if not self.knowledge_bases:
            raise ValueError("没有注册任何知识库，请先调用 add_kb()")

        # 如果只有一个知识库，不需要路由
        if len(self.knowledge_bases) == 1:
            name = list(self.knowledge_bases.keys())[0]
            if verbose:
                print(f"     📚 只有一个知识库 [{name}]，直接使用")
            return RouteDecision(selected_kbs=[name], reason="仅一个知识库可用")

        # 构建知识库描述
        kb_desc_parts = []
        for name, kb in self.knowledge_bases.items():
            desc = f"- **{name}**: {kb.description}"
            if kb.file_list:
                files = ", ".join(kb.file_list)
                desc += f"（包含: {files}）"
            kb_desc_parts.append(desc)
        kb_descriptions = "\n".join(kb_desc_parts)

        system_prompt = _ROUTER_SYSTEM.format(kb_descriptions=kb_descriptions)

        if verbose:
            print("     🔀 Router: 分析问题类型，选择知识库...")

        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.1,
        )

        raw = response.choices[0].message.content.strip()

        # 解析 JSON
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # 解析失败：默认搜所有库
            all_names = list(self.knowledge_bases.keys())
            if verbose:
                print(f"     ⚠️ 路由结果解析失败，搜索全部知识库: {all_names}")
            return RouteDecision(
                selected_kbs=all_names,
                reason="路由解析失败，回退到全部知识库",
            )

        selected = data.get("selected", [])
        reason = data.get("reason", "")

        # 过滤不存在的知识库名称
        valid_selected = [name for name in selected if name in self.knowledge_bases]

        # 如果全被过滤掉了（LLM 幻觉了不存在的名称），搜所有库
        if not valid_selected:
            valid_selected = list(self.knowledge_bases.keys())
            reason += "（路由目标无效，回退到全部知识库）"

        decision = RouteDecision(selected_kbs=valid_selected, reason=reason)

        if verbose:
            kb_list = ", ".join(f"[{name}]" for name in decision.selected_kbs)
            print(f"     路由到: {kb_list}")
            print(f"     理由: {decision.reason}")

        return decision

    def search(
        self,
        question: str,
        kb_names: list[str],
        top_k: int = 3,
        verbose: bool = True,
    ) -> list[SearchResult]:
        """
        在指定的知识库中检索。

        如果有多个知识库，每个库各取 top_k 条，然后按 score 降序合并，
        截取总共 top_k 条。

        Args:
            question: 检索查询
            kb_names: 要搜索的知识库名称列表
            top_k: 返回结果数量
            verbose: 是否打印检索过程

        Returns:
            合并后的检索结果
        """
        all_results: list[SearchResult] = []

        for name in kb_names:
            kb = self.knowledge_bases.get(name)
            if kb is None:
                continue

            if verbose:
                print(f"     📖 在 [{name}] 中检索...")

            results = kb.retriever.search(question, top_k=top_k)
            # 给每条结果标记来源知识库
            for r in results:
                r.chunk.metadata["knowledge_base"] = name
            all_results.extend(results)

            if verbose:
                print(f"        找到 {len(results)} 条结果")

        # 按 score 降序排序，取 top_k
        all_results.sort(key=lambda r: r.score, reverse=True)
        merged = all_results[:top_k]

        if verbose and len(kb_names) > 1:
            print(f"     📋 多库合并后保留 Top-{len(merged)} 条")

        return merged

    def route_and_search(
        self,
        question: str,
        top_k: int = 3,
        verbose: bool = True,
    ) -> tuple[list[SearchResult], RouteDecision]:
        """
        路由 + 检索一步到位。

        Returns:
            (检索结果列表, 路由决策)
        """
        decision = self.route(question, verbose=verbose)
        results = self.search(question, decision.selected_kbs, top_k=top_k, verbose=verbose)
        return results, decision
