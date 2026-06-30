"""可审计的 RAG 评估器。

阅读顺序：
  1. phase2_04_models.py：流程中的数据长什么样；
  2. phase2_04_metrics.py：Prompt、Judge 输出结构和分数公式；
  3. 本文件：如何收集 RAG 输出、并发评分并生成报告。

运行：
    uv run python phase2_04_evaluator.py --samples 1  # 默认 official
    uv run python phase2_04_evaluator.py --samples 1 --prompt-profile custom_zh
    uv run python phase2_04_evaluator.py --samples 1 --trace  # 展示 Profile 与完整评分过程
    uv run python phase2_04_evaluator.py --compare
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

from config import config
from phase2_04_eval_dataset import EVAL_SAMPLES
from phase2_04_metrics import (
    METRIC_LABELS,
    METRIC_TIPS,
    EvaluatorMetric,
    build_metrics,
)
from phase2_04_models import (
    EvaluationReport,
    EvaluationSample,
    MetricScore,
    RAGOutput,
    SampleEvaluation,
)
from phase2_04_prompt_profiles import (
    DEFAULT_PROMPT_PROFILE,
    PROMPT_PROFILE_NAMES,
    get_prompt_profile,
)
from rag_chain import RAGChain


# ──────────────────────────────────────────────
# Judge 模型
# ──────────────────────────────────────────────


def build_judge_llm() -> Any:
    """用 RAGAS 的统一客户端连接 OpenAI 兼容的 DeepSeek 接口。"""
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    client = AsyncOpenAI(
        api_key=config.ragas_llm_api_key,
        base_url=config.ragas_llm_base_url,
    )
    return llm_factory(
        config.ragas_llm_model, client=client, temperature=0, max_tokens=4096
    )


def build_judge_embeddings() -> Any:
    """连接 Answer Relevancy 使用的 OpenAI 兼容 Embedding 接口。"""
    from openai import AsyncOpenAI
    from ragas.embeddings import OpenAIEmbeddings

    client = AsyncOpenAI(
        api_key=config.ragas_embedding_api_key,
        base_url=config.ragas_embedding_base_url,
    )
    return OpenAIEmbeddings(
        client=client,
        model=config.ragas_embedding_model,
    )


# ──────────────────────────────────────────────
# RAG 配置
# ──────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """一次待评估的 RAG 配置。"""

    name: str
    label: str
    retriever_type: str = "dense"
    use_reranker: bool = False
    query_rewrite: str = "none"
    top_k: int = 3


BASELINE_CONFIG = EvalConfig(
    name="baseline",
    label="Baseline（Dense only）",
)

FULL_CONFIG = EvalConfig(
    name="full",
    label="Full（Hybrid + Reranker + Multi-Query）",
    retriever_type="hybrid",
    use_reranker=True,
    query_rewrite="multi_query",
)


# ──────────────────────────────────────────────
# 第一步：运行 RAG，得到强类型输出
# ──────────────────────────────────────────────


def collect_rag_outputs(
    rag: RAGChain,
    samples: list[EvaluationSample],
    top_k: int = 3,
) -> list[RAGOutput]:
    """逐条查询 RAG，将松散的响应字典转换成明确的 ``RAGOutput``。"""
    print(f"\n{'─' * 60}")
    print(f"📥 收集 RAG 输出（共 {len(samples)} 条）")
    print(f"{'─' * 60}")

    outputs: list[RAGOutput] = []
    for index, sample in enumerate(samples, 1):
        print(f"\n  [{index}/{len(samples)}] {sample.question}")
        try:
            raw_result = rag.query(sample.question, top_k=top_k, verbose=False)
            contexts = [
                source["content_preview"]
                for source in raw_result.get("sources", [])
                if source.get("content_preview")
            ]
            if not contexts and raw_result.get("context"):
                contexts = [raw_result["context"]]

            output = RAGOutput(
                question=sample.question,
                answer=raw_result["answer"],
                contexts=contexts,
                ground_truth=sample.ground_truth,
            )
            outputs.append(output)
            print(
                f"     ✅ answer({len(output.answer)} 字) "
                f"contexts({len(output.contexts)} 条)"
            )
        except Exception as error:
            print(f"     ❌ 查询失败，保留为空输出: {error}")
            outputs.append(RAGOutput.failed(sample))

    return outputs


# ──────────────────────────────────────────────
# 第二步：运行我们自己的可审计指标
# ──────────────────────────────────────────────


def print_metric_trace(
    sample_index: int,
    total_samples: int,
    question: str,
    score: MetricScore,
) -> None:
    """打印单条指标结果；Judge 的完整交互在 ``--trace`` 时展示。"""
    label = METRIC_LABELS[score.metric_name]
    print(
        f"  📐 [{sample_index}/{total_samples}] {label}: "
        f"{score.value:.4f}（{score.calculation}）"
    )
    print(f"     Prompt Profile: {score.prompt_profile}")
    print(f"     问题: {question}")
    for step_index, step in enumerate(score.judge_steps, 1):
        print(f"\n     ┌─ Judge Step {step_index}: {step.name}")
        print("     │ Prompt:")
        for line in step.prompt.splitlines():
            print(f"     │   {line}")
        print("     │ Structured Output:")
        output_json = json.dumps(step.output, ensure_ascii=False, indent=2)
        for line in output_json.splitlines():
            print(f"     │   {line}")
        print("     └─")


async def evaluate_outputs(
    outputs: list[RAGOutput],
    metrics: dict[str, EvaluatorMetric],
    trace: bool = False,
) -> EvaluationReport:
    """对每个 ``RAGOutput × Metric`` 评分，并保留逐条完整结果。"""
    profile_names = {metric.prompt_profile.name for metric in metrics.values()}
    if len(profile_names) != 1:
        raise ValueError(f"一次评估只能使用一个 Prompt Profile: {profile_names}")
    prompt_profile = profile_names.pop()
    scores_by_sample: list[dict[str, MetricScore]] = [{} for _ in outputs]

    async def score_one(
        sample_index: int,
        metric: EvaluatorMetric,
    ) -> tuple[int, str, MetricScore | None, Exception | None]:
        try:
            score = await metric.score(outputs[sample_index])
            return sample_index, metric.name, score, None
        except Exception as error:
            return sample_index, metric.name, None, error

    tasks = [
        asyncio.create_task(score_one(sample_index, metric))
        for sample_index in range(len(outputs))
        for metric in metrics.values()
    ]

    for task in asyncio.as_completed(tasks):
        sample_index, metric_name, score, error = await task
        human_index = sample_index + 1
        if error is not None:
            print(
                f"  ❌ [{human_index}/{len(outputs)}] "
                f"{METRIC_LABELS[metric_name]} 失败: {error}"
            )
            continue
        if score is None:
            print(
                f"  ⏭️  [{human_index}/{len(outputs)}] "
                f"{METRIC_LABELS[metric_name]}：输入不完整，跳过"
            )
            continue

        scores_by_sample[sample_index][metric_name] = score
        if trace:
            print_metric_trace(
                human_index,
                len(outputs),
                outputs[sample_index].question,
                score,
            )
        else:
            print(
                f"  ✅ [{human_index}/{len(outputs)}] "
                f"{METRIC_LABELS[metric_name]} = {score.value:.4f}"
            )

    return EvaluationReport(
        prompt_profile=prompt_profile,
        samples=[
            SampleEvaluation(output=output, scores=scores_by_sample[index])
            for index, output in enumerate(outputs)
        ]
    )


def run_metrics(
    outputs: list[RAGOutput],
    metrics: dict[str, EvaluatorMetric],
    trace: bool = False,
) -> EvaluationReport:
    """同步 CLI 到异步指标执行器的唯一边界。"""
    return asyncio.run(evaluate_outputs(outputs, metrics, trace=trace))


def evaluate_config(
    eval_config: EvalConfig,
    samples: list[EvaluationSample],
    metrics: dict[str, EvaluatorMetric],
    trace: bool = False,
) -> EvaluationReport:
    """运行一组完整的 RAG 配置并返回结构化报告。"""
    print(f"\n{'=' * 70}")
    print(f"🧪 评估配置：{eval_config.label}")
    print(f"🧾 Prompt Profile：{next(iter(metrics.values())).prompt_profile.name}")
    print(f"{'=' * 70}")

    rag = RAGChain(
        embedder_type="api",
        store_type="simple",
        retriever_type=eval_config.retriever_type,
        use_reranker=eval_config.use_reranker,
        query_rewrite=eval_config.query_rewrite,
    )
    if not rag.load_index():
        print("📦 本地索引不存在，重新构建...")
        rag.build_index()
        rag.save_index()

    started_at = time.time()
    outputs = collect_rag_outputs(rag, samples, top_k=eval_config.top_k)
    print(f"\n⏱️  RAG 输出收集耗时: {time.time() - started_at:.1f}s")

    print(f"\n{'─' * 70}")
    print("📊 运行自定义可审计指标（LLM-as-Judge）")
    print(f"{'─' * 70}")
    started_at = time.time()
    report = run_metrics(outputs, metrics, trace=trace)
    print(f"⏱️  指标评估耗时: {time.time() - started_at:.1f}s")
    return report


# ──────────────────────────────────────────────
# 第三步：报告
# ──────────────────────────────────────────────


def score_bar(score: float | None, width: int = 20) -> str:
    if score is None:
        return "N/A"
    filled = int(round(score * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {score:.4f}"


def print_single_report(label: str, report: EvaluationReport) -> None:
    """打印聚合分数；逐条结果仍保存在 ``EvaluationReport`` 中。"""
    averages = report.average_scores()
    print(f"\n{'=' * 70}")
    print(f"📋 评估报告：{label}")
    print(f"🧾 Prompt Profile：{report.prompt_profile}")
    print(f"{'=' * 70}")
    for metric_name, friendly_name in METRIC_LABELS.items():
        print(f"\n  {friendly_name}")
        print(f"  {score_bar(averages.get(metric_name))}")
        print(f"  💡 {METRIC_TIPS[metric_name]}")


def print_comparison_report(
    baseline_report: EvaluationReport,
    full_report: EvaluationReport,
) -> None:
    baseline = baseline_report.average_scores()
    full = full_report.average_scores()

    print(f"\n{'=' * 70}")
    print("📊 对比报告：Baseline vs Full")
    print(
        "🧾 Prompt Profile："
        f"{baseline_report.prompt_profile} / {full_report.prompt_profile}"
    )
    print(f"{'=' * 70}")
    print(f"{'指标':<30} {'Baseline':>12} {'Full':>12} {'提升':>12}")
    print(f"{'─' * 70}")
    for metric_name in METRIC_LABELS:
        baseline_value = baseline.get(metric_name)
        full_value = full.get(metric_name)
        baseline_text = f"{baseline_value:.4f}" if baseline_value is not None else "N/A"
        full_text = f"{full_value:.4f}" if full_value is not None else "N/A"
        if baseline_value is None or full_value is None:
            delta_text = "─"
        else:
            delta = full_value - baseline_value
            delta_text = f"{delta:+.4f}"
        print(
            f"  {metric_name:<28} {baseline_text:>12} {full_text:>12} {delta_text:>12}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可审计的 RAG 评估器")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="对比 Baseline 与 Full 两组配置",
    )
    parser.add_argument(
        "--config",
        choices=["baseline", "full"],
        default="full",
        help="单组评估使用的配置（默认 full）",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="只跑前 N 条样本；0 表示全部",
    )
    parser.add_argument(
        "--prompt-profile",
        choices=PROMPT_PROFILE_NAMES,
        default=DEFAULT_PROMPT_PROFILE,
        help="Judge Prompt 配置（默认 official）",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="打印每次 Judge 的完整 Prompt、结构化输出和分数公式",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = EVAL_SAMPLES[: args.samples] if args.samples > 0 else EVAL_SAMPLES

    print("\n🚀 可审计 RAG 评估启动")
    print(f"   样本数量: {len(samples)}")
    print(f"   模式: {'Baseline + Full 对比' if args.compare else args.config}")
    profile = get_prompt_profile(args.prompt_profile)
    print(
        f"   Prompt Profile: {profile.name} "
        f"({profile.description}; source={profile.source_version})"
    )
    print(f"   完整 Judge Trace: {'开启' if args.trace else '关闭'}")

    print("\n🔧 初始化 Judge LLM、Embedding 与自定义指标...")
    metrics = build_metrics(
        build_judge_llm(),
        build_judge_embeddings(),
        prompt_profile=profile,
    )
    started_at = time.time()

    if args.compare:
        baseline_report = evaluate_config(
            BASELINE_CONFIG,
            samples,
            metrics,
            trace=args.trace,
        )
        full_report = evaluate_config(
            FULL_CONFIG,
            samples,
            metrics,
            trace=args.trace,
        )
        print_single_report(BASELINE_CONFIG.label, baseline_report)
        print_single_report(FULL_CONFIG.label, full_report)
        print_comparison_report(baseline_report, full_report)
    else:
        eval_config = BASELINE_CONFIG if args.config == "baseline" else FULL_CONFIG
        report = evaluate_config(eval_config, samples, metrics, trace=args.trace)
        print_single_report(eval_config.label, report)

    print(f"\n✅ 评估完成，总耗时: {time.time() - started_at:.1f}s")


if __name__ == "__main__":
    main()
