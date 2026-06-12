"""
Reusable agent graph engine for modular multi-agent workflows.

This is intentionally separate from StateMachine_GraphFlow.rewrite_graph.
rewrite_graph.py is a good learning scaffold; this engine adds the pieces that
make modular composition practical:

- stable run state with dynamic data/artifacts
- modules with required/optional inputs and declared outputs
- graph compile validation before running
- runtime event log and error collection
- conditional routing with explicit END handling
- runtime checks for missing inputs and undeclared outputs
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal


END = "__END__"
ErrorPolicy = Literal["raise", "stop", "record"]


class GraphCompileError(ValueError):
    """Raised when a graph cannot be compiled safely."""


class GraphRuntimeError(RuntimeError):
    """Raised when graph execution fails under error_policy='raise'."""


@dataclass
class RunEvent:
    step: int
    node: str
    kind: str
    detail: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


@dataclass
class RunState:
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)
    events: list[RunEvent] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    current_node: str = ""
    step_count: int = 0
    status: Literal["ready", "running", "completed", "failed", "max_steps"] = "ready"
    answer: Any = None

    def emit(self, node: str, kind: str, detail: dict[str, Any] | None = None) -> None:
        self.events.append(
            RunEvent(
                step=self.step_count,
                node=node,
                kind=kind,
                detail=detail or {},
            )
        )


@dataclass(frozen=True)
class ModuleContext:
    module_name: str
    inputs: dict[str, Any]
    state: RunState

    def get(self, key: str, default: Any = None) -> Any:
        return self.state.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.state.data[key] = value

    def artifact(self, key: str, value: Any) -> None:
        self.state.artifacts[key] = value

    def emit(self, kind: str, detail: dict[str, Any] | None = None) -> None:
        self.state.emit(self.module_name, kind, detail)


ModuleRunFunc = Callable[[ModuleContext], dict[str, Any]]
RouteFunc = Callable[[RunState], str]


@dataclass(frozen=True)
class AgentModule:
    name: str
    inputs: list[str]
    outputs: list[str]
    run: ModuleRunFunc
    optional_inputs: list[str] = field(default_factory=list)
    description: str = ""
    allow_extra_outputs: bool = False
    require_all_outputs: bool = False

    def read_inputs(self, state: RunState) -> tuple[dict[str, Any], list[str]]:
        missing = [key for key in self.inputs if key not in state.data]
        payload = {key: state.data[key] for key in self.inputs if key in state.data}
        for key in self.optional_inputs:
            if key in state.data:
                payload[key] = state.data[key]
        return payload, missing


@dataclass(frozen=True)
class Edge:
    source: str
    target: str


@dataclass(frozen=True)
class ConditionalEdge:
    source: str
    router: RouteFunc
    routes: dict[str, str]


@dataclass(frozen=True)
class CompiledAgentGraph:
    modules: dict[str, AgentModule]
    edges: dict[str, list[str]]
    conditional_edges: dict[str, ConditionalEdge]
    start: str

    def run(
        self,
        state: RunState | None = None,
        *,
        max_steps: int = 30,
        error_policy: ErrorPolicy = "raise",
    ) -> RunState:
        if state is None:
            state = RunState()

        state.status = "running"
        current = self.start

        while current != END:
            if state.step_count >= max_steps:
                state.status = "max_steps"
                state.emit(current, "max_steps", {"max_steps": max_steps})
                return state

            module = self.modules.get(current)
            if module is None:
                self._handle_error(
                    state,
                    current,
                    "unknown_node",
                    {"node": current},
                    error_policy,
                )
                return state

            state.current_node = current
            state.step_count += 1
            state.emit(current, "enter", {"inputs": module.inputs})

            payload, missing = module.read_inputs(state)
            if missing:
                self._handle_error(
                    state,
                    current,
                    "missing_inputs",
                    {"missing": missing},
                    error_policy,
                )
                return state

            try:
                result = module.run(ModuleContext(current, payload, state))
            except Exception as exc:  # noqa: BLE001
                self._handle_error(
                    state,
                    current,
                    "module_exception",
                    {"error": repr(exc)},
                    error_policy,
                )
                return state

            if not isinstance(result, dict):
                self._handle_error(
                    state,
                    current,
                    "invalid_output",
                    {"expected": "dict", "actual": type(result).__name__},
                    error_policy,
                )
                return state

            accepted, ignored, missing_outputs = self._apply_outputs(module, state, result)
            state.emit(
                current,
                "exit",
                {
                    "accepted_outputs": accepted,
                    "ignored_outputs": ignored,
                    "missing_outputs": missing_outputs,
                },
            )

            if missing_outputs and module.require_all_outputs:
                self._handle_error(
                    state,
                    current,
                    "missing_outputs",
                    {"missing": missing_outputs},
                    error_policy,
                )
                return state

            current = self._next_node(current, state, error_policy)
            if state.status in {"failed", "completed"}:
                return state

        state.status = "completed"
        state.emit(END, "completed", {})
        return state

    def _apply_outputs(
        self, module: AgentModule, state: RunState, result: dict[str, Any]
    ) -> tuple[list[str], list[str], list[str]]:
        declared = set(module.outputs)
        accepted: list[str] = []
        ignored: list[str] = []

        for key, value in result.items():
            if key in declared or module.allow_extra_outputs:
                state.data[key] = value
                accepted.append(key)
            else:
                ignored.append(key)

        missing_outputs = [key for key in module.outputs if key not in result]
        return accepted, ignored, missing_outputs

    def _next_node(
        self, current: str, state: RunState, error_policy: ErrorPolicy
    ) -> str:
        conditional = self.conditional_edges.get(current)
        if conditional is not None:
            route_key = conditional.router(state)
            target = conditional.routes.get(route_key)
            state.emit(current, "route", {"route_key": route_key, "target": target})
            if target is None:
                self._handle_error(
                    state,
                    current,
                    "unknown_route",
                    {"route_key": route_key, "known": sorted(conditional.routes)},
                    error_policy,
                )
                return END
            return target

        targets = self.edges.get(current, [])
        if not targets:
            return END
        if len(targets) > 1:
            self._handle_error(
                state,
                current,
                "ambiguous_edges",
                {"targets": targets},
                error_policy,
            )
            return END
        return targets[0]

    def _handle_error(
        self,
        state: RunState,
        node: str,
        kind: str,
        detail: dict[str, Any],
        error_policy: ErrorPolicy,
    ) -> None:
        error = {"node": node, "kind": kind, "detail": detail}
        state.errors.append(error)
        state.emit(node, "error", error)

        if error_policy == "raise":
            raise GraphRuntimeError(f"{kind} at {node}: {detail}")
        if error_policy in {"stop", "record"}:
            state.status = "failed"


class AgentGraph:
    def __init__(self) -> None:
        self._modules: dict[str, AgentModule] = {}
        self._edges: list[Edge] = []
        self._conditional_edges: dict[str, ConditionalEdge] = {}
        self._start: str | None = None

    def add_module(self, module: AgentModule) -> "AgentGraph":
        if module.name in self._modules:
            raise GraphCompileError(f"duplicate module: {module.name}")
        self._modules[module.name] = module
        return self

    def add_edge(self, source: str, target: str) -> "AgentGraph":
        self._edges.append(Edge(source, target))
        return self

    def add_conditional_edges(
        self, source: str, router: RouteFunc, routes: dict[str, str]
    ) -> "AgentGraph":
        if source in self._conditional_edges:
            raise GraphCompileError(f"duplicate conditional edge source: {source}")
        self._conditional_edges[source] = ConditionalEdge(source, router, routes)
        return self

    def set_start(self, name: str) -> "AgentGraph":
        self._start = name
        return self

    def compile(self, initial_keys: set[str] | None = None) -> CompiledAgentGraph:
        initial_keys = initial_keys or set()
        self._validate_references()
        self._validate_start()
        self._validate_reachability()
        self._validate_dataflow(initial_keys)

        edge_map: dict[str, list[str]] = {name: [] for name in self._modules}
        for edge in self._edges:
            edge_map[edge.source].append(edge.target)

        return CompiledAgentGraph(
            modules=dict(self._modules),
            edges=edge_map,
            conditional_edges=dict(self._conditional_edges),
            start=self._start or "",
        )

    def _validate_start(self) -> None:
        if self._start is None:
            raise GraphCompileError("start node is not set")
        if self._start not in self._modules:
            raise GraphCompileError(f"start node does not exist: {self._start}")

    def _validate_references(self) -> None:
        for edge in self._edges:
            if edge.source not in self._modules:
                raise GraphCompileError(f"edge source does not exist: {edge.source}")
            if edge.target not in self._modules and edge.target != END:
                raise GraphCompileError(f"edge target does not exist: {edge.target}")

        for source, conditional in self._conditional_edges.items():
            if source not in self._modules:
                raise GraphCompileError(f"conditional source does not exist: {source}")
            for route_key, target in conditional.routes.items():
                if target != END and target not in self._modules:
                    raise GraphCompileError(
                        f"route {source}:{route_key} points to unknown target: {target}"
                    )

    def _validate_reachability(self) -> None:
        if self._start is None:
            return

        adjacency: dict[str, set[str]] = {name: set() for name in self._modules}
        for edge in self._edges:
            if edge.target != END:
                adjacency[edge.source].add(edge.target)
        for conditional in self._conditional_edges.values():
            for target in conditional.routes.values():
                if target != END:
                    adjacency[conditional.source].add(target)

        seen: set[str] = set()
        stack = [self._start]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(sorted(adjacency[node] - seen))

        unreachable = sorted(set(self._modules) - seen)
        if unreachable:
            raise GraphCompileError(f"unreachable modules: {unreachable}")

    def _validate_dataflow(self, initial_keys: set[str]) -> None:
        if self._start is None:
            return

        adjacency: dict[str, set[str]] = {name: set() for name in self._modules}
        for edge in self._edges:
            if edge.target != END:
                adjacency[edge.source].add(edge.target)
        for conditional in self._conditional_edges.values():
            for target in conditional.routes.values():
                if target != END:
                    adjacency[conditional.source].add(target)

        available_at: dict[str, set[str]] = {name: set() for name in self._modules}
        available_at[self._start] = set(initial_keys)

        changed = True
        while changed:
            changed = False
            for name, module in self._modules.items():
                available = available_at[name]
                if not set(module.inputs).issubset(available):
                    continue
                produced = available | set(module.outputs)
                for target in adjacency[name]:
                    before = len(available_at[target])
                    available_at[target].update(produced)
                    changed = changed or len(available_at[target]) != before

        failures: list[str] = []
        for name, module in self._modules.items():
            missing = sorted(set(module.inputs) - available_at[name])
            if missing:
                failures.append(
                    f"{name} missing {missing}; available={sorted(available_at[name])}"
                )
        if failures:
            raise GraphCompileError("dataflow validation failed: " + " | ".join(failures))
