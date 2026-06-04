import pprint
import logging
from logging import Formatter, StreamHandler, getLogger


def get_logger(name: str) -> logging.Logger:
    logger = getLogger(name)

    if not logger.handlers:
        handler = StreamHandler()
        handler.setFormatter(
            Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


logger = get_logger(__name__)

EPS = None

State = int
Symbol = str | None
Transition = dict[State, dict[Symbol, set[State]]]


class Frag:
    def __init__(self, start: State, accept: State):
        self.start = start
        self.accept = accept


class Builder:
    def __init__(self):
        self.transition: Transition = {}
        self._next_state: State = 0

    def _new_state(self) -> State:
        state = self._next_state
        self.transition[state] = {}
        self._next_state += 1
        return state

    def _add_edge(self, frm: State, to: State, char: Symbol = EPS):
        self.transition[frm].setdefault(char, set()).add(to)

    def lit(self, char: Symbol) -> Frag:
        start = self._new_state()
        accept = self._new_state()

        self._add_edge(start, accept, char)

        return Frag(start, accept)

    def concat(self, left: Frag, right: Frag) -> Frag:
        start = self._new_state()
        accept = self._new_state()

        self._add_edge(left.accept, right.start)
        self._add_edge(start, left.start)
        self._add_edge(right.accept, accept)

        return Frag(start, accept)

    def union(self, up: Frag, down: Frag) -> Frag:
        start = self._new_state()
        accept = self._new_state()

        self._add_edge(start, up.start)
        self._add_edge(start, down.start)
        self._add_edge(up.accept, accept)
        self._add_edge(down.accept, accept)

        return Frag(start, accept)

    def star(self, frag: Frag) -> Frag:
        start = self._new_state()
        accept = self._new_state()

        self._add_edge(start, frag.start)
        self._add_edge(frag.accept, accept)
        # 重复读
        self._add_edge(frag.accept, frag.start)
        # 不读
        self._add_edge(start, accept)

        return Frag(start, accept)

    def compile(self, frag: Frag) -> "NFA":
        return NFA(frag.start, frag.accept, self.transition)


class NFA:
    def __init__(self, start: State, accept: State, transition: Transition):
        self.start = start
        self.accept = accept
        self.transition = transition

    def _eps_closure(self, states: set[State]) -> set[State]:
        current_states = set(states)
        stack = list(states)

        while stack:
            state = stack.pop()
            eps_states = self.transition[state].get(EPS, frozenset())

            new_states = eps_states - current_states
            current_states.update(new_states)
            stack.extend(new_states)

        return current_states

    def run(self, text: str) -> bool:
        current_states = self._eps_closure({self.start})

        for char in text:
            # 接收字符，所有状态并进
            nxt_states: set[State] = set()
            for frm_state in current_states:
                to_states = self.transition[frm_state].get(char, ())
                nxt_states.update(to_states)

            # 状态机推导出所有状态
            current_states = self._eps_closure(nxt_states)

        return self.accept in current_states


def printFrag(frag: Frag):
    logger.info(frag.start)
    logger.info(frag.accept)


builder = Builder()
frag_a = builder.lit("a")
frag_b = builder.lit("b")
frag_ab = builder.concat(frag_a, frag_b)
frag_a_or_ab = builder.union(builder.lit("a"), frag_ab)  # a|ab
regex = builder.star(frag_a_or_ab)

nfa = builder.compile(regex)
r = nfa.run("abababab")
print(r)
