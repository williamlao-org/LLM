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


# builder = Builder()
# frag_a = builder.lit("a")
# frag_b = builder.lit("b")
# frag_ab = builder.concat(frag_a, frag_b)
# frag_a_or_ab = builder.union(builder.lit("a"), frag_ab)  # a|ab
# regex = builder.star(frag_a_or_ab)

# nfa = builder.compile(regex)
# r = nfa.run("ababababaab")
# print(r)


# (a|ab)*ab
# lit -> star -> concat -> union
class Parser:
    def __init__(self, pattern: str, builder: Builder):
        self.pattern = pattern
        self.builder = builder
        self.pos = 0

    def peek(self) -> str | None:
        if self.pos < len(self.pattern):
            return self.pattern[self.pos]
        return None

    def advance(self) -> str:
        ch = self.pattern[self.pos]
        self.pos += 1
        return ch

    # lit
    def atom(self) -> Frag:
        # 有括号
        if self.peek() == "(":
            self.advance()
            frag = self.expr()
            self.advance()  # 吃掉 ")"
            return frag

        # 没括号
        char = self.advance()
        if not (char.isascii() and char.isalpha()):
            raise ValueError(f"Unexpected char: {char} at pos {self.pos - 1}")

        frag = self.builder.lit(char)
        return frag

    # star
    def factor(self) -> Frag:  # star
        frag = self.atom()

        while self.peek() == "*":
            self.advance()
            frag = self.builder.star(frag)

        return frag

    # concat
    def term(self) -> Frag:
        left = self.factor()

        char = self.peek()
        while isinstance(char, str) and char.isascii() and char.isalpha():
            # concat 没消耗任何符号，不需要advance

            right = self.factor()
            left = self.builder.concat(left, right)

            char = self.peek()

        return left

    # union
    def expr(self) -> Frag:
        up = self.term()

        while self.peek() == "|":
            self.advance()
            down = self.term()
            up = self.builder.union(up, down)

        return up

    def parse(self) -> Frag:
        regex = self.expr()

        return regex


def compile_regex(pattern: str) -> NFA:
    builder = Builder()
    parser = Parser(pattern, builder)
    regex = parser.parse()

    nfa = builder.compile(regex)
    return nfa


pattern = "(a|ab)*"
nfa = compile_regex(pattern)

r = nfa.run("ababaaababb")
print(r)
