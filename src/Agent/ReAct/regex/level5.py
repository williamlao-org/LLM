StateId = int
Char = str | None
EPS = None


class Frag:
    def __init__(
        self,
        start_id: StateId,
        accept_id: StateId,
    ) -> None:
        self.start_id = start_id
        self.accept_id = accept_id


class Builder:
    def __init__(self) -> None:
        self.transition: dict[StateId, dict[Char, set[StateId]]] = {}
        self._next_state_id = 0

    def new_state(self) -> StateId:
        state_id = self._next_state_id
        self._next_state_id += 1
        self.transition[state_id] = {}
        return state_id

    def add_edge(self, frm_id: StateId, char: Char, to_id: StateId):
        self.transition[frm_id].setdefault(char, set()).add(to_id)

    def lit(self, char: str) -> Frag:
        start_id = self.new_state()
        accept_id = self.new_state()
        self.add_edge(start_id, char, accept_id)
        return Frag(start_id, accept_id)

    def concat(self, left_frag: Frag, right_frag: Frag) -> Frag:
        self.add_edge(left_frag.accept_id, EPS, right_frag.start_id)
        return Frag(left_frag.start_id, right_frag.accept_id)

    def union(self, left_frag: Frag, right_frag: Frag) -> Frag:
        start_id = self.new_state()
        accept_id = self.new_state()
        self.add_edge(start_id, EPS, left_frag.start_id)
        self.add_edge(start_id, EPS, right_frag.start_id)
        self.add_edge(left_frag.accept_id, EPS, accept_id)
        self.add_edge(right_frag.accept_id, EPS, accept_id)
        return Frag(start_id, accept_id)

    def star(self, frag: Frag) -> Frag:
        start_id = self.new_state()
        accept_id = self.new_state()
        self.add_edge(start_id, EPS, frag.start_id)
        self.add_edge(frag.accept_id, EPS, frag.start_id)
        self.add_edge(frag.accept_id, EPS, accept_id)
        self.add_edge(start_id, EPS, accept_id)

        return Frag(start_id, accept_id)

    def compile(self, frag: Frag) -> "NFA":
        # 拷贝转移表，编译出的 NFA 不再受后续 builder 操作影响
        transition = {
            state: {char: set(targets) for char, targets in edges.items()}
            for state, edges in self.transition.items()
        }
        # transition=copy.deepcopy(self,transition) 无脑且正确
        return NFA(transition, frag.start_id, frag.accept_id)


class NFA:
    def __init__(
        self,
        transition: dict[StateId, dict[Char, set[StateId]]],
        start_id: StateId,
        accept_id: StateId,
    ) -> None:
        self.transition = transition
        self.start_id = start_id
        self.accept_id = accept_id

    def _eps_closure(self, states: set[StateId]) -> set[StateId]:
        closure = set(states)
        stack = list(states)
        while stack:
            state = stack.pop()
            for to_id in self.transition[state].get(EPS, ()):
                if to_id not in closure:
                    closure.add(to_id)
                    stack.append(to_id)
        return closure

    def run(self, text: str) -> bool:
        current_states = self._eps_closure({self.start_id})
        for char in text:
            next_states = set()
            for state in current_states:
                next_states.update(self.transition[state].get(char, ()))

            current_states = self._eps_closure(next_states)
        return self.accept_id in current_states


builder = Builder()
frag_a = builder.lit("a")
frag_ab = builder.concat(builder.lit("a"), builder.lit("b"))
frag_a_or_ab = builder.union(frag_a, frag_ab)
regex = builder.star(frag_a_or_ab)

nfa = builder.compile(regex)
print(nfa.run("ababab"))


class Parser:
    def __init__(self, pattern: str, builder: Builder) -> None:
        self.pattern = pattern
        self.pos = 0  # 当前读到哪个位置
        self.builder = builder

    def peek(self) -> str | None:
        # 偷看当前字符，但不移动游标；到末尾了就返回 None
        if self.pos < len(self.pattern):
            return self.pattern[self.pos]
        return None

    def eat(self) -> str:
        # 吃掉当前字符，游标往后挪一位，并把这个字符返回
        char = self.pattern[self.pos]
        self.pos += 1
        return char

    def atom(self) -> Frag:
        char = self.peek()
        if char == "(":
            self.eat()  # 吃掉 '('
            frag = self.expr()  # 括号里面又是一个完整正则，跳回最外层
            self.eat()  # 吃掉 ')'
            return frag
        else:
            self.eat()  # 吃掉这个普通字符
            return self.builder.lit(char)

    def factor(self) -> Frag:
        frag = self.atom()
        while self.peek() == "*":
            # 为什么用 while 循环而不是一个 if？因为 a** 是合法的，意思是对 a 连续套两层 star。
            self.eat()  # 吃掉 '*'
            frag = self.builder.star(frag)
        return frag

    def term(self) -> Frag:
        frag = self.factor()
        while self.peek() is not None and self.peek() not in ("|", ")"):
            right = self.factor()
            frag = self.builder.concat(frag, right)
        return frag

    def expr(self) -> Frag:
        frag = self.term()
        while self.peek() == "|":
            self.eat()  # 吃掉 '|'
            right = self.term()
            frag = self.builder.union(frag, right)
        return frag


def compile_regex(pattern: str) -> NFA:
    builder = Builder()
    parser = Parser(pattern, builder)
    frag = parser.expr()
    return builder.compile(frag)


nfa = compile_regex("(a|ab)*")
print(nfa.run("ababab"))  # True
print(nfa.run("aab"))  # True
print(nfa.run("ba"))  # False
print(nfa.run(""))  # True

# nfa = compile_regex("a|bc")
# print(nfa.run("a"))    # True
# print(nfa.run("bc"))   # True
# print(nfa.run("ab"))   # False
# print(nfa.run("b"))    # False
