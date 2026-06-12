def find_matching_paren(pattern, start):
    """找到 start 位置的 ( 所对应的 ) 的位置"""
    depth = 1
    i = start + 1
    while i < len(pattern):
        if pattern[i] == "(":
            depth += 1
        elif pattern[i] == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def find_top_level_pipe(pattern):
    """找到第一个不在括号内部的 | 的位置"""
    depth = 0
    for i, ch in enumerate(pattern):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "|" and depth == 0:
            return i
    return -1


def match(pattern: str, text: str) -> bool:
    """判断 pattern 是否能完整匹配 text"""

    # ① 处理 |（只找顶层的）
    pipe_pos = find_top_level_pipe(pattern)
    if pipe_pos != -1:
        left = pattern[:pipe_pos]
        right = pattern[pipe_pos + 1 :]
        return match(left, text) or match(right, text)

    # ② pattern 用完了
    if len(pattern) == 0:
        return len(text) == 0

    # ③ 处理括号分组
    if pattern[0] == "(":
        close = find_matching_paren(pattern, 0)
        group = pattern[1:close]  # 括号里的内容
        rest = pattern[close + 1 :]  # 括号后面的内容

        # 检查括号后面有没有量词
        if rest and rest[0] in ("*", "+", "?"):
            quantifier = rest[0]
            after = rest[1:]  # 量词后面的内容

            if quantifier == "*":
                # 跳过（0次）
                if match(after, text):
                    return True
                # 消耗：group 匹配前 i 个字符，然后 (group)* 继续
                for i in range(1, len(text) + 1):
                    if match(group, text[:i]) and match(pattern, text[i:]):
                        return True
                return False

            elif quantifier == "+":
                # 至少一次，然后变成 *
                for i in range(1, len(text) + 1):
                    if match(group, text[:i]) and match(
                        "(" + group + ")*" + after, text[i:]
                    ):
                        return True
                return False

            elif quantifier == "?":
                # 跳过（0次）或恰好一次
                if match(after, text):
                    return True
                for i in range(1, len(text) + 1):
                    if match(group, text[:i]) and match(after, text[i:]):
                        return True
                return False
        else:
            # 没有量词：group 匹配前缀，rest 匹配剩余
            for i in range(len(text) + 1):
                if match(group, text[:i]) and match(rest, text[i:]):
                    return True
            return False

    # ④ 单字符 + 量词（和之前一样）
    has_quantifier = len(pattern) >= 2 and pattern[1] in ("*", "+", "?")

    if has_quantifier:
        quantifier = pattern[1]
        char_match = len(text) > 0 and (pattern[0] == "." or pattern[0] == text[0])

        if quantifier == "*":
            return match(pattern[2:], text) or (char_match and match(pattern, text[1:]))
        elif quantifier == "+":
            return char_match and match(pattern[0] + "*" + pattern[2:], text[1:])
        elif quantifier == "?":
            return match(pattern[2:], text) or (
                char_match and match(pattern[2:], text[1:])
            )

    # ⑤ 普通字符（和之前一样）
    else:
        if len(text) == 0:
            return False
        if pattern[0] == "." or pattern[0] == text[0]:
            return match(pattern[1:], text[1:])
        return False

    return False


if __name__ == "__main__":
    # 括号 + |
    print(match("c(a|o)t", "cat"))  # True
    print(match("c(a|o)t", "cot"))  # True
    print(match("c(a|o)t", "cut"))  # False

    # 括号 + *
    print(match("(ab)*c", "c"))  # True   (0次)
    print(match("(ab)*c", "abc"))  # True   (1次)
    print(match("(ab)*c", "ababc"))  # True   (2次)
    print(match("(ab)*c", "abbc"))  # False

    # 括号 + +
    print(match("(ab)+c", "abc"))  # True   (1次)
    print(match("(ab)+c", "ababc"))  # True   (2次)
    print(match("(ab)+c", "c"))  # False  (至少1次)

    # 括号 + ?
    print(match("(ab)?c", "c"))  # True   (0次)
    print(match("(ab)?c", "abc"))  # True   (1次)
    print(match("(ab)?c", "ababc"))  # False  (最多1次)

    # 嵌套括号
    print(match("((a|b)c)+", "acbc"))  # True

    # 综合
    print(match("a(bc)*d", "ad"))  # True
    print(match("a(bc)*d", "abcd"))  # True
    print(match("a(bc)*d", "abcbcd"))  # True
