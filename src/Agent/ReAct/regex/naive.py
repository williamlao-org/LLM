# 最朴素的起点：逐字符比较，支持 . 和 * 和 |

def match(pattern, text):
    # 模式用完了，文本也得用完才算配上
    if pattern == "":
        return text == ""

    # ---- | 选择：找到 |，切两半，两边都试 ----
    if '|' in pattern:
        idx = pattern.index('|')
        left  = pattern[:idx]     # | 左边
        right = pattern[idx+1:]   # | 右边
        return match(left, text) or match(right, text)

    # ---- c* 重复 ----
    if len(pattern) >= 2 and pattern[1] == '*':
        c = pattern[0]
        rest = pattern[2:]

        if match(rest, text):
            return True
        if text != "" and (c == '.' or c == text[0]):
            return match(pattern, text[1:])
        return False

    # ---- 普通字符 / . ----
    if text == "":
        return False
    if pattern[0] == '.' or pattern[0] == text[0]:
        return match(pattern[1:], text[1:])
    return False

# 试试
print("--- 基本 ---")
print(match("abc", "abc"))      # True
print(match("a.c", "aXc"))      # True
print(match("a*b", "aab"))      # True

print("--- | 选择 ---")
print(match("a|b",   "a"))      # True  -- 选了左边
print(match("a|b",   "b"))      # True  -- 选了右边
print(match("a|b",   "c"))      # False -- 两边都不行
print(match("abc|de", "abc"))   # True
print(match("abc|de", "de"))    # True
print(match("abc|de", "ab"))    # False

print("--- 混合 ---")
print(match("a*|b",  "aaa"))    # True  -- 左边 a* 吃掉 aaa
print(match("a*|b",  "b"))      # True  -- 右边 b
print(match("a|b|c", "c"))      # True  -- 多个 |

