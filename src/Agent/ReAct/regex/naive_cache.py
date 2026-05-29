# 对比：加缓存 vs 不加缓存

call_count = 0

def match_slow(pattern, text):
    """原版：不记忆，重复计算"""
    global call_count
    call_count += 1

    if pattern == "":
        return text == ""
    if len(pattern) >= 2 and pattern[1] == '*':
        c = pattern[0]
        rest = pattern[2:]
        if match_slow(rest, text):
            return True
        if text != "" and (c == '.' or c == text[0]):
            return match_slow(pattern, text[1:])
        return False
    if text == "":
        return False
    if pattern[0] == '.' or pattern[0] == text[0]:
        return match_slow(pattern[1:], text[1:])
    return False


def match_memo(pattern, text, cache=None):
    """加了缓存：算过的就不再算"""
    global call_count
    call_count += 1

    if cache is None:
        cache = {}

    key = (pattern, text)
    if key in cache:
        return cache[key]   # 算过了，直接拿结果

    if pattern == "":
        result = text == ""
    elif len(pattern) >= 2 and pattern[1] == '*':
        c = pattern[0]
        rest = pattern[2:]
        if match_memo(rest, text, cache):
            result = True
        elif text != "" and (c == '.' or c == text[0]):
            result = match_memo(pattern, text[1:], cache)
        else:
            result = False
    elif text == "":
        result = False
    elif pattern[0] == '.' or pattern[0] == text[0]:
        result = match_memo(pattern[1:], text[1:], cache)
    else:
        result = False

    cache[key] = result    # 记住这次的结果
    return result


# 用 a*a*a*a*b vs 20 个 a 来对比
text = "a" * 20
pattern = "a*a*a*a*b"

call_count = 0
match_slow(pattern, text)
print(f"不加缓存: {call_count} 次调用")

call_count = 0
match_memo(pattern, text)
print(f"加了缓存: {call_count} 次调用")
