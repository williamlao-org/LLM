# 追踪爆炸过程：a*a*b vs "aa"

call_count = 0

def match(pattern, text, depth=0):
    global call_count
    call_count += 1
    indent = "  " * depth
    print(f"{indent}#{call_count} match(\"{pattern}\", \"{text}\")")

    if pattern == "":
        result = text == ""
        print(f"{indent}  → 模式用完, {'✓' if result else '✗'}")
        return result

    if len(pattern) >= 2 and pattern[1] == '*':
        c = pattern[0]
        rest = pattern[2:]

        print(f"{indent}  碰到 {c}*, 先猜 0 次（跳过 {c}*）")
        if match(rest, text, depth + 1):
            return True

        if text != "" and (c == '.' or c == text[0]):
            print(f"{indent}  0 次不行, 吃掉一个 '{text[0]}', {c}* 留着继续猜")
            if match(pattern, text[1:], depth + 1):
                return True

        print(f"{indent}  → {c}* 所有猜法都失败 ✗")
        return False

    if text == "":
        print(f"{indent}  → 文本用完但模式还有 '{pattern}' ✗")
        return False
    if pattern[0] == '.' or pattern[0] == text[0]:
        return match(pattern[1:], text[1:], depth + 1)
    print(f"{indent}  → '{pattern[0]}' != '{text[0]}' ✗")
    return False

print("=" * 50)
print('模式: a*a*b    文本: "aa"')
print("=" * 50)
call_count = 0
result = match("a*a*b", "aa")
print(f"\n结果: {result}, 共调用 {call_count} 次")
