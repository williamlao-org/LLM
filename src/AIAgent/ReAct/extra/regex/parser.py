# string = "1+2+3"


# class Parser:
#     def __init__(self):
#         self.pos = 0

#     def peek(self):
#         if self.pos < len(string):
#             return string[self.pos]  # 看一眼当前字符，不移动
#         return None

#     def consume(self):
#         ch = string[self.pos]
#         self.pos += 1  # 移动光标
#         return ch  # 返回刚读到的字符
string = "1+(2+2)*4"
pos = 0


def peek():
    if pos < len(string):
        return string[pos]  # 看一眼当前字符，不移动
    return None


def consume():
    global pos
    ch = string[pos]
    pos += 1  # 移动光标
    return ch  # 返回刚读到的字符


def atom():
    if peek() == "(":
        consume()
        result = expr()
        consume()
        return result

    return int(consume())


def term():
    left = atom()

    while peek() == "*":
        consume()
        right = atom()
        left = ("*", left, right)

    return left


def expr():
    left = term()

    while peek() == "+":
        consume()
        right = term()
        left = ("+", left, right)

    return left


result = expr()
print(result)
