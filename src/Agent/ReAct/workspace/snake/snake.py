# 贪吃蛇游戏
import turtle
import time
import random

# 设置窗口
wn = turtle.Screen()
wn.title("贪吃蛇 by 李建")
wn.bgcolor("black")
wn.setup(width=600, height=600)
wn.tracer(0)  # 关闭动画

# 蛇头
head = turtle.Turtle()
head.speed(0)
head.shape("square")
head.color("green")
head.penup()
head.goto(0, 0)
head.direction = "stop"

# 食物
food = turtle.Turtle()
food.speed(0)
food.shape("circle")
food.color("red")
food.penup()
food.goto(0, 100)

segments = []

# 分数显示
pen = turtle.Turtle()
pen.speed(0)
pen.color("white")
pen.penup()
pen.hideturtle()
pen.goto(0, 260)
pen.write("分数: 0  最高分: 0", align="center", font=("Courier", 24, "normal"))

score = 0
high_score = 0

def go_up():
    if head.direction != "down":
        head.direction = "up"

def go_down():
    if head.direction != "up":
        head.direction = "down"

def go_left():
    if head.direction != "right":
        head.direction = "left"

def go_right():
    if head.direction != "left":
        head.direction = "right"

def move():
    if head.direction == "up":
        y = head.ycor()
        head.sety(y + 20)
    if head.direction == "down":
        y = head.ycor()
        head.sety(y - 20)
    if head.direction == "left":
        x = head.xcor()
        head.setx(x - 20)
    if head.direction == "right":
        x = head.xcor()
        head.setx(x + 20)

# 键盘绑定
wn.listen()
wn.onkeypress(go_up, "Up")
wn.onkeypress(go_down, "Down")
wn.onkeypress(go_left, "Left")
wn.onkeypress(go_right, "Right")

# 主循环
while True:
    wn.update()
    
    # 边界碰撞检测
    if head.xcor() > 290 or head.xcor() < -290 or head.ycor() > 290 or head.ycor() < -290:
        time.sleep(1)
        head.goto(0, 0)
        head.direction = "stop"
        for seg in segments:
            seg.goto(1000, 1000)
        segments.clear()
        score = 0
        pen.clear()
        pen.write("分数: {}  最高分: {}".format(score, high_score), align="center", font=("Courier", 24, "normal"))
    
    # 吃食物
    if head.distance(food) < 20:
        x = random.randint(-14, 14) * 20
        y = random.randint(-14, 14) * 20
        food.goto(x, y)
        
        new_segment = turtle.Turtle()
        new_segment.speed(0)
        new_segment.shape("square")
        new_segment.color("grey")
        new_segment.penup()
        segments.append(new_segment)
        
        score += 10
        if score > high_score:
            high_score = score
        pen.clear()
        pen.write("分数: {}  最高分: {}".format(score, high_score), align="center", font=("Courier", 24, "normal"))
    
    # 蛇身移动
    for index in range(len(segments)-1, 0, -1):
        x = segments[index-1].xcor()
        y = segments[index-1].ycor()
        segments[index].goto(x, y)
    if len(segments) > 0:
        x = head.xcor()
        y = head.ycor()
        segments[0].goto(x, y)
    
    move()
    
    # 自身碰撞检测
    for seg in segments:
        if seg.distance(head) < 20:
            time.sleep(1)
            head.goto(0, 0)
            head.direction = "stop"
            for seg in segments:
                seg.goto(1000, 1000)
            segments.clear()
            score = 0
            pen.clear()
            pen.write("分数: {}  最高分: {}".format(score, high_score), align="center", font=("Courier", 24, "normal"))
    
    time.sleep(0.1)

wn.mainloop()