import subprocess

# 模拟一个持续输出的命令（如 ping）
process = subprocess.Popen(
    ['ping', '-c', '4', '8.8.8.8'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

for line in process.stdout:
    print(line)


# # 实时按行读取输出
# while True:
#     output = process.stdout.readline()
#     if output == '' and process.poll() is not None:
#         break
#     if output:
#         print(f"[实时输出] {output.strip()}")

# rc = process.poll()
# print("进程结束，退出码:", rc)

#     def _reader():
#         for line in proc.stdout:
#             output_lines.append(line)
#             if on_output:
#                 on_output(line)
#         proc.wait()
#         # 先更新 cwd，再 set done_event，保证调用方 wait() 返回时 cwd 已就绪
#         _read_cwd_file(cwd_file)
#         done_event.set()