from Agent.ReAct.tools import list_files_tool, execute_command_tool

print(list_files_tool.func("../workspace"))


print(execute_command_tool.func("uname -r"))
