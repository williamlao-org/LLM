from .file_tools import (
    list_files_tool,
    read_file_tool,
    write_file_tool,
    edit_file_tool,
)
from .command_tools import execute_python_tool, execute_command_tool
from .web_tools import web_search_tool, http_request_tool

tools = [
    # tools about file
    list_files_tool,
    read_file_tool,
    write_file_tool,
    edit_file_tool,
    # execute tools
    execute_python_tool,
    execute_command_tool,
    # web tools
    web_search_tool,
    http_request_tool,
]
