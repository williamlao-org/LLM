import traceback
import contextlib
import httpx

# from dotenv import load_dotenv
import os
import json
import io

# load_dotenv()




from Agent.ReAct.tools import list_files,execute_command

print(list_files())


print(list_files("../workspace"))

print(execute_command(['uname','-r']))