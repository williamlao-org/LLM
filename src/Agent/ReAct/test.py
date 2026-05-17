import contextlib
import httpx
from dotenv import load_dotenv
import os
import json
import io

load_dotenv()

buffer = io.StringIO()

with contextlib.redirect_stdout(buffer):
    exec('print("Hello World")')

output = buffer.getvalue()

print(repr(output)) 