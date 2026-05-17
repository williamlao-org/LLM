import traceback
import contextlib
import httpx
# from dotenv import load_dotenv
import os
import json
import io

# load_dotenv()

def execute_python(code:str):
    buffer=io.StringIO()
    err_info=''
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code)
    except Exception as e:
        err_info=traceback.format_exc()
    
    if err_info:
        return {
            'ok':False,
            'err':err_info,
            'content':''
        }
    else:
        return {
            'ok':True,
            'err':'',
            'content':buffer.getvalue()
        }

print(execute_python("print(1/0)"))
print(execute_python("print(1/1)"))
