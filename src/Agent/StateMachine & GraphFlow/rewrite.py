from ast import Dict
from typing import Any, Callable, Optional, TypedDict

from attr import dataclass

class State(TypedDict):
    user_input: str
    intent:str
    confidence: float
    tool_result:Optional[Dict[str,Any]]
    draft:str
    answer:str
    rejected:bool


NodeFunc=Callable[[State],None]
Condition=Callable[[State],bool]

@dataclass
class Edge:
    condition:Condition
    target:str

@dataclass
class Node:
    name:str
    func:NodeFunc
    edges:list[Edge]