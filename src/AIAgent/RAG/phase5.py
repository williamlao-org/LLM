import json
from phase4_structured_memory import MemoryExtraction

print(json.dumps(MemoryExtraction.model_json_schema(), indent=2))