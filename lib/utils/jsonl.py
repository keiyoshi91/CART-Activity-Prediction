import json
import os
from typing import Dict, List


def dump_jsonl(jsonl: List[Dict], file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as jsonl_file:
        for i in jsonl:
            line = json.dumps(i)  # , separators=(",", ": "), indent=4
            jsonl_file.write(f"{line}\n")


def read_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, "r") as jsonl_file:
        jsonl = [json.loads(i) for i in jsonl_file.readlines()]
    return jsonl
