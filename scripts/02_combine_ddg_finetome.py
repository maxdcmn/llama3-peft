import json
import os
import random

from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

DDG_FILE = "data/ddg-search.jsonl"
REPO_NAME = "finetome-ddg-tool"
SEED = 553499234
ROLE_MAP = {"human": "user", "gpt": "assistant"}


def normalize_finetome(ex):
    return {
        "conversations": [
            {
                "role": ROLE_MAP.get(msg.get("from"), msg.get("from", "user")),
                "content": msg.get("value", ""),
            }
            for msg in ex["conversations"]
        ],
        "tools": None,
    }


def main():
    random.seed(SEED)
    token = os.getenv("HF_TOKEN") or exit("HF_TOKEN required")

    api = HfApi(token=token)
    repo_id = f"{api.whoami(token=token)['name']}/{REPO_NAME}"

    base = [
        normalize_finetome(ex)
        for ex in load_dataset("mlabonne/FineTome-100k", split="train")
    ]

    ddg = [json.loads(line) for line in open(DDG_FILE) if line.strip()]

    all_rows = base + ddg
    random.shuffle(all_rows)
    ds = Dataset.from_list(all_rows)
    ds.push_to_hub(repo_id, token=token)
    print(
        f"✓ Pushed {len(base)} FineTome + {len(ddg)} DDG tool-calling examples to {repo_id}"
    )


if __name__ == "__main__":
    main()
