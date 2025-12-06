import os
from collections import defaultdict

import pandas as pd
import wandb
from dotenv import load_dotenv

load_dotenv()

api = wandb.Api(api_key=os.getenv("WANDB_API_KEY"))
PROJECT = "maxdcmn-kth-royal-institute-of-technology/huggingface"
OUTPUT_DIR = "logs/csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

runs = api.runs(PROJECT)
runs_by_name = defaultdict(list)

for run in runs:
    df = run.history()
    if not df.empty:
        runs_by_name[run.name].append((run.created_at, df))

for name, run_list in runs_by_name.items():
    run_list.sort(key=lambda x: x[0])
    dfs, step_offset = [], 0
    for _, df in run_list:
        df = df.copy()
        if "_step" in df.columns:
            df["_step"] = df["_step"] + step_offset
            step_offset = df["_step"].max() + 1
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    safe_name = name.replace("/", "_").replace(" ", "_")
    merged.to_csv(f"{OUTPUT_DIR}/{safe_name}.csv", index=False)
    print(f"Saved {safe_name}.csv ({len(merged)} rows, {len(run_list)} runs)")
