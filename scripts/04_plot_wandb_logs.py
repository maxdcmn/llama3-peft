import os

import matplotlib.pyplot as plt
import pandas as pd

CSV_DIR = "logs/csv"
PLOT_DIR = "logs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

groups = {
    "Llama 3.2 3B Different Hyperparameters": {
        "files": [
            "llama-3.2-3B-finetome.csv",
            "llama-3.2-3B-finetome-0.csv",
            "llama-3.2-3B-finetome-1.csv",
            "llama-3.2-3B-finetome-2.csv",
            "llama-3.2-3B-finetome-3.csv",
        ],
        "xlim": (0, 200),
    },
    "Llama 3.2 3B Different Datasets": {
        "files": [
            "llama-3.2-3B-ddg.csv",
            "llama-3.2-3B-finetome.csv",
            "llama-3.2-3B-finetome-ddg.csv",
        ],
        "xlim": (0, 500),
    },
    "Different Model Sizes": {
        "files": [
            "llama-3.2-1B-finetome.csv",
            "llama-3.2-3B-finetome.csv",
            "llama-3.1-8B-finetome.csv",
        ],
        "xlim": (0, 200),
    },
    "Long Training (5 Epochs)": {
        "files": [
            "llama-3.2-3B-finetome.csv",
        ],
        "xlim": (0, 2500),
    },
}

for group_name, config in groups.items():
    plt.figure(figsize=(5, 4))

    for file in config["files"]:
        path = os.path.join(CSV_DIR, file)
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "train/loss" in df.columns and "_step" in df.columns:
                label = file.replace(".csv", "")
                plt.plot(df["_step"], df["train/loss"], label=label, alpha=0.8)

    plt.xlabel("Log Entry")
    plt.ylabel("Train Loss")
    plt.title(group_name)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(config["xlim"])
    plt.tight_layout()

    safe_name = (
        group_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )
    plt.savefig(f"{PLOT_DIR}/{safe_name}.png", dpi=150)
    print(f"Saved {PLOT_DIR}/{safe_name}.png")
