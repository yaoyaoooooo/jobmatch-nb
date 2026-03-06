import re
import yaml
import pandas as pd
from pathlib import Path

def load_rules(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        rules = yaml.safe_load(f)

    compiled = []
    for label, patterns in rules.items():
        compiled.append((label, [re.compile(p, re.I) for p in patterns]))
    return compiled

def label_title(title: str, compiled_rules):
    title = str(title)
    for label, patterns in compiled_rules:
        for p in patterns:
            if p.search(title):
                return label
    return "other"

def apply_labeling(df: pd.DataFrame, rules_path: Path):
    compiled = load_rules(rules_path)
    df = df.copy()
    df["label"] = df["job_title"].apply(lambda x: label_title(x, compiled))
    return df