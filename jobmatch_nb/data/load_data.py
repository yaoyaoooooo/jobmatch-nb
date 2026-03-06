from pathlib import Path
import pandas as pd
from jobmatch_nb.data.schema import UNIFIED_COLUMNS

def read_csv_auto(path: Path):
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"无法读取文件: {path}")

def pick_col(df, aliases):
    for c in aliases:
        if c in df.columns:
            return c
    return None

def unify_dataframe(df: pd.DataFrame, source_name: str):
    out = pd.DataFrame()
    for target, aliases in UNIFIED_COLUMNS.items():
        col = pick_col(df, aliases)
        out[target] = df[col].astype(str) if col else ""
    out["source"] = source_name
    return out

def load_all_data(aistudio_path: Path, heywhale_path: Path = None):
    dfs = []
    if aistudio_path.exists():
        dfs.append(unify_dataframe(read_csv_auto(aistudio_path), "aistudio"))
    if heywhale_path and heywhale_path.exists():
        dfs.append(unify_dataframe(read_csv_auto(heywhale_path), "heywhale"))
    if not dfs:
        raise ValueError("没有找到任何原始数据文件")
    return pd.concat(dfs, ignore_index=True)