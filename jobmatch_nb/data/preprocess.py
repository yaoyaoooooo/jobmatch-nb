import pandas as pd
from jobmatch_nb.utils.text_utils import normalize_text, join_fields

def preprocess_jobs(df: pd.DataFrame):
    df = df.copy()

    for col in ["job_title", "job_text", "city", "salary", "exp", "edu", "company", "company_type", "industry"]:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("").map(normalize_text)

    df["job_text"] = df.apply(
        lambda row: join_fields(
            row.get("job_title", ""),
            row.get("job_text", ""),
            row.get("exp", ""),
            row.get("edu", "")
        ),
        axis=1
    )

    df = df[df["job_text"].str.len() >= 10].copy()
    df = df.drop_duplicates(subset=["job_title", "job_text", "company"]).reset_index(drop=True)
    return df