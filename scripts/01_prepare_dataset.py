from pathlib import Path
from jobmatch_nb.paths import RAW_DIR, PROCESSED_DIR
from jobmatch_nb.data.load_data import load_all_data
from jobmatch_nb.data.preprocess import preprocess_jobs
from jobmatch_nb.data.labeler import apply_labeling

def main():
    aistudio_path = RAW_DIR / "aistudio_job.csv"
    heywhale_path = RAW_DIR / "heywhale_job.csv"
    rules_path = Path("jobmatch_nb/data/label_rules.yml")

    df = load_all_data(aistudio_path, heywhale_path)
    df = preprocess_jobs(df)
    df = apply_labeling(df, rules_path)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "jobs_labeled.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"处理完成，输出文件: {out_path}")
    print(df["label"].value_counts())

if __name__ == "__main__":
    main()