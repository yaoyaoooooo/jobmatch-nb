import argparse
import pandas as pd
from pathlib import Path
from jobmatch_nb.paths import MODELS_DIR, PROCESSED_DIR
from jobmatch_nb.model.predict import load_model
from jobmatch_nb.matching.matcher import match_jobs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, required=True, help="简历文本文件路径")
    parser.add_argument("--topk", type=int, default=20, help="返回TopK岗位")
    args = parser.parse_args()

    resume_path = Path(args.resume)
    resume_text = resume_path.read_text(encoding="utf-8")

    model = load_model(MODELS_DIR / "nb_job_classifier.joblib")
    jobs_df = pd.read_csv(PROCESSED_DIR / "jobs_labeled.csv")

    pred_label, prob_map, result_df = match_jobs(model, jobs_df, resume_text, topk=args.topk)

    print("=" * 80)
    print("预测岗位类别:", pred_label)
    print("=" * 80)
    print("类别概率:")
    for k, v in sorted(prob_map.items(), key=lambda x: x[1], reverse=True):
        print(f"{k}: {v:.4f}")

    print("=" * 80)
    print("Top 匹配岗位:")
    show_cols = ["job_title", "company", "city", "salary", "label", "match_score", "similarity", "job_text"]
    existing_cols = [c for c in show_cols if c in result_df.columns]
    print(result_df[existing_cols].to_string(index=False))

if __name__ == "__main__":
    main()