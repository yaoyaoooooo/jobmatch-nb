import pandas as pd
from jobmatch_nb.paths import PROCESSED_DIR, MODELS_DIR, REPORTS_DIR
from jobmatch_nb.model.predict import load_model
from jobmatch_nb.model.train import train_model
from jobmatch_nb.model.evaluate import evaluate_model, save_metrics
from jobmatch_nb.model.explain import extract_top_tokens_per_class, save_top_tokens

def main():
    data_path = PROCESSED_DIR / "jobs_labeled.csv"
    df = pd.read_csv(data_path)

    model_path = MODELS_DIR / "nb_job_classifier.joblib"
    if model_path.exists():
        model = load_model(model_path)
        _, X_train, X_test, y_train, y_test = train_model(df)
    else:
        model, X_train, X_test, y_train, y_test = train_model(df)

    metrics, cm_df = evaluate_model(model, X_test, y_test)
    top_tokens = extract_top_tokens_per_class(model, topn=20)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, REPORTS_DIR / "metrics.json")
    cm_df.to_csv(REPORTS_DIR / "confusion_matrix.csv", encoding="utf-8-sig")
    save_top_tokens(top_tokens, REPORTS_DIR / "top_tokens_per_class.json")

    print("评估完成。")
    print("Accuracy:", metrics["accuracy"])
    print("Macro F1:", metrics["macro_f1"])

if __name__ == "__main__":
    main()