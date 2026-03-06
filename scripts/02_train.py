import pandas as pd
from jobmatch_nb.paths import PROCESSED_DIR, MODELS_DIR
from jobmatch_nb.model.train import train_model, save_model

def main():
    data_path = PROCESSED_DIR / "jobs_labeled.csv"
    df = pd.read_csv(data_path)

    model, X_train, X_test, y_train, y_test = train_model(df)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "nb_job_classifier.joblib"
    save_model(model, model_path)

    print(f"模型训练完成，已保存到: {model_path}")
    print(f"训练集样本数: {len(X_train)}")
    print(f"测试集样本数: {len(X_test)}")

if __name__ == "__main__":
    main()