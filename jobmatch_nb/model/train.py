import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from jobmatch_nb.utils.text_utils import chinese_tokenizer
from jobmatch_nb.config import TextConfig, TrainConfig

def build_pipeline():
    text_cfg = TextConfig()
    train_cfg = TrainConfig()

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            tokenizer=chinese_tokenizer,
            lowercase=False,
            ngram_range=text_cfg.ngram_range,
            min_df=text_cfg.min_df,
            max_df=text_cfg.max_df,
            max_features=text_cfg.max_features
        )),
        ("nb", MultinomialNB(alpha=train_cfg.alpha))
    ])
    return pipeline

def train_model(df: pd.DataFrame):
    train_cfg = TrainConfig()

    label_counts = df["label"].value_counts()
    keep_labels = label_counts[label_counts >= train_cfg.min_samples_per_class].index.tolist()
    df = df[df["label"].isin(keep_labels)].copy()

    X = df["job_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=train_cfg.test_size,
        random_state=train_cfg.random_state,
        stratify=y
    )

    pipeline = build_pipeline()

    if train_cfg.do_grid_search:
        param_grid = {
            "nb__alpha": [0.3, 0.5, 0.8, 1.0, 1.2],
            "tfidf__min_df": [2, 3, 5]
        }
        grid = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=3,
            scoring="f1_macro",
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
    else:
        model = pipeline.fit(X_train, y_train)

    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test

def save_model(model, path):
    joblib.dump(model, path)