import json
import numpy as np

def extract_top_tokens_per_class(model, topn=20):
    vectorizer = model.named_steps["tfidf"]
    nb = model.named_steps["nb"]

    feature_names = np.array(vectorizer.get_feature_names_out())
    classes = nb.classes_
    top_tokens = {}

    for i, cls in enumerate(classes):
        idx = np.argsort(nb.feature_log_prob_[i])[::-1][:topn]
        top_tokens[str(cls)] = feature_names[idx].tolist()

    return top_tokens

def save_top_tokens(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)