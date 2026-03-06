import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from jobmatch_nb.config import MatchConfig
from jobmatch_nb.utils.text_utils import normalize_text

def match_jobs(model, jobs_df: pd.DataFrame, resume_text: str, topk=20):
    cfg = MatchConfig()
    resume_text = normalize_text(resume_text)

    pred_label = model.predict([resume_text])[0]
    proba = model.predict_proba([resume_text])[0]
    class_names = model.classes_
    prob_map = {cls: float(p) for cls, p in zip(class_names, proba)}

    tfidf = model.named_steps["tfidf"]
    job_matrix = tfidf.transform(jobs_df["job_text"])
    resume_vec = tfidf.transform([resume_text])

    sims = cosine_similarity(resume_vec, job_matrix).flatten()

    out = jobs_df.copy()
    out["category_posterior"] = out["label"].map(lambda x: prob_map.get(x, 0.0))
    out["similarity"] = sims
    out["match_score"] = (
        cfg.posterior_weight * out["category_posterior"]
        + cfg.similarity_weight * out["similarity"]
    )

    out = out.sort_values("match_score", ascending=False).head(topk).reset_index(drop=True)

    return pred_label, prob_map, out