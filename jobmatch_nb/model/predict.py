import joblib

def load_model(path):
    return joblib.load(path)

def predict_text(model, text: str):
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    class_names = model.classes_
    prob_map = {cls: float(p) for cls, p in zip(class_names, proba)}
    return pred, prob_map