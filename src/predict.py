import joblib
import numpy as np

def predict_category(resume_text, model_path, tfidf_path, label_encoder_path):
    """
    Predict the job category for a given resume text.
    """

    # ✅ Load saved model and objects
    model = joblib.load(model_path)
    tfidf_vectorizer = joblib.load(tfidf_path)
    label_encoder = joblib.load(label_encoder_path)

    # ✅ Transform resume text into features
    X = tfidf_vectorizer.transform([resume_text])

    # ✅ Predict category
    y_pred = model.predict(X)

    # ✅ Decode predicted label
    predicted_category = label_encoder.inverse_transform(y_pred)[0]

    return predicted_category
