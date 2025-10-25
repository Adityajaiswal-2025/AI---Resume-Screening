import joblib
import numpy as np

def predict_category(resume_text, model_path, tfidf_path, label_encoder_path):
    """
    Predict the job category for a given resume text.
    """

    model = joblib.load(model_path)
    tfidf_vectorizer = joblib.load(tfidf_path)
    label_encoder = joblib.load(label_encoder_path)

  
    X = tfidf_vectorizer.transform([resume_text])

 
    y_pred = model.predict(X)


    predicted_category = label_encoder.inverse_transform(y_pred)[0]

    return predicted_category

text = cleaned_text.lower()
if ("machine learning" in text or "tensorflow" in text or "pytorch" in text) and "data scientist" not in category.lower():
    category = "Machine Learning Engineer"
elif ("power bi" in text or "tableau" in text or "excel" in text) and "data analyst" not in category.lower():
    category = "Data Analyst"

