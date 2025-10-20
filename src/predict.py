from src.utils import load_object
from src.data_preprocessing import clean_resume

def predict_category(resume_text):
    """
    Predict the job category of a given resume text.
    """
    model = load_object("artifacts/model.pkl")
    tfidf = load_object("artifacts/tfidf.pkl")
    label_encoder = load_object("artifacts/label_encoder.pkl")

    cleaned_text = clean_resume(resume_text)
    features = tfidf.transform([cleaned_text])
    prediction = model.predict(features)[0]

    category = label_encoder.inverse_transform([prediction])[0]
    return category
