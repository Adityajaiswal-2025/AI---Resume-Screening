import streamlit as st
import joblib
import re
import PyPDF2

# Load trained model + vectorizer + encoder
model = joblib.load("artifacts/model.pkl")
tfidf = joblib.load("artifacts/tfidf.pkl")
label_encoder = joblib.load("artifacts/label_encoder.pkl")

# Text cleaning function
def clean_resume(text):
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# Streamlit App UI
st.set_page_config(page_title="AI Resume Screener", layout="centered")
st.title("🤖 AI-Powered Resume Screening System")
st.write("Upload your resume and get instant predicted job domain! 🚀")

uploaded_file = st.file_uploader("Upload Resume (PDF only)", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Analyzing your resume... ⏳"):
        resume_text = extract_text_from_pdf(uploaded_file)
        cleaned_text = clean_resume(resume_text)
        features = tfidf.transform([cleaned_text])
        prediction = model.predict(features)[0]
        job_category = label_encoder.inverse_transform([prediction])[0]

    st.success(f"✅ **Predicted Job Category:** {job_category}")
    st.info("Try uploading another resume!")

st.markdown("---")
st.caption("Developed by **Aditya Jaiswal** • AI Resume Screener Project 🧑‍💻")
