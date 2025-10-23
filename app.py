
import streamlit as st
import joblib
import PyPDF2
import re

# ============ Load Models ============
model = joblib.load("artifacts/model.pkl")
tfidf = joblib.load("artifacts/tfidf.pkl")
label_encoder = joblib.load("artifacts/label_encoder.pkl")

# ============ Text Cleaning ============
def clean_resume(resume_text):
    resume_text = re.sub(r'http\S+\s*', ' ', resume_text)
    resume_text = re.sub(r'RT|cc', ' ', resume_text)
    resume_text = re.sub(r'#\S+', '', resume_text)
    resume_text = re.sub(r'@\S+', ' ', resume_text)
    resume_text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,./:;<=>?@[\]^_`{|}~"""), ' ', resume_text)
    resume_text = re.sub(r'\s+', ' ', resume_text)
    resume_text = re.sub(r'\d+', ' ', resume_text)
    resume_text = resume_text.lower()
    return resume_text

# ============ PDF Text Extraction ============
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# ============ Streamlit UI ============
st.set_page_config(page_title="AI Resume Screener", page_icon="🧠", layout="centered")

st.title("🧠 AI-Powered Resume Screening System")
st.write("Upload your resume and let AI predict your professional domain instantly!")

uploaded_file = st.file_uploader("📄 Upload Resume (PDF only)", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Analyzing your resume..."):
        resume_text = extract_text_from_pdf(uploaded_file)
        cleaned_text = clean_resume(resume_text)
        features = tfidf.transform([cleaned_text])
        prediction = model.predict(features)[0]
        category = label_encoder.inverse_transform([prediction])[0]

        st.success(f"✅ Predicted Job Category: **{category}**")
        
        st.caption("🎯 Model trained with 99.5% accuracy using TF-IDF + LinearSVC")

st.markdown("---")
st.caption("👨‍💻 Developed by **Aditya Jaiswal** | AI Resume Screener Project")
