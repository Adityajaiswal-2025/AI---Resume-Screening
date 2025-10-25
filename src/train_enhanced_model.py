import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# 🔹 Step 1: Load dataset
print("📥 Loading Enhanced Resume Dataset...")
data = pd.read_csv("data/EnhancedResumeDataset.csv")

# 🔹 Step 2: Clean text
def clean_resume(resume_text):
    resume_text = re.sub(r'http\S+', ' ', resume_text)
    resume_text = re.sub(r'RT|cc', ' ', resume_text)
    resume_text = re.sub(r'#\S+', '', resume_text)
    resume_text = re.sub(r'@\S+', ' ', resume_text)
    resume_text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,./:;<=>?@[\]^_`{|}~"""), ' ', resume_text)
    resume_text = re.sub(r'\s+', ' ', resume_text)
    resume_text = re.sub(r'\d+', ' ', resume_text)
    return resume_text.lower()

print("🧹 Cleaning resumes...")
data["cleaned_resume"] = data["Resume"].apply(clean_resume)

# 🔹 Step 3: Encode labels
le = LabelEncoder()
y = le.fit_transform(data["Category"])

# 🔹 Step 4: TF-IDF vectorization
print("🔠 Extracting features with TF-IDF (1,2-grams)...")
tfidf = TfidfVectorizer(sublinear_tf=True, stop_words="english", ngram_range=(1,2), max_features=6000)
X = tfidf.fit_transform(data["cleaned_resume"])

# 🔹 Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 🔹 Step 6: Train model
print("🤖 Training optimized LinearSVC model...")
model = LinearSVC(C=2.0, class_weight='balanced', max_iter=2000)
model.fit(X_train, y_train)

# 🔹 Step 7: Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {acc:.3f}\n")
print("📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 🔹 Step 8: Save artifacts
os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/model_enhanced.pkl")
joblib.dump(tfidf, "artifacts/tfidf_enhanced.pkl")
joblib.dump(le, "artifacts/label_encoder_enhanced.pkl")

print("\nAll enhanced model artifacts saved in 'artifacts/' ✅")
