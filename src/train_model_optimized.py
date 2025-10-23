import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# -----------------------------
# 🧹 Step 1: Data Preprocessing
# -----------------------------
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

print("📥 Loading dataset...")
data = pd.read_csv(r"E:\AI-Resume screener\data\UpdatedResumeDataSet.csv")


data['cleaned_resume'] = data['Resume'].apply(clean_resume)

print("🔠 Extracting features with TF-IDF (1,2)...")
tfidf = TfidfVectorizer(stop_words='english', max_features=8000, ngram_range=(1,2))
X = tfidf.fit_transform(data['cleaned_resume'])


le = LabelEncoder()
y = le.fit_transform(data['Category'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("🤖 Training LinearSVC model...")
model = LinearSVC()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n✅ Model Accuracy:", round(accuracy, 3))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
plt.figure(figsize=(10, 8))
disp.plot(xticks_rotation='vertical', cmap='viridis')
plt.title("Confusion Matrix - Resume Classifier")
plt.show()


joblib.dump(model, "src/model.pkl")
joblib.dump(tfidf, "src/tfidf.pkl")
joblib.dump(le, "src/label_encoder.pkl")

print("\n💾 Model, TF-IDF & LabelEncoder saved successfully in src/")
