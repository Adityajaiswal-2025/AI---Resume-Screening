import os, re, sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle

ORIG_PATH = r"E:\AI-Resume screener\data\UpdatedResumeDataSet.csv" 
ENH_PATH  = r"data/EnhancedResumeDataset.csv"       
MERGED_OUT = r"data\MergedResumeDataset.csv"
ARTIFACT_DIR = r"artifacts"


def clean_text(t):
    if pd.isna(t): return ""
    s = str(t)
    s = re.sub(r'http\S+', ' ', s)
    s = re.sub(r'[\r\n]+', ' ', s)
    s = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', s)
    s = re.sub(r'\d+', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip().lower()

def main():
    
    if not os.path.exists(ORIG_PATH):
        print("ERROR: Original dataset not found at:", ORIG_PATH)
        sys.exit(1)
    if not os.path.exists(ENH_PATH):
        print("ERROR: Enhanced dataset not found at:", ENH_PATH)
        sys.exit(1)

    
    orig = pd.read_csv(ORIG_PATH)
    enh = pd.read_csv(ENH_PATH)

    
    orig = orig.rename(columns={orig.columns[0]:"Category", orig.columns[1]:"Resume"}) if list(orig.columns[:2])!=["Category","Resume"] else orig
    enh = enh.rename(columns={enh.columns[0]:"Category", enh.columns[1]:"Resume"}) if list(enh.columns[:2])!=["Category","Resume"] else enh

    
    merged = pd.concat([orig[["Category","Resume"]], enh[["Category","Resume"]]], ignore_index=True)
    merged = shuffle(merged, random_state=42).reset_index(drop=True)
    os.makedirs(os.path.dirname(MERGED_OUT), exist_ok=True)
    merged.to_csv(MERGED_OUT, index=False)
    print(f"✅ Merged saved to {MERGED_OUT}  (shape: {merged.shape})")

    print("Cleaning text...")
    merged["cleaned_resume"] = merged["Resume"].apply(clean_text)

    le = LabelEncoder()
    y = le.fit_transform(merged["Category"])

    
    print("Creating TF-IDF (1,2-grams, top 10000 features)")
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=10000, sublinear_tf=True)
    X = tfidf.fit_transform(merged["cleaned_resume"])

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    print("🤖 Training LogisticRegression (class_weight='balanced') ...")
    model = LogisticRegression(max_iter=2000, class_weight='balanced', C=3.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy on test set: {acc:.4f}\n")
    print("📊 Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(ARTIFACT_DIR, "model.pkl"))
    joblib.dump(tfidf, os.path.join(ARTIFACT_DIR, "tfidf.pkl"))
    joblib.dump(le, os.path.join(ARTIFACT_DIR, "label_encoder.pkl"))
    print(f"\n💾 Artifacts saved to {ARTIFACT_DIR}: model.pkl, tfidf.pkl, label_encoder.pkl")

    
    sample_texts = merged["cleaned_resume"].sample(5, random_state=1).tolist()
    print("\n🔎 Sample predictions:")
    for s in sample_texts:
        p = le.inverse_transform(model.predict(tfidf.transform([s])))[0]
        print("-", p, " | ", s[:140].replace("\n"," "))

if __name__ == "__main__":
    main()
