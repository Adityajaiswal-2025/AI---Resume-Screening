import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import os
import joblib

def create_features_and_labels(csv_path, save_dir='artifacts'):
  
    df = pd.read_csv(csv_path)


    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform(df['cleaned_resume']).toarray()

   
    le = LabelEncoder()
    y = le.fit_transform(df['Category'])

 
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(tfidf, os.path.join(save_dir, 'tfidf_vectorizer.pkl'))
    joblib.dump(le, os.path.join(save_dir, 'label_encoder.pkl'))

    print(f"✅ TF-IDF & LabelEncoder saved in: {save_dir}")
    return X, y, tfidf, le
