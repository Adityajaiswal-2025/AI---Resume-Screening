from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

def create_features_and_labels(df):
    """
    Perform TF-IDF vectorization and label encoding.
    """
    tfidf = TfidfVectorizer(sublinear_tf=True, stop_words='english', max_features=3000)
    X = tfidf.fit_transform(df['cleaned_resume'])
    
    le = LabelEncoder()
    y = le.fit_transform(df['Category'])
    
    return X, y, tfidf, le
