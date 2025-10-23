from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def train_model(X, y):
    model = MultinomialNB()
    model.fit(X, y)
    accuracy = accuracy_score(y, model.predict(X))
    
    print(f"✅ Model Trained Successfully! Training Accuracy: {accuracy:.2f}")
    return model
