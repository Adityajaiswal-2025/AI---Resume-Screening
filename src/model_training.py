from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from src.utils import save_object

def train_model(X, y):
    """
    Train the Logistic Regression model and evaluate it.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"✅ Model Accuracy: {acc*100:.2f}%")
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

    # Save model
    save_object(model, "artifacts/model.pkl")

    return model
