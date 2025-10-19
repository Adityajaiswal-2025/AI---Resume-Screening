import os
import joblib

def save_object(obj, filename):
    """
    Save any Python object using joblib.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(obj, filename)
    print(f"✅ Object saved at: {filename}")

def load_object(filename):
    """
    Load a previously saved Python object.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"❌ File not found: {filename}")
    return joblib.load(filename)
