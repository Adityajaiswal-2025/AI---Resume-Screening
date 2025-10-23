import pickle
import os

def save_object(obj, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    print(f"💾 Saved: {filename}")

def load_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
