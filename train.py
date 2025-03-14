import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)
    return data

def preprocess_data(data):
    print("Columns in the dataset:", data.columns)  # Debugging line
    X = data['text']
    y = data['label']  # Make sure a 'label' column exists in your CSV
    return X, y


def train_model(X, y):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    return model, vectorizer

def save_model(model, vectorizer, model_path='svm_model.pkl', vectorizer_path='vectorizer.pkl'):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train.py <data_file_path> <model_type>")
        sys.exit(1)
    
    data_file_path = sys.argv[1]
    model_type = sys.argv[2]
    
    if model_type != 'SVM':
        print("Currently only SVM model type is supported.")
        sys.exit(1)
    
    data = pd.read_csv(sys.argv[1])
    print("First few rows of the dataset:", data.head())  # Debugging line
    data = load_data(data_file_path)
    X, y = preprocess_data(data)
    model, vectorizer = train_model(X, y)
    save_model(model, vectorizer)
