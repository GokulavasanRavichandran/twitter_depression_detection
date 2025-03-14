import sys
import os
import joblib

def load_model(model_path):
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        sys.exit(1)
    return joblib.load(model_path)

def predict(model, data):
    return model.predict(data)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict.py <model_path> <data>")
        sys.exit(1)

    model_path = sys.argv[1]
    data = sys.argv[2]  # This should be preprocessed data

    model = load_model(model_path)
    prediction = predict(model, [data])
    print(f"Prediction: {prediction}")
