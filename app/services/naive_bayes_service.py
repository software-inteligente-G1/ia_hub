# app/services/naive_bayes_service.py
import joblib
from app.schemas.naive_bayes_schema import NaiveBayesInput

# Cargar modelo entrenado
model = joblib.load("app/models/naive_bayes_model.pkl")


def predict_naive_bayes(data: NaiveBayesInput):
    prediction = model.predict([data.features])
    return prediction[0]
