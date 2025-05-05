# app/services/naive_bayes_service.py
import joblib
from app.schemas.naive_bayes_schema import NaiveBayesInput

# Cargar modelo entrenado
model = joblib.load("app/models/naive_bayes_model.pkl")


def predict_naive_bayes(data: NaiveBayesInput) -> int: # Añadir tipo de retorno (buena práctica)
    if model is None:
        # Loggear el error y quizás devolver un error específico
        raise RuntimeError("Modelo Naive Bayes no cargado correctamente.")
    try:      
        prediction_array = model.predict([data.features])

        result = int(prediction_array[0])

        return result # Devolver el int estándar

    except Exception as e:  
        print(f"Error durante la predicción de Naive Bayes: {e}")

        raise ValueError("Error durante la predicción") from e
