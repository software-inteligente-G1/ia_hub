# app/routers/naive_bayes.py
from fastapi import APIRouter
from app.schemas.naive_bayes_schema import NaiveBayesInput
from app.services.naive_bayes_service import predict_naive_bayes

router = APIRouter()


@router.post("/predict")
def predict(input_data: NaiveBayesInput):
    result = predict_naive_bayes(input_data)
    return {"prediction": result}
