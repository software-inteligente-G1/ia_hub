# app/schemas/naive_bayes_schema.py
from pydantic import BaseModel
from typing import List

class NaiveBayesInput(BaseModel):
    features: List[float]
