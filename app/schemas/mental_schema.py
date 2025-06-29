# app/schemas/mental_schema.py
from pydantic import BaseModel


class MentalHealthInput(BaseModel):
    message: str  # Mensaje del usuario en español
