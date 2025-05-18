# app/schemas/energy_schema.py
from pydantic import BaseModel
from datetime import date


class EnergyInput(BaseModel):
    home_id: int
    appliance_type: int
    energy_consumption: float  # (opcional si no se usa en input)
    time: int  # Hora en formato 24h (ej. 14 para 2pm)
    date: str  # "YYYY-MM-DD" (no se usará, pero se envía)
    temperature: float
    season: int  # Codificado: 0=winter, 1=spring, etc.
    household_size: int

class EnergyOutput(BaseModel):
    predicted_kwh: float
    daily: float
    weekly: float
    monthly: float
    category: str
