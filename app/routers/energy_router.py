# app/routers/energy_router.py
from fastapi import APIRouter
from app.schemas.energy_schema import EnergyInput, EnergyOutput
from app.services.energy_service import predict_energy_consumption

router = APIRouter()


@router.post("/predict", response_model=EnergyOutput)
def predict(data: EnergyInput):
    return predict_energy_consumption(data)
