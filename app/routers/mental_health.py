# app/routers/mental_health.py
from fastapi import APIRouter
from app.schemas.mental_schema import MentalHealthInput
from app.services.mental_service import get_mental_health_response
from app.logger import logger

router = APIRouter()

@router.post("/respond")
def mental_chat(input_data: MentalHealthInput):
    response = get_mental_health_response(input_data)
    logger.info(f"[MentalHealth] Entrada: {input_data.message} - Respuesta: {response['response']}")
    return response
