# app/routers/genetic.py
from fastapi import APIRouter
from app.logger import logger
from app.schemas.genetic_schema import GeneticInput
from app.services.genetic_service import run_genetic_algorithm

router = APIRouter()


@router.post("/run")
def run_algorithm(config: GeneticInput):
    result = run_genetic_algorithm(config)
    logger.info(f"[Genetic] Entrada: {config.dict()} - Resultado: {result}")
    return result

