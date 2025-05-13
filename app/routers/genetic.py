# app/routers/genetic.py
from fastapi import APIRouter
from app.schemas.genetic_schema import GeneticInput
from app.services.genetic_service import run_genetic_algorithm
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/run")
def run_algorithm(config: GeneticInput):
    result = run_genetic_algorithm(config)
    logger.info(f"[Genetic] Entrada: {config.dict()} - Resultado: {result}")
    return result


