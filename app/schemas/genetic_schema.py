# app/schemas/genetic_schema.py
from pydantic import BaseModel
from typing import List


class Course(BaseModel):
    name: str
    hours: int


class GeneticInput(BaseModel):
    courses: List[Course]        # Lista de cursos con horas
    hours_target: int            # Horas objetivo a cubrir
    population_size: int         # Tamaño de la población
    generations: int             # Número de generaciones
    mutation_rate: float         # Tasa de mutación (0.0 a 1.0)
