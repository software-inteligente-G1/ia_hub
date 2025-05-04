# app/schemas/genetic_schema.py
from pydantic import BaseModel


class GeneticInput(BaseModel):
    gene_length: int         # Tamaño del cromosoma
    population_size: int     # Tamaño de la población
    generations: int         # Número de generaciones
    mutation_rate: float     # Tasa de mutación (0.0 a 1.0)
