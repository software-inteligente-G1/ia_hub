# app/services/genetic_service.py
import random
from app.schemas.genetic_schema import GeneticInput


def fitness(individual):
    return sum(individual)  # Queremos maximizar el número de unos


def mutate(individual, mutation_rate):
    return [
        gene if random.random() > mutation_rate else 1 - gene
        for gene in individual
    ]


def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    return parent1[:point] + parent2[point:]


def select(population):
    return sorted(population, key=fitness, reverse=True)[:2]


def run_genetic_algorithm(config: GeneticInput):
    # Generar población inicial
    population = [
        [random.randint(0, 1) for _ in range(config.gene_length)]
        for _ in range(config.population_size)
    ]

    for _ in range(config.generations):
        parent1, parent2 = select(population)
        children = [mutate(crossover(parent1, parent2), config.mutation_rate)
                    for _ in range(config.population_size)]
        population = children

    best = max(population, key=fitness)
    return {
        "best_solution": best,
        "fitness": fitness(best)
    }
