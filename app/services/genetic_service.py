# app/services/genetic_service.py
import random
from typing import List, Tuple


def inicializar_poblacion(tamano_poblacion: int, longitud_cromosoma: int) -> List[List[int]]:
    return [[random.randint(0, 1) for _ in range(longitud_cromosoma)] for _ in range(tamano_poblacion)]


def evaluar(cromosoma: List[int], cursos: List[Tuple[str, int]], horas_objetivo: int) -> float:
    total = sum(cursos[i][1]
                for i in range(len(cromosoma)) if cromosoma[i] == 1)
    if total > horas_objetivo:
        return 0
    return 1 / (1 + abs(horas_objetivo - total))


def seleccion(poblacion: List[List[int]], cursos: List[Tuple[str, int]], horas_objetivo: int) -> List[int]:
    torneo = random.sample(poblacion, 2)
    return max(torneo, key=lambda ind: evaluar(ind, cursos, horas_objetivo))


def cruce(padre1: List[int], padre2: List[int]) -> Tuple[List[int], List[int]]:
    punto = random.randint(1, len(padre1) - 1)
    return padre1[:punto] + padre2[punto:], padre2[:punto] + padre1[punto:]


def mutacion(cromosoma: List[int], tasa: float) -> List[int]:
    return [1 - gen if random.random() < tasa else gen for gen in cromosoma]


def decodificar(cromosoma: List[int], cursos: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    return [cursos[i] for i in range(len(cromosoma)) if cromosoma[i] == 1]


def run_genetic_algorithm(config) -> dict:
    cursos = config.courses  # List of dicts: [{name: "Math", hours: 3}, ...]
    # cursos_tuplas = [(curso["name"], curso["hours"]) for curso in cursos]
    cursos_tuplas = [(curso.name, curso.hours) for curso in cursos]

    tamano_poblacion = config.population_size
    generaciones = config.generations
    longitud = len(cursos)
    tasa_mutacion = config.mutation_rate
    horas_objetivo = config.hours_target

    poblacion = inicializar_poblacion(tamano_poblacion, longitud)

    for _ in range(generaciones):
        nueva_poblacion = []

        for _ in range(tamano_poblacion // 2):
            padre1 = seleccion(poblacion, cursos_tuplas, horas_objetivo)
            padre2 = seleccion(poblacion, cursos_tuplas, horas_objetivo)
            hijo1, hijo2 = cruce(padre1, padre2)
            hijo1 = mutacion(hijo1, tasa_mutacion)
            hijo2 = mutacion(hijo2, tasa_mutacion)
            nueva_poblacion.extend([hijo1, hijo2])

        poblacion = nueva_poblacion

        for crom in poblacion:
            if evaluar(crom, cursos_tuplas, horas_objetivo) == 1:
                seleccionados = decodificar(crom, cursos_tuplas)
                return {
                    "exact_solution": True,
                    "selected_courses": [{"name": c[0], "hours": c[1]} for c in seleccionados]
                }

    # Si no se encuentra solución exacta
    mejor = max(poblacion, key=lambda c: evaluar(
        c, cursos_tuplas, horas_objetivo))
    seleccionados = decodificar(mejor, cursos_tuplas)
    return {
        "exact_solution": False,
        "selected_courses": [{"name": c[0], "hours": c[1]} for c in seleccionados]
    }
