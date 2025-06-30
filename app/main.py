# app/main.py
from fastapi import FastAPI
from app.routers import genetic, naive_bayes, energy_router, mental_health, image_classifier_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="IA Project Hub")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React con Vite
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir las rutas
app.include_router(naive_bayes.router,
                   prefix="/naive-bayes", tags=["Naive Bayes"])
app.include_router(genetic.router, prefix="/genetic",
                   tags=["Genetic Algorithm"])
app.include_router(energy_router.router, prefix="/energy",
                   tags=["Energy Prediction"])
app.include_router(mental_health.router, prefix="/mental",
                   tags=["Mental Health NLP"])
app.include_router(image_classifier_router.router, prefix="/fruit", 
                   tags=["Fruit"])
