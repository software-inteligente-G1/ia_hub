# app/main.py
from fastapi import FastAPI
from app.routers import genetic
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="IA Project Hub")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # O especifica ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir las rutas
# app.include_router(naive_bayes.router,
#                    prefix="/naive-bayes", tags=["Naive Bayes"])
app.include_router(genetic.router, prefix="/genetic",
                   tags=["Genetic Algorithm"])
