from fastapi import FastAPI
from app.api.routers import api_router

api = FastAPI()

api.include_router(api_router)