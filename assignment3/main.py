from app.main import app
from .api import router

app.include_router(router)
