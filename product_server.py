"""One origin for the built dashboard and API. No model is loaded in this process."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from server import create_app

app = FastAPI(docs_url=None, redoc_url=None)
app.mount('/api', create_app())
assets = Path(__file__).resolve().parent / 'dashboard/dist'
if assets.is_dir():
    app.mount('/', StaticFiles(directory=assets, html=True), name='dashboard')
