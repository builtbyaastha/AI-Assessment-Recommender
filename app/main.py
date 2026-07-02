from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from app.api.routes import router
from app.catalog.catalog_loader import get_catalog
from app.llm.client import get_llm_client
from app.retrieval.retriever import Retriever

load_dotenv()
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # build once at startup, not per request - embedding/index setup and
    # LLM client init are too slow to redo inside a 30s request budget
    catalog = get_catalog()
    app.state.catalog = catalog
    app.state.retriever = Retriever(catalog)
    app.state.llm = get_llm_client()
    logging.getLogger(__name__).info("startup complete, catalog size: %d", len(catalog))
    yield


app = FastAPI(title="SHL Assessment Recommender", lifespan=lifespan)
app.include_router(router)
