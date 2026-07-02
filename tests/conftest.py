import os

import pytest
from fastapi.testclient import TestClient

os.environ["LLM_PROVIDER"] = "mock"

from app.main import app  # noqa: E402  (must import after env var is set)


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c
