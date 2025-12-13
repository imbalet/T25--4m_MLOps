from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.api import (
    DataLoader,
    app,
    get_data_loader,
)


@pytest.fixture
def mock_get_data_loader():
    return MagicMock(spec=DataLoader)


@asynccontextmanager
async def fake_lifespan(app):
    yield


@pytest.fixture
async def async_client(mock_get_data_loader):
    app.dependency_overrides.update(
        {
            get_data_loader: lambda: mock_get_data_loader,
        }
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()
