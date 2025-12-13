from unittest.mock import MagicMock

import pytest


@pytest.mark.asyncio
async def test_get_params(async_client, mock_get_data_loader: MagicMock):
    mock_get_data_loader.get_params.return_value = {
        "mark": {"name": "Марка", "type": "str"},
        "model": {"name": "Модель", "type": "str"},
    }

    response = await async_client.get("/api/v1/params")
    assert response.status_code == 200
    assert "mark" in response.json()


@pytest.mark.asyncio
async def test_get_marks(async_client, mock_get_data_loader):
    """Тест: GET /api/v1/marks возвращает список марок"""
    fake_db = mock_get_data_loader.get_db.return_value
    fake_db.get_marks.return_value = ["Toyota", "BMW"]

    response = await async_client.get("/api/v1/marks")
    assert response.status_code == 200
    assert response.json() == ["Toyota", "BMW"]


@pytest.mark.asyncio
async def test_post_models(async_client, mock_get_data_loader):
    """Тест: POST /api/v1/models возвращает список моделей"""
    fake_db = mock_get_data_loader.get_db.return_value
    fake_db.get_models_car.return_value = ["Camry", "Corolla"]

    response = await async_client.post("/api/v1/models", json={"mark": "Toyota"})
    assert response.status_code == 200
    assert "Camry" in response.text


@pytest.mark.asyncio
async def test_predict_error(async_client, mock_get_data_loader):
    mock_get_data_loader.get_model.return_value.predict.side_effect = Exception()

    payload = {
        "mark": "Toyota",
        "model": "Camry",
        "super_gen_name": "VII",
        "body_type_type": "седан",
        "transmission": "автомат",
        "engine": "бензин",
        "year": 2020,
        "mileage": 50000,
        "color": "черный",
        "owners": "1",
        "region": "Москва",
        "gear_type": "передний",
        "steering_wheel": "левый",
        "complectation": "Комфорт",
        "power": 150,
        "displacement": 2.0,
    }
    response = await async_client.post("/api/v1/predict", json=payload)
    assert response.status_code == 500
