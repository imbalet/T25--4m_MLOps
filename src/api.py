# noqa: B008
import json
import time
from contextlib import asynccontextmanager
from typing import Annotated

import joblib
import numpy as np
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    Summary,
    generate_latest,
)
from pydantic import BaseModel

from .tools.decoder import apply_encoders_and_scaler
from .tools.json_db import Db

# Пути к моделям и данным
MODEL_PATH = "models/final_xgboost_model.pkl"
ENCODERS_PATH = "models/encoders.pkl"
SCALER_PATH = "models/scaler.pkl"
LABELS_PATH = "src/files/labels.json"
DATA_PATH = "src/files/d.json"
PARAMS_PATH = "src/files/parameters.json"


# ------------------------------
# Класс загрузки данных и моделей
# ------------------------------
class DataLoader:
    def __init__(
        self,
        model_path: str,
        encoders_path: str,
        scaler_path: str,
        labels_path: str,
        data_path: str,
        params_path: str,
    ) -> None:
        self.model_path = model_path
        self.encoders_path = encoders_path
        self.scaler_path = scaler_path
        self.labels_path = labels_path
        self.data_path = data_path
        self.params_path = params_path

        self.db: Db | None = None
        self.model = None
        self.encoders = None
        self.scaler = None
        self.params = None

    def load_all(self):
        """Загружает все ресурсы (один раз при запуске)"""
        self.db = Db(self.labels_path, self.data_path)
        self.model = joblib.load(self.model_path)
        self.encoders = joblib.load(self.encoders_path)
        self.scaler = joblib.load(self.scaler_path)
        with open(self.params_path, "r", encoding="utf-8") as f:
            self.params = json.load(f)

    def get_db(self) -> Db:
        return self.db  # type: ignore

    def get_params(self):
        return self.params

    def get_encoders(self):
        return self.encoders

    def get_scaler(self):
        return self.scaler

    def get_model(self):
        return self.model


# ------------------------------
# Функции зависимостей
# ------------------------------
def get_data_loader(request: Request) -> DataLoader:
    return request.app.state.data_loader


def get_db(loader: Annotated[DataLoader, Depends(get_data_loader)]) -> Db:
    return loader.get_db()


def get_params(loader: Annotated[DataLoader, Depends(get_data_loader)]):
    return loader.get_params()


def get_model(loader: Annotated[DataLoader, Depends(get_data_loader)]):
    return loader.get_model()


def get_scaler(loader: Annotated[DataLoader, Depends(get_data_loader)]):
    return loader.get_scaler()


def get_encoders(loader: Annotated[DataLoader, Depends(get_data_loader)]):
    return loader.get_encoders()


# ------------------------------
# Инициализация приложения
# ------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    loader = DataLoader(
        model_path=MODEL_PATH,
        encoders_path=ENCODERS_PATH,
        scaler_path=SCALER_PATH,
        labels_path=LABELS_PATH,
        data_path=DATA_PATH,
        params_path=PARAMS_PATH,
    )
    loader.load_all()
    app.state.data_loader = loader
    yield


REQUEST_COUNT = Counter(
    "request_count_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "http_status"],
)

REQUEST_LATENCY = Histogram("request_latency_seconds", "Latency of HTTP requests in seconds", ["endpoint"])

PREDICTION_DISTRIBUTION = Summary("prediction_price_distribution", "Distribution of predicted car prices")

app = FastAPI(
    title="Car Price Prediction API",
    version="2.0",
    lifespan=lifespan,
)

router = APIRouter(prefix="/api/v1", tags=["Car API"])


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    endpoint = request.url.path
    status_code = response.status_code

    REQUEST_LATENCY.labels(endpoint).observe(process_time)
    REQUEST_COUNT.labels(request.method, endpoint, status_code).inc()

    return response


# ------------------------------
# Модели запросов
# ------------------------------
class CarBase(BaseModel):
    mark: str


class CarModel(CarBase):
    model: str


class CarGeneration(CarModel):
    super_gen_name: str


class PredictRequest(BaseModel):
    mark: str
    model: str
    super_gen_name: str
    body_type_type: str
    transmission: str
    engine: str
    year: int
    mileage: float
    color: str
    owners: str
    region: str
    gear_type: str
    steering_wheel: str
    complectation: str
    power: float
    displacement: float


# ------------------------------
# Эндпоинты
# ------------------------------


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/params")
async def get_params_endpoint(params=Depends(get_params)):
    return JSONResponse(params)


@router.get("/marks")
async def get_marks(db: Annotated[Db, Depends(get_db)]):
    return JSONResponse(db.get_marks())


@router.get("/steering_wheels")
async def get_steering_wheels(db: Annotated[Db, Depends(get_db)]):
    return JSONResponse(db.get_steering_wheel())


@router.get("/owners")
async def get_owners(db: Annotated[Db, Depends(get_db)]):
    return JSONResponse(db.get_owners())


@router.get("/regions")
async def get_regions(db: Annotated[Db, Depends(get_db)]):
    return JSONResponse(db.get_regions())


@router.get("/gear_types")
async def get_gear_types(db: Annotated[Db, Depends(get_db)]):
    return JSONResponse(db.get_gear_type())


@router.get("/colors")
async def get_colors(db: Annotated[Db, Depends(get_db)]):
    return JSONResponse(db.get_colors())


@router.post("/models")
async def get_models(data: CarBase, db: Annotated[Db, Depends(get_db)]):
    return JSONResponse(db.get_models_car(data.mark))


@router.post("/gens")
async def get_gens(data: CarModel, db: Annotated[Db, Depends(get_db)]):
    return JSONResponse(db.get_super_gen_names_car(data.mark, data.model))


@router.post("/bodies")
async def get_bodies(data: CarGeneration, db: Annotated[Db, Depends(get_db)]):
    return JSONResponse(db.get_body_types_car(data.mark, data.model, data.super_gen_name))


@router.post("/complectations")
async def get_complectations(data: CarGeneration, db: Annotated[Db, Depends(get_db)]):
    return JSONResponse(db.get_complectations_car(data.mark, data.model, data.super_gen_name))


@router.post("/transmissions")
async def get_transmissions(data: CarGeneration, db: Annotated[Db, Depends(get_db)]):
    return JSONResponse(db.get_transmissions_car(data.mark, data.model, data.super_gen_name))


@router.post("/engines")
async def get_engines(data: CarGeneration, db: Annotated[Db, Depends(get_db)]):
    return JSONResponse(db.get_engines_car(data.mark, data.model, data.super_gen_name))


@router.post("/years")
async def get_years(data: CarGeneration, db: Annotated[Db, Depends(get_db)]):
    return JSONResponse(db.get_years_car(data.mark, data.model, data.super_gen_name))


@router.post("/predict")
async def predict(
    data: PredictRequest,
    model=Depends(get_model),
    encoders=Depends(get_encoders),
    scaler=Depends(get_scaler),
):
    try:
        processed_df = apply_encoders_and_scaler(data.dict(), encoders=encoders, scaler=scaler)
        X = processed_df.values
        pred_scaled = model.predict(X)

        if hasattr(scaler, "mean_") and "price" in scaler.feature_names_in_:
            temp = np.zeros((1, len(scaler.feature_names_in_)))
            price_idx = list(scaler.feature_names_in_).index("price")
            temp[0, price_idx] = pred_scaled[0]
            temp_real = scaler.inverse_transform(temp)
            price_real = temp_real[0, price_idx]
        else:
            price_real = float(pred_scaled[0])

        PREDICTION_DISTRIBUTION.observe(price_real)

        return JSONResponse({"predicted": float(price_real)})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при предсказании: {e}")


app.include_router(router)
