from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PATH = REPO_ROOT / "data" / "features" / "car_features.parquet"

# 1. Entity
car_entity = Entity(name="car_id", join_keys=["car_id"])

# 2. Источник данных
car_source = FileSource(
    name="car_features_source",
    path=str(PROCESSED_PATH),
    timestamp_field="event_timestamp",
)

# 3. Feature View
car_feature_view = FeatureView(
    name="car_features",
    entities=[car_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="owners", dtype=Int64),
        Field(name="year", dtype=Int64),
        Field(name="region", dtype=String),
        Field(name="mileage", dtype=Float32),
        Field(name="mark", dtype=String),
        Field(name="model", dtype=String),
        Field(name="complectation", dtype=String),
        Field(name="steering_wheel", dtype=String),
        Field(name="gear_type", dtype=String),
        Field(name="engine", dtype=String),
        Field(name="transmission", dtype=String),
        Field(name="power", dtype=Float32),
        Field(name="displacement", dtype=Float32),
        Field(name="color", dtype=String),
        Field(name="body_type_type", dtype=String),
        Field(name="super_gen_name", dtype=String),
    ],
    online=True,
    source=car_source,
)
