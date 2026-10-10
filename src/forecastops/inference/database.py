from datetime import UTC, datetime

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    delete,
    insert,
    select,
    text,
)

metadata = MetaData()
forecasts = Table(
    "forecasts",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("forecast_date", Date, nullable=False),
    Column("target_date", Date, nullable=False),
    Column("store_id", String, nullable=False),
    Column("item_id", String, nullable=False),
    Column("category", String, nullable=False),
    Column("prediction", Float, nullable=False),
    Column("lower_bound", Float, nullable=False),
    Column("upper_bound", Float, nullable=False),
    Column("model_version", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("forecast_date", "target_date", "store_id", "item_id"),
)


class Database:
    def __init__(self, url):
        self.engine = create_engine(url, pool_pre_ping=True)

    def initialize(self):
        metadata.create_all(self.engine)

    def healthy(self):
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))

    def save(self, rows):
        if not rows:
            raise ValueError("Refusing to publish an empty batch")
        origin = rows[0]["forecast_date"]
        if any(row["forecast_date"] != origin for row in rows):
            raise ValueError("A batch must have one forecast date")
        now = datetime.now(UTC)
        with self.engine.begin() as conn:
            conn.execute(delete(forecasts).where(forecasts.c.forecast_date == origin))
            conn.execute(insert(forecasts), [{**r, "created_at": now} for r in rows])

    def query(self, item=None, store=None, horizon=14, origin=None, limit=1000, offset=0):
        # Latest batch is global: never silently serve an old batch for a missing item.
        with self.engine.connect() as conn:
            if origin is None:
                origin = conn.execute(
                    select(forecasts.c.forecast_date)
                    .order_by(forecasts.c.forecast_date.desc())
                    .limit(1)
                ).scalar()
            if origin is None:
                return []
            from datetime import timedelta

            stmt = select(forecasts).where(
                forecasts.c.forecast_date == origin,
                forecasts.c.target_date <= origin + timedelta(days=horizon),
            )
            if item:
                stmt = stmt.where(forecasts.c.item_id == item)
            if store:
                stmt = stmt.where(forecasts.c.store_id == store)
            stmt = (
                stmt.order_by(forecasts.c.store_id, forecasts.c.item_id, forecasts.c.target_date)
                .offset(offset)
                .limit(limit)
            )
            return [dict(row) for row in conn.execute(stmt).mappings()]
