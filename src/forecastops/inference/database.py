import hashlib
import json
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

from sqlalchemy import (
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    event,
    insert,
    inspect,
    select,
    text,
    update,
)

metadata = MetaData()
batches = Table(
    "forecast_batches",
    metadata,
    Column("batch_id", String(64), primary_key=True),
    Column("forecast_date", Date, nullable=False),
    Column("model_version", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("source", String, nullable=False),
)
forecasts = Table(
    "forecast_history",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("batch_id", String(64), ForeignKey("forecast_batches.batch_id"), nullable=False),
    Column("forecast_date", Date, nullable=False),
    Column("target_date", Date, nullable=False, index=True),
    Column("store_id", String, nullable=False),
    Column("item_id", String, nullable=False),
    Column("category", String, nullable=False),
    Column("prediction", Float, nullable=False),
    Column("lower_bound", Float, nullable=False),
    Column("upper_bound", Float, nullable=False),
    Column("model_version", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("batch_id", "target_date", "store_id", "item_id"),
    CheckConstraint("0 <= lower_bound AND lower_bound <= prediction AND prediction <= upper_bound"),
)
publications = Table(
    "forecast_publications",
    metadata,
    Column("forecast_date", Date, primary_key=True),
    Column("batch_id", String(64), ForeignKey("forecast_batches.batch_id"), nullable=False),
)
migrations = Table("forecast_schema", metadata, Column("version", Integer, primary_key=True))
FIELDS = (
    "forecast_date",
    "target_date",
    "store_id",
    "item_id",
    "category",
    "prediction",
    "lower_bound",
    "upper_bound",
    "model_version",
)


def _canonical(rows):
    if not rows:
        raise ValueError("Refusing to publish an empty batch")
    rows = [{key: row[key] for key in FIELDS} for row in rows]
    if len({row["forecast_date"] for row in rows}) != 1:
        raise ValueError("A batch must have one forecast date")
    if len({row["model_version"] for row in rows}) != 1:
        raise ValueError("A batch must have one model version")
    for row in rows:
        if not 1 <= (row["target_date"] - row["forecast_date"]).days <= 14:
            raise ValueError("Forecast target must be 1–14 days after its origin")
        for key in ("prediction", "lower_bound", "upper_bound"):
            # Preserve None so a NOT NULL violation tests transaction rollback as usual.
            if row[key] is not None:
                row[key] = float(row[key])
    rows.sort(key=lambda row: (row["store_id"], row["item_id"], row["target_date"]))
    payload = json.dumps(rows, sort_keys=True, default=str, allow_nan=False, separators=(",", ":"))
    return rows, hashlib.sha256(payload.encode()).hexdigest()


class Database:
    def __init__(self, url, *, clock=None):
        self.engine = create_engine(url, pool_pre_ping=True)
        self.clock = clock or (lambda: datetime.now(UTC))
        if self.engine.dialect.name == "sqlite":

            @event.listens_for(self.engine, "connect")
            def foreign_keys(connection, record):
                connection.execute("PRAGMA foreign_keys=ON")
                connection.execute("PRAGMA busy_timeout=30000")

    @contextmanager
    def _writer(self):
        # Serialize migrations/publications even when different workers share the DB.
        with self.engine.begin() as conn:
            if conn.dialect.name == "sqlite":
                conn.exec_driver_sql("BEGIN IMMEDIATE")
            elif conn.dialect.name == "postgresql":
                conn.execute(text("SELECT pg_advisory_xact_lock(741229021)"))
            else:
                raise ValueError("Forecast storage supports SQLite and PostgreSQL")
            yield conn

    def initialize(self):
        with self._writer() as conn:
            metadata.create_all(conn)
            if conn.execute(select(migrations.c.version).where(migrations.c.version == 1)).first():
                return
            if inspect(conn).has_table("forecasts"):
                legacy = Table("forecasts", MetaData(), autoload_with=conn)
                origins = conn.execute(select(legacy.c.forecast_date).distinct()).scalars().all()
                for origin in origins:
                    old = [
                        dict(row)
                        for row in conn.execute(
                            select(legacy).where(legacy.c.forecast_date == origin)
                        ).mappings()
                    ]
                    rows, batch_id = _canonical(old)
                    timestamps = {
                        (row["store_id"], row["item_id"], row["target_date"]): row["created_at"]
                        for row in old
                    }
                    self._publish(
                        conn, rows, batch_id, min(timestamps.values()), "legacy_import", timestamps
                    )
            conn.execute(insert(migrations).values(version=1))

    def healthy(self):
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))

    @staticmethod
    def _publish(conn, rows, batch_id, issued_at, source, timestamps=None):
        exists = conn.execute(
            select(batches.c.batch_id).where(batches.c.batch_id == batch_id)
        ).first()
        origin = rows[0]["forecast_date"]
        if not exists:
            conn.execute(
                insert(batches).values(
                    batch_id=batch_id,
                    forecast_date=origin,
                    model_version=rows[0]["model_version"],
                    created_at=issued_at,
                    source=source,
                )
            )
            conn.execute(
                insert(forecasts),
                [
                    {
                        **row,
                        "batch_id": batch_id,
                        "created_at": timestamps[
                            (row["store_id"], row["item_id"], row["target_date"])
                        ]
                        if timestamps
                        else issued_at,
                    }
                    for row in rows
                ],
            )
        # Reusing a batch only changes the active pointer; issuance times never change.
        result = conn.execute(
            update(publications)
            .where(publications.c.forecast_date == origin)
            .values(batch_id=batch_id)
        )
        if not result.rowcount:
            conn.execute(insert(publications).values(forecast_date=origin, batch_id=batch_id))

    def save(self, rows):
        rows, batch_id = _canonical(rows)
        with self._writer() as conn:
            self._publish(conn, rows, batch_id, self.clock(), "issued")
        return batch_id

    def query(
        self, item=None, store=None, horizon=14, origin=None, limit=1000, offset=0, batch_id=None
    ):
        with self.engine.connect() as conn:
            if batch_id is None:
                if origin is None:
                    origin = conn.execute(
                        select(publications.c.forecast_date)
                        .order_by(publications.c.forecast_date.desc())
                        .limit(1)
                    ).scalar()
                batch_id = conn.execute(
                    select(publications.c.batch_id).where(publications.c.forecast_date == origin)
                ).scalar()
            if batch_id is None:
                return []
            batch_origin = conn.execute(
                select(batches.c.forecast_date).where(batches.c.batch_id == batch_id)
            ).scalar()
            if batch_origin is None or (origin is not None and batch_origin != origin):
                return []
            stmt = select(forecasts).where(
                forecasts.c.batch_id == batch_id,
                forecasts.c.target_date <= batch_origin + timedelta(days=horizon),
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
