import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import Column, MetaData, Table, create_engine, func, insert, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateSchema, DropSchema

from forecastops.inference.database import Database, batches, forecasts

BACKENDS = ["sqlite"] + (["postgres"] if os.environ.get("FORECASTOPS_TEST_POSTGRES_URL") else [])


@pytest.fixture(params=BACKENDS)
def db(request, tmp_path):
    if request.param == "sqlite":
        database = Database(f"sqlite:///{tmp_path / 'history.db'}")
        yield database
        database.engine.dispose()
    else:
        url = os.environ["FORECASTOPS_TEST_POSTGRES_URL"]
        admin = create_engine(url)
        schema = "test_" + uuid4().hex
        with admin.begin() as conn:
            conn.execute(CreateSchema(schema))
        database = Database(url)
        database.engine.dispose()
        database.engine = create_engine(url, connect_args={"options": f"-csearch_path={schema}"})
        try:
            yield database
        finally:
            database.engine.dispose()
            with admin.begin() as conn:
                conn.execute(DropSchema(schema, cascade=True))
            admin.dispose()


def row(version="v1", prediction=3.0, item="B"):
    return dict(
        forecast_date=date(2024, 1, 1),
        target_date=date(2024, 1, 2),
        store_id="A",
        item_id=item,
        category="C",
        prediction=prediction,
        lower_bound=0.0,
        upper_bound=10.0,
        model_version=version,
    )


def count(db, table):
    with db.engine.connect() as conn:
        return conn.scalar(select(func.count()).select_from(table))


def legacy_table(db, rows):
    # Snapshot of the old schema: all forecast fields except the new batch ID.
    table = Table(
        "forecasts",
        MetaData(),
        *[
            Column(
                column.name, column.type, primary_key=column.primary_key, nullable=column.nullable
            )
            for column in forecasts.columns
            if column.name != "batch_id"
        ],
    )
    table.create(db.engine)
    with db.engine.begin() as conn:
        conn.execute(insert(table), rows)
    return table


def test_identical_batch_is_idempotent_even_after_reordering(db):
    db.initialize()
    issued = datetime(2024, 1, 1, 23, tzinfo=UTC)
    db.clock = lambda: issued
    first = db.save([row(), row(item="C")])
    issued += timedelta(days=30)
    assert db.save([row(item="C"), row(prediction=3)]) == first
    assert count(db, batches) == 1
    assert count(db, forecasts) == 2
    assert db.query()[0]["created_at"].replace(tzinfo=UTC).day == 1


def test_changed_model_or_values_preserve_all_issued_batches(db):
    db.initialize()
    first = db.save([row()])
    second = db.save([row(version="v2")])
    third = db.save([row(version="v2", prediction=4)])
    assert len({first, second, third}) == 3
    assert count(db, batches) == 3
    assert db.query()[0]["batch_id"] == third
    assert db.query(batch_id=first)[0]["model_version"] == "v1"
    assert db.query(batch_id=second)[0]["prediction"] == 3
    db.save([row()])  # Deliberate rollback changes only the active pointer.
    assert db.query()[0]["batch_id"] == first
    assert count(db, batches) == 3
    assert db.query(batch_id=first, origin=date(2024, 1, 3)) == []


def test_failed_batch_rolls_back_rows_batch_header_and_pointer(db):
    db.initialize()
    first = db.save([row()])
    with pytest.raises(IntegrityError):
        db.save([row(version="v2", item="good"), row(version="v2", prediction=None)])
    assert count(db, batches) == 1
    assert count(db, forecasts) == 1
    assert db.query()[0]["batch_id"] == first


def test_duplicate_series_target_rolls_back_publication(db):
    db.initialize()
    with pytest.raises(IntegrityError):
        db.save([row(), row()])
    assert count(db, batches) == 0
    assert count(db, forecasts) == 0
    assert db.query() == []


def test_concurrent_identical_writers_only_issue_once(db):
    db.initialize()
    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = list(pool.map(lambda _: db.save([row()]), range(2)))
    assert ids[0] == ids[1]
    assert count(db, batches) == 1
    assert count(db, forecasts) == 1


def test_legacy_migration_preserves_timestamps_and_never_reimports(db):
    issued = datetime(2024, 1, 1, 23, tzinfo=UTC)
    legacy = legacy_table(db, [{**row(), "created_at": issued}])
    db.initialize()
    original = db.query()[0]
    assert original["created_at"].replace(tzinfo=UTC) == issued
    with db.engine.connect() as conn:
        assert conn.scalar(select(batches.c.source)) == "legacy_import"
    replacement = db.save([row(version="v2")])
    db.initialize()
    assert db.query()[0]["batch_id"] == replacement
    assert db.query(batch_id=original["batch_id"])[0]["prediction"] == 3
    assert count(db, legacy) == 1
    assert count(db, batches) == 2


def test_migration_failure_leaves_legacy_data_intact_and_can_retry(db):
    issued = datetime(2024, 1, 1, 23, tzinfo=UTC)
    legacy = legacy_table(
        db,
        [
            {**row(), "created_at": issued},
            {**row(version="mixed", item="C"), "created_at": issued},
        ],
    )
    with pytest.raises(ValueError, match="one model version"):
        db.initialize()
    assert count(db, legacy) == 2
    with db.engine.begin() as conn:
        conn.execute(update(legacy).values(model_version="v1"))
    db.initialize()
    assert count(db, forecasts) == 2


def test_latest_origin_does_not_fall_back_for_missing_series(db):
    db.initialize()
    first = db.save([row()])
    later = row(item="new")
    later["forecast_date"] += timedelta(days=1)
    later["target_date"] += timedelta(days=1)
    db.save([later])
    assert db.query(item="B") == []
    assert db.query(item="B", batch_id=first)


def test_mid_migration_failure_rolls_back_and_retries_without_duplicates(db, monkeypatch):
    issued = datetime(2024, 1, 1, 23, tzinfo=UTC)
    later = {**row(), "forecast_date": date(2024, 1, 2), "target_date": date(2024, 1, 3)}
    legacy = legacy_table(
        db,
        [
            {**row(), "created_at": issued},
            {**later, "created_at": issued + timedelta(days=1)},
        ],
    )
    publish = db._publish
    calls = []

    def fail_after_insert(*args):
        publish(*args)
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("migration interrupted")

    monkeypatch.setattr(db, "_publish", fail_after_insert)
    with pytest.raises(RuntimeError, match="interrupted"):
        db.initialize()
    assert count(db, legacy) == 2
    monkeypatch.setattr(db, "_publish", publish)
    db.initialize()
    assert count(db, forecasts) == 2
    assert count(db, batches) == 2
