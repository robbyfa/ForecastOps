.PHONY: install test lint demo serve train forecast monitor compose
install:
	uv sync --frozen
test:
	uv run pytest -q
lint:
	uv run ruff check .
	uv run ruff format --check .
demo:
	uv run forecastops demo
serve:
	uv run uvicorn forecastops.api.app:app --host 127.0.0.1 --port 8000
train:
	uv run forecastops train --data $(DATA)
forecast:
	uv run forecastops forecast --data $(DATA)
monitor:
	uv run forecastops monitor --data $(DATA)
compose:
	docker compose up --build -d
