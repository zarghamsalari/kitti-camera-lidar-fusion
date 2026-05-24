.PHONY: install test lint format lidar-projection lidar-frustum demo clean

PY ?= python

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"
	pre-commit install

test:
	pytest

lint:
	ruff check src tests
	ruff format --check src tests
	mypy src

format:
	ruff format src tests
	ruff check --fix src tests

lidar-projection:
	$(PY) scripts/run_projection_demo.py --config configs/kitti_lidar.yaml

lidar-frustum:
	$(PY) scripts/run_frustum_demo.py --config configs/kitti_lidar.yaml

demo:
	streamlit run streamlit_app/app.py

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
