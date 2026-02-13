export PYTHONPATH=$(PWD)/src:$(PWD)/rust

all: lint test

clean:
	rm -f .coverage
	rm -rf .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

format:
	ruff format src
	ruff check src --select I --fix  # optimizes imports

install:
	uv pip install .
	maturin develop --release

install-test: install
	uv pip install .[test]

lint: lint-ruff lint-mypy

lint-ruff:
	ruff check src
	ruff format --check src

lint-mypy:
	mypy src
	mypy tests

test:
	pytest tests --cov src
