.PHONY: install test lint clean build run preprocess inference

install:
	pip install -e .

test:
	pytest

lint:
	flake8 src tests
	mypy src tests
	black src tests --check

format:
	black src tests

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:
	docker-compose build

run:
	docker-compose up score_model

preprocess:
	docker-compose run --rm preprocess \
		--input data/raw/input.json \
		--output data/processed/output.json \
		--task MATH

inference:
	docker-compose run --rm inference \
		--model-path outputs/score_model.bin \
		--input data/test/questions.json \
		--output outputs/results.json \
		--task MATH 