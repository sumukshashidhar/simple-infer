# Minimal makefile for Sphinx documentation

.PHONY: help docs clean test install

help:
	@echo "Available commands:"
	@echo "  docs     - Build documentation"
	@echo "  clean    - Clean build artifacts"
	@echo "  test     - Run tests"
	@echo "  install  - Install package in development mode"

docs:
	uv run sphinx-build -b html docs docs/_build/html

clean:
	rm -rf docs/_build/
	rm -rf dist/
	rm -rf *.egg-info/

test:
	uv run pytest tests/ -v

install:
	uv sync --extra test --extra docs