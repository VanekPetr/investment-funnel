.DEFAULT_GOAL := help

SHELL=/bin/bash

UNAME=$(shell uname -s)

uv:
	@curl -LsSf https://astral.sh/uv/install.sh | sh

.PHONY: install
install:  uv ## Install a virtual environment
	@uv sync -vv --frozen


.PHONY: fmt
fmt:  ## Run autoformatting and linting
	@uvx pre-commit run --all-files


.PHONY: test
test: install ## Run tests
	@uv run pytest


.PHONY: clean
clean:  ## Clean up caches and build artifacts
	@git clean -X -d -f


.PHONY: coverage
coverage: install ## test and coverage
	@uv run coverage run --source=funnel/. -m pytest
	@uv run coverage report -m
	@uv run coverage html

	@if [ ${UNAME} == "Darwin" ]; then \
		open htmlcov/index.html; \
	elif [ ${UNAME} == "linux" ]; then \
		xdg-open htmlcov/index.html 2> /dev/null; \
	fi


.PHONY: help
help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort


.PHONY: funnel
funnel: install ## run the funnel app
	@uv run funnel
