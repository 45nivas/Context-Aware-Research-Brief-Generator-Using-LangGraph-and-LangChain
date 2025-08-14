# Default target
default: help

# Variables
PYTHON := python
PIP := pip
PYTEST := pytest
BLACK := black
ISORT := isort
FLAKE8 := flake8
MYPY := mypy

# Help
.PHONY: help
help: ## Show this help message
	@echo "Research Brief Generator - Makefile Commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Setup and Installation
.PHONY: install
install: ## Install dependencies
	$(PIP) install -r requirements.txt

.PHONY: install-dev
install-dev: ## Install development dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-asyncio pytest-mock pytest-cov black isort flake8 mypy bandit safety

.PHONY: setup
setup: install-dev ## Complete development setup
	cp .env.example .env
	@echo "Please edit .env with your API keys"

# Code Quality
.PHONY: format
format: ## Format code with black and isort
	$(BLACK) app tests
	$(ISORT) app tests

.PHONY: lint
lint: ## Run linting checks
	$(FLAKE8) app tests --max-line-length=100 --ignore=E203,W503
	$(MYPY) app --ignore-missing-imports

.PHONY: check-format
check-format: ## Check code formatting without fixing
	$(BLACK) --check app tests
	$(ISORT) --check-only app tests

.PHONY: quality
quality: check-format lint ## Run all code quality checks

# Testing
.PHONY: test
test: ## Run tests
	$(PYTEST) tests/ -v

.PHONY: test-cov
test-cov: ## Run tests with coverage
	$(PYTEST) tests/ -v --cov=app --cov-report=html --cov-report=term-missing

.PHONY: test-parallel
test-parallel: ## Run tests in parallel
	$(PYTEST) tests/ -v -n auto

.PHONY: test-watch
test-watch: ## Run tests in watch mode
	$(PYTEST) tests/ -v --lf --tb=short -x

# Security
.PHONY: security
security: ## Run security checks
	bandit -r app/ -f json -o bandit-report.json || true
	safety check --json --output safety-report.json || true

# Development
.PHONY: serve
serve: ## Start development server
	$(PYTHON) -m app.api

.PHONY: cli
cli: ## Show CLI help
	$(PYTHON) -m app.cli --help

.PHONY: demo
demo: ## Run a demo research brief
	$(PYTHON) -m app.cli generate "Artificial Intelligence trends in 2024" --depth 2 --user demo_user

# Database
.PHONY: db-init
db-init: ## Initialize database
	$(PYTHON) -c "import asyncio; from app.database import db_manager; asyncio.run(db_manager.init_db())"

.PHONY: db-reset
db-reset: ## Reset database (WARNING: Deletes all data)
	rm -f research_briefs.db
	$(MAKE) db-init

# Deployment
.PHONY: build
build: ## Build distribution packages
	$(PYTHON) -m pip install build
	$(PYTHON) -m build

.PHONY: docker-build
docker-build: ## Build Docker image
	docker build -t research-brief-generator .

.PHONY: docker-run
docker-run: ## Run Docker container
	docker run -p 8000:8000 --env-file .env research-brief-generator

# Documentation
.PHONY: docs
docs: ## Generate documentation
	@echo "API documentation available at http://localhost:8000/docs when server is running"

.PHONY: docs-serve
docs-serve: ## Serve documentation locally
	$(PYTHON) -m app.api &
	@echo "Documentation available at http://localhost:8000/docs"
	@echo "Press Ctrl+C to stop"

# Cleanup
.PHONY: clean
clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/
	rm -f bandit-report.json safety-report.json

# CI/CD Simulation
.PHONY: ci
ci: quality test-cov security ## Run full CI pipeline locally

.PHONY: pre-commit
pre-commit: format quality test ## Run pre-commit checks

# Benchmarking
.PHONY: benchmark
benchmark: ## Run performance benchmarks
	@echo "Running performance benchmarks..."
	$(PYTHON) -c "
import asyncio
import time
from app.models import BriefRequest, DepthLevel
from app.workflow import research_workflow

async def benchmark():
    request = BriefRequest(
        topic='AI benchmark test',
        depth=DepthLevel.QUICK,
        user_id='benchmark_user'
    )
    
    start = time.time()
    # This would need mocked dependencies for actual benchmarking
    print(f'Benchmark setup completed in {time.time() - start:.2f}s')

asyncio.run(benchmark())
"

# Version Management
.PHONY: version
version: ## Show current version
	$(PYTHON) -c "from app import __version__; print(__version__)"

# Environment
.PHONY: env-check
env-check: ## Check environment configuration
	$(PYTHON) -c "from app.config import config; config.validate(); print('✓ Environment configuration is valid')"

# All-in-one commands
.PHONY: dev
dev: install-dev format test serve ## Full development workflow

.PHONY: release
release: clean quality test-cov build ## Prepare for release

# Docker Compose (if using)
.PHONY: up
up: ## Start services with docker-compose
	docker-compose up -d

.PHONY: down
down: ## Stop services with docker-compose
	docker-compose down

.PHONY: logs
logs: ## View service logs
	docker-compose logs -f

# Monitoring
.PHONY: health
health: ## Check service health
	curl -f http://localhost:8000/health || echo "Service not running"

.PHONY: metrics
metrics: ## Show service metrics
	curl -s http://localhost:8000/metrics | python -m json.tool
