.PHONY: install install-dev test lint typecheck format clean help

# ── Installation ───────────────────────────────────────────────────────────

install:
	pip install -e ".[train]"

install-flash:
	pip install -e ".[train,flash-attn]"

install-inference:
	pip install -e ".[inference]"

install-dev:
	pip install -e ".[all]"

# ── Quality ────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing --cov-report=html

lint:
	ruff check src/ scripts/ tests/

format:
	black src/ scripts/ tests/
	ruff check src/ scripts/ tests/ --fix

typecheck:
	mypy src/ --ignore-missing-imports

# ── Training ───────────────────────────────────────────────────────────────

train-lora-7b:
	python scripts/train.py --config configs/lora_7b.yaml

train-qlora-13b:
	python scripts/train.py --config configs/qlora_13b.yaml

dry-run:
	python scripts/train.py --config configs/lora_7b.yaml --dry-run

# ── Inference ──────────────────────────────────────────────────────────────

merge:
	@echo "Usage: make merge BASE=<model> ADAPTER=<path> OUT=<path>"
	python scripts/merge_adapter.py \
		--base-model $(BASE) \
		--adapter-path $(ADAPTER) \
		--output-path $(OUT)

quantise:
	@echo "Usage: make quantise MODEL=<path> OUT=<path> QUANT=Q4_K_M"
	python scripts/quantise.py \
		--model-path $(MODEL) \
		--output-path $(OUT) \
		--quant-type $(or $(QUANT), Q4_K_M)

benchmark:
	@echo "Usage: make benchmark MODEL=<gguf_path>"
	python scripts/benchmark_inference.py \
		--model-path $(MODEL) \
		--backend llama-cpp \
		--n-runs 10

# ── Cleanup ────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.py[cod]" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage

# ── Help ───────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  LLM Fine-Tuning Pipeline — Available Commands"
	@echo "  ─────────────────────────────────────────────"
	@echo "  install           Install training dependencies"
	@echo "  install-dev       Install all dependencies (train + inference + dev)"
	@echo "  test              Run test suite"
	@echo "  test-cov          Run tests with coverage report"
	@echo "  lint              Run ruff linter"
	@echo "  format            Auto-format with black + ruff --fix"
	@echo "  typecheck         Run mypy type checker"
	@echo "  train-lora-7b     Train with LoRA on 7B model"
	@echo "  train-qlora-13b   Train with QLoRA on 13B model"
	@echo "  dry-run           Validate config without training"
	@echo "  merge             Merge adapter into base model"
	@echo "  quantise          Export merged model to GGUF"
	@echo "  benchmark         Benchmark inference throughput"
	@echo "  clean             Remove cache and temp files"
	@echo ""
