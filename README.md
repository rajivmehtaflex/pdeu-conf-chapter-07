# Chapter 7: Deep Agent + Human-in-the-loop

Adds CFO approval interrupt behavior for large penalties.

## Setup

```bash
uv sync
cp .env.example .env
```

Set `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` in `.env`.

## Run

```bash
uv run python main.py --self-check
uv run python main.py --approval-demo
uv run python main.py "Audit the account for Gujarat Steel Corp."
uv run pytest
```
