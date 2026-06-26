"""Module entry point so ``python -m llm_bench`` runs the CLI.

Used by the report server's run launcher to spawn a benchmark as a subprocess.
"""

from __future__ import annotations

from llm_bench.llm_bench import app

if __name__ == "__main__":
    app()
