# Backlog

Deferred ideas captured during specification. Not part of the current spec scope.

| ID | Idea | Description | Rationale for deferral | Suggested by |
|----|------|-------------|------------------------|--------------|
| BL-001 | Streamlit explorer | Optional `dashboard` subcommand offering an interactive Streamlit view over the persisted Parquet data, for ad-hoc re-slicing beyond the static HTML report. | The static HTML report plus terminal summary covers the core need; a live server is a non-goal for MVP. | llm-bench-core |
| BL-002 | Dedicated multi-model comparison report | A single comparative report (overlaid curves, side-by-side cost/perf table) produced from one invocation across several models. | Multi-model comparison is achievable today by chaining separate runs; a dedicated comparison report is an enhancement, not MVP. | llm-bench-core |
| BL-003 | Native non-OpenAI protocol adapters | Direct support for Anthropic, Vertex, or Bedrock native SDKs without an OpenAI-compatible gateway. | Explicit non-goal; an OpenAI-compatible gateway covers these cases. | llm-bench-core |
| BL-004 | Distributed multi-machine load generation | Coordinated load generation across several machines for very high aggregate load. | Explicit non-goal; single-process asyncio targets the I/O-bound workload. | llm-bench-core |
