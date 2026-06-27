# Config backup

A backup mirror of the live `~/.config/llm-bench/` setup, kept in the repo for safekeeping.

Nothing here is secret: every endpoint and key is a `$ENV:VAR` reference resolved at
runtime from the shell environment (the actual URLs and keys live in `~/.bashrc`, never
in these files). `resolved_config.json` in a run dir always stores `api_key` as `***`.

## Contents

- `config.yaml` - the model registry (IBM ICA gateway, EI gateways, bob-proxy2, local) plus
  run defaults, SLO profiles, and the evaluation block. Some gateway models set
  `send_temperature: false` (e.g. `claude-opus-4-8`, which 400s on the `temperature` param).
- `prompts/quality.yaml` - quality-eval profile: every prompt declares an `expected_output`,
  so the async quality eval scores almost every request.
- `prompts/code-quality.yaml` - the same, for small deterministic coding tasks.

`prompts/short.yaml` and `prompts/long.yaml` are reproducible from `llm-bench init`, so they
are not mirrored here.

## Restore

```sh
cp config/config.yaml ~/.config/llm-bench/config.yaml
mkdir -p ~/.config/llm-bench/prompts
cp config/prompts/*.yaml ~/.config/llm-bench/prompts/
llm-bench init   # idempotent; fills in short.yaml / long.yaml / dashboards if missing
```

The `bob-*` models require the VPN; without it that gateway is unreachable.
