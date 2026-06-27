"""Tests for the local web server (``llm-bench serve``).

These cover the pure HTML builders, run resolution, and a live round-trip
against a real (ephemeral-port) server, so the serve path is exercised without
opening a browser or blocking on ``serve_forever``.
"""

from __future__ import annotations

import json
import threading
from typing import TYPE_CHECKING
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from llm_bench.prompts import parse_prompts
from llm_bench.serve import (
    JobRegistry,
    ReportServeError,
    RunRequest,
    build_dashboards_page,
    build_prompts_page,
    build_prompts_yaml,
    build_run_command,
    build_run_page,
    estimate_run_seconds,
    iter_runs,
    list_config_models,
    list_prompt_files,
    make_server,
    parse_run_form,
    prompts_to_form,
    read_prompt_file,
    render_dashboard,
    resolve_run,
    safe_prompt_path,
    save_prompt_file,
)

if TYPE_CHECKING:
    from pathlib import Path

_SUMMARY = {
    "run_id": "abc123def456",
    "model": "claude-haiku-4-5",
    "mode": "closed",
    "status": "completed",
    "levels": [
        {
            "level_or_rate": 1,
            "steady_samples": 5,
            "completed": 5,
            "failed": 0,
            "ttft": {"p50": 1.2, "p99": 1.9},
            "tpot": {"p50": 0.008},
            "e2e": {"p50": 2.0, "p99": 2.8},
            "input_tokens": {"p50": 320.0, "mean": 322.0},
            "output_tokens": {"p50": 99.0, "mean": 101.0},
            "system_tok_s": 36.7,
            "rps": 0.5,
            "goodput_attainment": 0.5,
        },
        {
            "level_or_rate": 8,
            "steady_samples": 39,
            "completed": 39,
            "failed": 0,
            "ttft": {"p50": 1.25, "p99": 1.92},
            "tpot": {"p50": 0.007},
            "e2e": {"p50": 1.95, "p99": 3.6},
            "input_tokens": {"p50": 318.0, "mean": 319.0},
            "output_tokens": {"p50": 105.0, "mean": 110.0},
            "system_tok_s": 335.8,
            "rps": 3.5,
            "goodput_attainment": 0.05,
        },
    ],
}


def _make_run(runs_dir: Path, name: str, summary: dict | None = None) -> Path:
    """Create a run directory under ``runs_dir`` with a ``summary.json``."""
    run = runs_dir / name
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps(summary if summary is not None else _SUMMARY), encoding="utf-8")
    return run


def test_iter_runs_newest_first_and_requires_summary(tmp_path: Path) -> None:
    """``iter_runs`` returns summary-bearing dirs newest-name first, skipping others."""
    _make_run(tmp_path, "2026-06-01_10-00-00")
    _make_run(tmp_path, "2026-06-03_10-00-00")
    (tmp_path / "not-a-run").mkdir()  # no summary.json -> ignored
    runs = iter_runs(tmp_path)
    assert [r.name for r in runs] == ["2026-06-03_10-00-00", "2026-06-01_10-00-00"]


def test_iter_runs_absent_dir_is_empty(tmp_path: Path) -> None:
    """A missing runs directory yields an empty list, not an error."""
    assert iter_runs(tmp_path / "nope") == []


def test_resolve_run_by_name_and_path(tmp_path: Path) -> None:
    """A run resolves by bare name under runs_dir and by full path."""
    run = _make_run(tmp_path, "r1")
    assert resolve_run("r1", tmp_path) == run
    assert resolve_run(str(run), tmp_path) == run


def test_resolve_run_missing_raises(tmp_path: Path) -> None:
    """An unknown run name raises ReportServeError naming the lookup dir."""
    with pytest.raises(ReportServeError, match="run not found"):
        resolve_run("ghost", tmp_path)


def test_live_server_home_is_dashboards(tmp_path: Path) -> None:
    """The home page (/) is the Dashboards tab and lists runs + dashboards."""
    ddir = tmp_path / "dashboards"
    ddir.mkdir()
    (ddir / "default.yaml").write_text(_DASH_YAML, encoding="utf-8")
    _make_run(tmp_path, "r1")
    _dash_records(tmp_path / "r1")
    server = make_server(tmp_path, host="127.0.0.1", port=0, dashboards_dir=ddir)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{port}/") as resp:
            index = resp.read().decode("utf-8")
            assert resp.status == 200
        assert ">Dashboards<" in index
        assert "value='r1'" in index  # run picker
        assert "value='default'" in index  # dashboard option, extension stripped
        assert "value='default.yaml'" not in index
        assert "<svg" in index  # the default dashboard renders against the newest run
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Charts + Carbon shell
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Config model listing + Run tab
# ---------------------------------------------------------------------------

_CONFIG_YAML = """
models:
  - name: ibm-haiku
    base_url: $ENV:IBM_ICA_BASE_URL
    model: claude-haiku-4-5
    api_key: $ENV:IBM_ICA_MODEL_KEY
  - name: local
    base_url: http://localhost:8080/v1
    model: local-model
run:
  duration: 10s
  warmup: 2s
  cooldown: 2s
  concurrency_levels: [1, 2, 4]
"""


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(_CONFIG_YAML, encoding="utf-8")
    return path


def test_list_config_models_reads_names_without_env(tmp_path: Path) -> None:
    """Model names are listed from raw YAML even with unresolved $ENV: secrets."""
    config = _write_config(tmp_path)
    models = list_config_models(config)
    assert [m["name"] for m in models] == ["ibm-haiku", "local"]
    assert models[0]["model"] == "claude-haiku-4-5"


def test_list_config_models_none_or_bad(tmp_path: Path) -> None:
    """A missing config yields no models, not an error."""
    assert list_config_models(None) == []
    assert list_config_models(tmp_path / "absent.yaml") == []


def test_build_run_page_lists_models(tmp_path: Path) -> None:
    """The Run tab renders a model picker and the start button."""
    config = _write_config(tmp_path)
    page = build_run_page(config, None, runs_count=0)
    assert "Run a benchmark" in page
    assert "value='ibm-haiku'" in page
    assert "Start run" in page


def test_build_run_page_without_models(tmp_path: Path) -> None:
    """With no config the Run tab guides the user to create one."""
    page = build_run_page(None, None, runs_count=0)
    assert "No models found" in page


# ---------------------------------------------------------------------------
# Run launcher: command building, estimate, job lifecycle
# ---------------------------------------------------------------------------


def test_build_run_command_closed_overrides(tmp_path: Path) -> None:
    """A closed-mode launch carries model, out dir, config, concurrency, tuning."""
    req = RunRequest(
        model="ibm-haiku",
        mode="closed",
        load="1,4",
        duration="30s",
        max_tokens="256",
        temperature="0.2",
        slo_profile="relaxed",
        seed="7",
    )
    cmd = build_run_command(tmp_path / "config.yaml", req, tmp_path / "out")
    after_run = cmd[cmd.index("run") :]
    assert after_run[after_run.index("-m") + 1] == "ibm-haiku"
    assert cmd[cmd.index("--mode") + 1] == "closed"
    assert cmd[cmd.index("--concurrency") + 1] == "1,4"
    assert cmd[cmd.index("--duration") + 1] == "30s"
    assert cmd[cmd.index("--max-tokens") + 1] == "256"
    assert cmd[cmd.index("--temperature") + 1] == "0.2"
    assert cmd[cmd.index("--slo-profile") + 1] == "relaxed"
    assert cmd[cmd.index("--seed") + 1] == "7"
    assert "--request-rate" not in cmd


def test_build_run_command_open_uses_repeated_request_rate(tmp_path: Path) -> None:
    """An open-mode launch repeats --request-rate and never sends --concurrency."""
    req = RunRequest(model="ibm-haiku", mode="open", load="5,20,50", duration="1m")
    cmd = build_run_command(None, req, tmp_path / "out")
    assert cmd[cmd.index("--mode") + 1] == "open"
    assert cmd.count("--request-rate") == 3
    assert "--concurrency" not in cmd


def test_estimate_uses_levels_and_durations(tmp_path: Path) -> None:
    """The estimate scales with the number of levels and per-level duration."""
    config = _write_config(tmp_path)
    # 3 levels * (10 + 2 + 2)s + 5s overhead = 47s.
    assert estimate_run_seconds(config, RunRequest(model="m")) == 47.0
    # Override to a single level and 1s duration: 1 * (1 + 2 + 2) + 5 = 10s.
    assert estimate_run_seconds(config, RunRequest(model="m", load="8", duration="1s")) == 10.0


class _FakeProc:
    """Minimal stand-in for subprocess.Popen exposing poll()."""

    def __init__(self) -> None:
        self.code: int | None = None

    def poll(self) -> int | None:
        return self.code


def test_job_lifecycle_running_then_done(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A job reports running, then done once the process exits with a summary."""
    monkeypatch.setattr("llm_bench.serve.subprocess.Popen", lambda *a, **k: _FakeProc())
    registry = JobRegistry(config_path=None)
    job_id = registry.start(RunRequest(model="ibm-haiku", load="1,2", duration="1s"))
    assert registry.status(job_id)["state"] == "running"

    job = registry._jobs[job_id]
    job["proc"].code = 0
    (job["out_dir"] / "summary.json").write_text("{}", encoding="utf-8")
    done = registry.status(job_id)
    assert done["state"] == "done"
    assert done["run"] == job["out_dir"].name


def test_job_lifecycle_failure_reports_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-zero exit with no summary surfaces as failed with the log tail."""
    monkeypatch.setattr("llm_bench.serve.subprocess.Popen", lambda *a, **k: _FakeProc())
    registry = JobRegistry(config_path=None)
    job_id = registry.start(RunRequest(model="local", load="1"))
    job = registry._jobs[job_id]
    job["proc"].code = 1
    (job["out_dir"] / "launch.log").write_text("environment variable not set: X", encoding="utf-8")
    failed = registry.status(job_id)
    assert failed["state"] == "failed"
    assert "environment variable" in failed["message"]


def test_status_unknown_job() -> None:
    """An unknown job id is reported as unknown, not an error."""
    assert JobRegistry(config_path=None).status("nope") == {"state": "unknown"}


def test_run_tab_and_start_over_http(tmp_path: Path) -> None:
    """The live server serves the Run tab and rejects an unknown model on start."""
    config = _write_config(tmp_path)
    _make_run(tmp_path, "r1")
    server = make_server(tmp_path, config_path=config, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{port}/run") as resp:
            run_tab = resp.read().decode("utf-8")
        assert "value='ibm-haiku'" in run_tab
        req = Request(
            f"http://127.0.0.1:{port}/run/start",
            data=urlencode({"model": "ghost"}).encode("utf-8"),
            method="POST",
        )
        try:
            urlopen(req)
            raised = False
        except Exception as exc:
            raised = True
            assert "unknown model" in exc.read().decode("utf-8")  # type: ignore[attr-defined]
        assert raised
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# (i) tooltips + richer Run form + form parsing
# ---------------------------------------------------------------------------


def test_run_form_has_modes_presets_and_tuning(tmp_path: Path) -> None:
    """The Run form exposes mode, concurrency/rate presets, duration and tuning fields."""
    page = build_run_page(_write_config(tmp_path), None, runs_count=0)
    assert "name='mode' value='closed'" in page
    assert "name='mode' value='open'" in page
    # concurrency presets with the documented defaults pre-checked
    assert "value='1' checked" in page and "value='32' checked" in page
    assert "value='500'" in page  # full preset range present
    assert "name='c_manual'" in page  # manual extra concurrency
    assert "name='r'" in page  # open-mode arrival-rate checkboxes
    assert "value='other'" in page  # duration 'Other'
    assert "name='max_tokens'" in page
    assert "name='temperature'" in page
    assert "name='slo_profile'" in page
    assert "name='seed'" in page
    # field-level (i) tooltips
    assert "Sampling temperature" in page


def test_parse_run_form_closed_merges_presets_and_manual() -> None:
    """Closed-mode parsing merges checked presets with manual values, sorted/deduped."""
    form = {"model": ["ibm-haiku"], "mode": ["closed"], "c": ["4", "1"], "c_manual": ["2, 4 ,16"], "duration": ["30s"]}
    req = parse_run_form(form, {"ibm-haiku"})
    assert req.mode == "closed"
    assert req.load == "1,2,4,16"
    assert req.duration == "30s"


def test_parse_run_form_open_rates_and_other_duration() -> None:
    """Open-mode parsing reads arrival rates and the 'Other' free-text duration."""
    form = {
        "model": ["ibm-haiku"],
        "mode": ["open"],
        "r": ["10", "5"],
        "duration": ["other"],
        "duration_other": ["45s"],
    }
    req = parse_run_form(form, {"ibm-haiku"})
    assert req.mode == "open"
    assert req.load == "5,10"
    assert req.duration == "45s"


def test_parse_run_form_rejects_bad_input() -> None:
    """Unknown model, empty load, non-integer concurrency, and bad duration all raise."""
    with pytest.raises(ReportServeError, match="unknown model"):
        parse_run_form({"model": ["ghost"], "mode": ["closed"], "c": ["1"]}, {"ibm-haiku"})
    with pytest.raises(ReportServeError, match="at least one concurrency"):
        parse_run_form({"model": ["m"], "mode": ["closed"]}, {"m"})
    with pytest.raises(ReportServeError, match="whole number"):
        parse_run_form({"model": ["m"], "mode": ["closed"], "c_manual": ["1.5"]}, {"m"})
    with pytest.raises(ReportServeError, match="invalid duration"):
        parse_run_form({"model": ["m"], "mode": ["closed"], "c": ["1"], "duration": ["nope"]}, {"m"})


# ---------------------------------------------------------------------------
# Prompts tab: file management + run-tab prompts selector
# ---------------------------------------------------------------------------

_VALID_PROMPTS = "- {id: p1, category: general, messages: [{role: user, content: hi}]}\n"


def _prompts_dir(tmp_path: Path) -> Path:
    pdir = tmp_path / "prompts"
    pdir.mkdir()
    (pdir / "short.yaml").write_text(_VALID_PROMPTS, encoding="utf-8")
    return pdir


def test_list_and_read_prompt_files(tmp_path: Path) -> None:
    """Prompt files are listed and read back from the prompts dir."""
    pdir = _prompts_dir(tmp_path)
    (pdir / "long.yaml").write_text(_VALID_PROMPTS, encoding="utf-8")
    assert list_prompt_files(pdir) == ["long.yaml", "short.yaml"]
    assert "id: p1" in read_prompt_file(pdir, "short.yaml")


def test_safe_prompt_path_rejects_traversal(tmp_path: Path) -> None:
    """A name escaping the prompts dir or with bad chars is rejected."""
    pdir = _prompts_dir(tmp_path)
    with pytest.raises(ReportServeError):
        safe_prompt_path(pdir, "../evil")
    with pytest.raises(ReportServeError):
        safe_prompt_path(pdir, "a/b")


def test_save_prompt_validates_then_writes(tmp_path: Path) -> None:
    """Saving a new file validates the YAML, writes it, and appends .yaml."""
    pdir = _prompts_dir(tmp_path)
    saved = save_prompt_file(pdir, "tools", _VALID_PROMPTS)
    assert saved == "tools.yaml"
    assert (pdir / "tools.yaml").is_file()
    # Invalid content is never written.
    with pytest.raises(ReportServeError, match="not a valid prompt set"):
        save_prompt_file(pdir, "broken", "just a string, not a list")
    assert not (pdir / "broken.yaml").exists()


def test_build_prompts_page_has_structured_editor(tmp_path: Path) -> None:
    """The Prompts tab shows the file picker, new-name field, and the card editor."""
    pdir = _prompts_dir(tmp_path)
    page = build_prompts_page(pdir, runs_count=0)
    assert ">Prompts<" in page  # active tab
    assert "Choose a Prompt Profile" in page
    assert "value='short'" in page  # extension stripped in the picker
    assert "value='short.yaml'" not in page
    assert "id='pnew'" in page  # new profile name field
    assert "id='cards'" in page  # the per-prompt card container
    assert "Add prompt" in page
    assert "onclick='savePrompt()'" in page


def test_prompts_to_form_and_back_round_trips(tmp_path: Path) -> None:
    """A file round-trips through prompts_to_form -> build_prompts_yaml unchanged."""
    multimodal = (
        "- id: vis\n"
        "  category: vision\n"
        "  messages:\n"
        "    - role: user\n"
        "      content:\n"
        "        - {type: text, text: hello}\n"
        "        - {type: image_url, image_url: {url: 'data:img'}}\n"
        "- id: tool\n"
        "  category: tool-use\n"
        "  messages: [{role: user, content: go}]\n"
        "  tools: [{type: function, function: {name: f}}]\n"
        "  tool_results: {f: {ok: true}}\n"
    )
    form = prompts_to_form(multimodal)
    assert form[0]["messages"][0]["json"] is True  # multimodal preserved as JSON text
    assert form[1]["tools"]  # tools captured as raw YAML text
    # simulate the JSON wire round-trip the browser performs
    payload = json.loads(json.dumps(form))
    rebuilt = build_prompts_yaml(payload)
    library = parse_prompts(rebuilt, "rebuilt").prompts
    assert [p.id for p in library] == ["vis", "tool"]
    assert isinstance(library[0].messages[0]["content"], list)  # image content restored
    assert library[1].tools and library[1].tool_results


def test_build_prompts_yaml_validation_errors() -> None:
    """Missing id, no messages, and invalid tools YAML all raise clear errors."""
    with pytest.raises(ReportServeError, match="add at least one prompt"):
        build_prompts_yaml([])
    with pytest.raises(ReportServeError, match="id is required"):
        build_prompts_yaml([{"messages": [{"role": "user", "content": "hi"}]}])
    with pytest.raises(ReportServeError, match="at least one non-empty message"):
        build_prompts_yaml([{"id": "x", "messages": [{"role": "user", "content": "  "}]}])
    with pytest.raises(ReportServeError, match="tools must be a list"):
        build_prompts_yaml([{"id": "x", "messages": [{"role": "user", "content": "hi"}], "tools": "{a: 1}"}])


def test_run_tab_prompts_selector_and_parse(tmp_path: Path) -> None:
    """The Run tab offers the prompts files and parsing resolves the chosen path."""
    config = _write_config(tmp_path)
    pdir = _prompts_dir(tmp_path)
    page = build_run_page(config, pdir, runs_count=0)
    assert "name='prompts'" in page
    assert "value='short'" in page  # extension stripped
    assert "built-in default" in page

    # the picker submits a stem; parsing resolves it to the .yaml path
    form = {"model": ["ibm-haiku"], "mode": ["closed"], "c": ["1"], "prompts": ["short"]}
    req = parse_run_form(form, {"ibm-haiku"}, pdir)
    assert req.prompts == str(pdir / "short.yaml")
    # the resolved path flows into the launch argv
    cmd = build_run_command(config, req, tmp_path / "out")
    assert cmd[cmd.index("--prompts") + 1] == str(pdir / "short.yaml")


def test_prompts_save_and_load_over_http(tmp_path: Path) -> None:
    """The live server saves a prompts file and serves it back via /prompts/load."""
    pdir = _prompts_dir(tmp_path)
    server = make_server(tmp_path, host="127.0.0.1", port=0, prompts_dir=pdir)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        save = Request(
            f"http://127.0.0.1:{port}/prompts/save",
            data=urlencode({"file": "extra.yaml", "content": _VALID_PROMPTS}).encode("utf-8"),
            method="POST",
        )
        with urlopen(save) as resp:
            assert json.loads(resp.read())["file"] == "extra.yaml"
        assert (pdir / "extra.yaml").is_file()
        with urlopen(f"http://127.0.0.1:{port}/prompts/load?file=extra.yaml") as resp:
            loaded = json.loads(resp.read())
        assert "id: p1" in loaded["content"]
        assert loaded["prompts"][0]["id"] == "p1"  # structured form payload for the editor
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_prompts_save_from_editor_payload_over_http(tmp_path: Path) -> None:
    """Saving the structured editor payload rebuilds and validates the YAML file."""
    pdir = _prompts_dir(tmp_path)
    server = make_server(tmp_path, host="127.0.0.1", port=0, prompts_dir=pdir)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    payload = json.dumps(
        [
            {
                "id": "edited",
                "category": "coding",
                "isl_bucket": "short",
                "messages": [{"role": "user", "content": "Write a haiku.", "json": False}],
                "expected_output": "",
                "tools": "",
                "tool_results": "",
            }
        ]
    )
    try:
        save = Request(
            f"http://127.0.0.1:{port}/prompts/save",
            data=urlencode({"file": "edited.yaml", "payload": payload}).encode("utf-8"),
            method="POST",
        )
        with urlopen(save) as resp:
            assert json.loads(resp.read())["file"] == "edited.yaml"
        text = (pdir / "edited.yaml").read_text(encoding="utf-8")
        assert "id: edited" in text
        assert parse_prompts(text, "edited").prompts[0].id == "edited"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Dashboards tab
# ---------------------------------------------------------------------------

_DASH_YAML = "- {title: Lat, x: level_or_rate, values: [{metric: e2e, agg: p50}]}\n"


def _dash_records(run_dir: Path) -> None:
    rows = [
        {
            "level_or_rate": 1,
            "e2e": 1.0,
            "output_tokens": 10,
            "prompt_tokens": 100,
            "osl_bucket": "short",
            "phase": "steady",
            "outcome": "success",
        },
        {
            "level_or_rate": 2,
            "e2e": 2.0,
            "output_tokens": 12,
            "prompt_tokens": 100,
            "osl_bucket": "short",
            "phase": "steady",
            "outcome": "success",
        },
    ]
    (run_dir / "raw.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_render_dashboard_produces_chart_and_table() -> None:
    """render_dashboard turns a dashboard + records into a chart and a recap table."""
    records = [
        {"level_or_rate": 1, "e2e": 1.0, "phase": "steady", "outcome": "success"},
        {"level_or_rate": 2, "e2e": 2.0, "phase": "steady", "outcome": "success"},
    ]
    html = render_dashboard(records, _DASH_YAML)
    assert "Lat" in html
    assert "<svg" in html
    assert "<table" not in html  # the per-panel table is gone
    assert "data-tip=" in html  # values surface on click instead


def test_build_dashboards_page_pickers_and_render(tmp_path: Path) -> None:
    """The Dashboards tab shows file/run pickers and renders panels for a chosen pair."""
    ddir = tmp_path / "dashboards"
    ddir.mkdir()
    (ddir / "d.yaml").write_text(_DASH_YAML, encoding="utf-8")
    run = _make_run(tmp_path, "r1")
    _dash_records(run)

    # explicit None,None falls back to the first dashboard + newest run and renders.
    page = build_dashboards_page(ddir, tmp_path, None, None, runs_count=1)
    assert ">Dashboards<" in page
    assert "value='d'" in page  # extension stripped in the picker
    assert "value='d.yaml'" not in page
    assert "value='r1'" in page

    rendered = build_dashboards_page(ddir, tmp_path, "d", "r1", runs_count=1)
    assert "Lat" in rendered
    assert "<svg" in rendered
    assert "Edit this dashboard" in rendered
    # a metrics & dimensions help panel sits beside the editor
    assert "class='helprow'" in rendered
    assert "What do the metrics and dimensions mean?" in rendered
    assert "Dimensions (x / group)" in rendered
    assert "Time To First Token" in rendered  # a metric explanation


def test_dashboards_load_save_over_http(tmp_path: Path) -> None:
    """The live server serves a dashboard for the editor and saves an edited one."""
    ddir = tmp_path / "dashboards"
    ddir.mkdir()
    (ddir / "d.yaml").write_text(_DASH_YAML, encoding="utf-8")
    server = make_server(tmp_path, host="127.0.0.1", port=0, dashboards_dir=ddir)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    payload = json.dumps(
        [
            {
                "title": "Tokens",
                "x": "level_or_rate",
                "group": "",
                "chart": "auto",
                "values": [{"metric": "output_tokens", "agg": "p50"}],
            }
        ]
    )
    try:
        with urlopen(f"http://127.0.0.1:{port}/dashboards/load?file=d.yaml") as resp:
            assert json.loads(resp.read())["panels"][0]["title"] == "Lat"
        save = Request(
            f"http://127.0.0.1:{port}/dashboards/save",
            data=urlencode({"file": "tokens.yaml", "payload": payload}).encode("utf-8"),
            method="POST",
        )
        with urlopen(save) as resp:
            assert json.loads(resp.read())["file"] == "tokens.yaml"
        assert "output_tokens" in (ddir / "tokens.yaml").read_text(encoding="utf-8")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
