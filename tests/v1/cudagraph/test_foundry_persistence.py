# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native/SAVE/LOAD contract test for V1 Foundry CUDA graph persistence.

This is an optional dedicated-GPU test.  It deliberately exercises the full
server lifecycle instead of mocking individual adapter hooks: a SAVE archive
must be consumed by a fresh LOAD process, the first request must replay a
restored graph, and a fixed batched workload must remain correct and usable.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.utils import (
    RemoteOpenAIServer,
    large_gpu_mark,
)

pytestmark = [
    pytest.mark.optional,
    large_gpu_mark(min_gb=16),
    pytest.mark.skipif(
        os.environ.get("VLLM_RUN_FOUNDRY_PERSISTENCE") != "1",
        reason="set VLLM_RUN_FOUNDRY_PERSISTENCE=1 for the dedicated Foundry GPU suite",
    ),
]

MODEL = os.environ.get(
    "VLLM_FOUNDRY_TEST_MODEL",
    os.environ.get("VLLM_TEST_MODEL", "Qwen/Qwen3-0.6B"),
)
CAPTURE_SIZES = [1, 2]
PROMPTS = [
    "The capital of France is",
    "The largest planet in the solar system is",
    "2 + 2 equals",
    "The opposite of hot is",
    "The chemical symbol for water is",
    "The first month of the year is",
    "A triangle has how many sides?",
    "The color made by mixing red and blue is",
]


@dataclass(frozen=True)
class RunResult:
    name: str
    startup_seconds: float
    restore_seconds: float
    outputs: tuple[str, ...]
    benchmark_seconds: float
    benchmark_output_tokens: int
    log_path: Path


class _LoggedServer(RemoteOpenAIServer):
    """RemoteOpenAIServer variant that retains the engine log for assertions."""

    def __init__(self, *args, log_path: Path, **kwargs):
        self._log_path = log_path
        self._log_file = None
        super().__init__(*args, **kwargs)

    def _start_server(self, model, vllm_serve_args, env_dict):
        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        # The current adapter integrates with the V1 GPUModelRunner.  Pin this
        # explicitly so a host-wide V2 default cannot make the test fail before
        # Foundry initialization.
        env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        if env_dict:
            env.update(env_dict)
        # Pin the server to this checkout.  The venv can contain an older
        # editable vLLM, which would silently bypass the adapter under test.
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
        self._log_file = self._log_path.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", model,
             *vllm_serve_args],
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def _shutdown(self):
        try:
            super()._shutdown()
        finally:
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None


def _write_foundry_config(path: Path, mode: str, workspace: Path) -> None:
    path.write_text(
        "\n".join(
            [
                f'mode = "{mode}"',
                f'workspace_root = "{workspace}"',
                'base_addr = "0x600000000000"',
                'region_size = "256GB"',
                'scratch_space_size = "2GB"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _compilation_config(foundry_config: Path | None) -> str:
    config = {"cudagraph_capture_sizes": CAPTURE_SIZES}
    if foundry_config is not None:
        config["graph_extension_config_path"] = str(foundry_config)
    return json.dumps(config)


def _server_args(foundry_config: Path | None) -> list[str]:
    return [
        "--trust-remote-code",
        "--tokenizer-mode",
        "kimi_k3",
        "--dtype",
        "bfloat16",
        "--quantization",
        "mxfp4",
        "--skip-mm-profiling",
        "--max-model-len",
        "128",
        "--max-num-seqs",
        "2",
        "--gpu-memory-utilization",
        "0.75",
        "--compilation-config",
        _compilation_config(foundry_config),
    ]


def _run_requests(server: _LoggedServer) -> tuple[tuple[str, ...], float, int]:
    client = server.get_client(timeout=600)
    outputs: list[str] = []
    output_tokens = 0
    start = time.perf_counter()
    for _ in range(3):
        response = client.completions.create(
            model=MODEL,
            prompt=PROMPTS,
            max_tokens=8,
            temperature=0.0,
            top_p=1.0,
            seed=0,
        )
        outputs.extend(choice.text for choice in response.choices)
        output_tokens += response.usage.completion_tokens
    return tuple(outputs), time.perf_counter() - start, output_tokens


def _run_arm(
    name: str,
    foundry_config: Path | None,
    log_path: Path,
    workspace: Path,
) -> RunResult:
    startup_start = time.perf_counter()
    try:
        with _LoggedServer(
            MODEL,
            _server_args(foundry_config),
            log_path=log_path,
            max_wait_seconds=900,
        ) as server:
            startup_seconds = time.perf_counter() - startup_start
            client = server.get_client(timeout=600)

            # The canary is a separate first request: LOAD must replay a restored
            # graph before the steady-state benchmark begins.
            canary = client.completions.create(
                model=MODEL,
                prompt="The capital of France is",
                max_tokens=8,
                temperature=0.0,
                top_p=1.0,
                seed=0,
            )
            assert canary.choices, "canary completion returned no choices"

            outputs, benchmark_seconds, output_tokens = _run_requests(server)
    except Exception as exc:
        log_tail = ""
        if log_path.exists():
            log_tail = log_path.read_text(encoding="utf-8")[-12000:]
        raise RuntimeError(
            f"{name} Foundry server failed; log={log_path}\n{log_tail}"
        ) from exc

    log_text = log_path.read_text(encoding="utf-8")
    restore_seconds = 0.0
    if name == "load":
        match = re.search(r"start_graph_builds returned in ([0-9.]+)s", log_text)
        assert match, "LOAD did not start Foundry graph builds"
        restore_seconds = float(match.group(1))
        assert "capture_or_load_graph called on LOAD" not in log_text

    if name == "save":
        archives = list(workspace.rglob("*.cugraph"))
        assert archives, "SAVE produced no Foundry graph archives"

    return RunResult(
        name=name,
        startup_seconds=startup_seconds,
        restore_seconds=restore_seconds,
        outputs=outputs,
        benchmark_seconds=benchmark_seconds,
        benchmark_output_tokens=output_tokens,
        log_path=log_path,
    )


@pytest.mark.timeout(1800)
def test_foundry_native_save_load_round_trip_and_benchmark(tmp_path: Path):
    """Compare native, SAVE, and LOAD on the same fixed batched workload."""
    pytest.importorskip("foundry")

    workspace = tmp_path / "foundry-archive"
    workspace.mkdir()
    save_config = tmp_path / "foundry-save.toml"
    load_config = tmp_path / "foundry-load.toml"
    _write_foundry_config(save_config, "save", workspace)
    _write_foundry_config(load_config, "load", workspace)

    log_root = Path(os.environ.get("VLLM_FOUNDRY_LOG_DIR", tmp_path))
    log_root.mkdir(parents=True, exist_ok=True)
    native = _run_arm("native", None, log_root / "native.log", workspace)
    save = _run_arm("save", save_config, log_root / "save.log", workspace)
    load = _run_arm("load", load_config, log_root / "load.log", workspace)

    # The full fixed workload is the correctness contract, not only the canary.
    assert save.outputs == native.outputs
    assert load.outputs == native.outputs

    native_rate = native.benchmark_output_tokens / native.benchmark_seconds
    load_rate = load.benchmark_output_tokens / load.benchmark_seconds
    print(
        "Foundry persistence benchmark: "
        f"startup native/save/load={native.startup_seconds:.2f}/"
        f"{save.startup_seconds:.2f}/{load.startup_seconds:.2f}s, "
        f"restore={load.restore_seconds:.3f}s, "
        f"output tok/s native/load={native_rate:.2f}/{load_rate:.2f}"
    )

    # LOAD should preserve steady-state behavior. Keep this relative threshold
    # loose enough for a shared nightly GPU while catching replay regressions.
    assert load_rate >= native_rate * 0.85
