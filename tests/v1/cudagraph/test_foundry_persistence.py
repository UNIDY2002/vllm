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
from dataclasses import asdict, dataclass
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
    graph_build_seconds: float
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
        # Exercise regular FULL decode graphs at the wrapper boundary.
        env["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "0"
        if env_dict:
            env.update(env_dict)
        # Pin the server to this checkout.  The venv can contain an older
        # editable vLLM, which would silently bypass the adapter under test.
        env["PYTHONPATH"] = os.pathsep.join(
            filter(
                None,
                [str(Path(__file__).resolve().parents[3]), env.get("PYTHONPATH")],
            )
        )
        self._log_file = self._log_path.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "vllm.entrypoints.cli.main",
                "serve",
                model,
                *vllm_serve_args,
            ],
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
    # Keep this CI workload on the FULL decode path.  PIECEWISE capture creates
    # dozens of backend subgraphs whose restore order is a separate contract;
    # this test targets the Foundry archive/replay lifecycle for the V1 wrapper
    # graphs and exercises two deterministic decode sizes.
    config = {
        # Foundry restores driver graphs without rebuilding a Torch compiled
        # model. Match that kernel path in the native reference as well, while
        # retaining FULL decode graph capture/replay in every arm.
        "mode": 0,
        "cudagraph_capture_sizes": CAPTURE_SIZES,
        "cudagraph_mode": "FULL_DECODE_ONLY",
    }
    if foundry_config is not None:
        # Exercise the adapter's automatic SAVE/LOAD compile-mode pin.
        config.pop("mode")
        config["graph_extension_config_path"] = str(foundry_config)
    return json.dumps(config)


def _server_args(foundry_config: Path | None) -> list[str]:
    return [
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
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


def _run_benchmark(server: _LoggedServer, result_path: Path) -> tuple[float, int]:
    """Use vLLM's serving benchmark for the steady-state performance gate."""
    with result_path.with_suffix(".log").open("w", encoding="utf-8") as log:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "vllm.entrypoints.cli.main",
                "bench",
                "serve",
                "--model",
                MODEL,
                "--base-url",
                server.url_root,
                "--backend",
                "openai",
                "--dataset-name",
                "random",
                "--random-input-len",
                "32",
                "--random-output-len",
                "32",
                "--num-prompts",
                "64",
                "--max-concurrency",
                "2",
                "--num-warmups",
                "2",
                "--ignore-eos",
                "--seed",
                "0",
                "--save-result",
                "--result-dir",
                str(result_path.parent),
                "--result-filename",
                result_path.name,
            ],
            check=True,
            timeout=300,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["completed"] == 64 and result["failed"] == 0, result
    assert result["total_output_tokens"] == 64 * 32, result
    return result["duration"], result["total_output_tokens"]


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
            canary_text = canary.choices[0].text
            if MODEL == "Qwen/Qwen3-0.6B":
                assert "paris" in canary_text.lower(), canary_text

            outputs, _, _ = _run_requests(server)
            outputs = (canary_text, *outputs)
            benchmark_seconds, output_tokens = _run_benchmark(
                server, log_path.with_name(f"{name}-benchmark.json")
            )
    except Exception as exc:
        log_tail = ""
        if log_path.exists():
            log_tail = log_path.read_text(encoding="utf-8")[-12000:]
        raise RuntimeError(
            f"{name} Foundry server failed; log={log_path}\n{log_tail}"
        ) from exc

    log_text = log_path.read_text(encoding="utf-8")
    graph_build_seconds = 0.0
    if name == "load":
        # start_graph_builds only schedules background work; its return time
        # is not the time spent reconstructing and instantiating the graphs.
        match = re.search(r"2 graphs built in ([0-9.]+) ms", log_text)
        assert match, "LOAD did not finish building the two archived graphs"
        graph_build_seconds = float(match.group(1)) / 1000
        assert log_text.count("restored wrapper graph descriptor=") == 2
        assert "LOAD recapture fallback" not in log_text
        assert "capture_or_load_graph called on LOAD" not in log_text

    if name == "save":
        archives = list(workspace.rglob("*.cugraph"))
        assert archives, "SAVE produced no Foundry graph archives"

    return RunResult(
        name=name,
        startup_seconds=startup_seconds,
        graph_build_seconds=graph_build_seconds,
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
    # vLLM's KV memory profile is not reproducible on the first SAVE pass.
    # SAVE pass 1 persists warmup_state.json; pass 2 reuses that state and
    # writes the archive from the deterministic allocation trajectory that
    # LOAD must reproduce.
    _run_arm("save-pass1", save_config, log_root / "save-pass1.log", workspace)
    save = _run_arm("save", save_config, log_root / "save-pass2.log", workspace)
    load = _run_arm("load", load_config, log_root / "load.log", workspace)

    # Preserve metrics and exact outputs even when the parity gate fails.
    (log_root / "results.json").write_text(
        json.dumps(
            [asdict(result) for result in (native, save, load)],
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # The full fixed workload is the correctness contract, not only the canary.
    assert save.outputs == native.outputs
    assert load.outputs == native.outputs

    native_rate = native.benchmark_output_tokens / native.benchmark_seconds
    load_rate = load.benchmark_output_tokens / load.benchmark_seconds
    print(
        "Foundry persistence benchmark: "
        f"startup native/save/load={native.startup_seconds:.2f}/"
        f"{save.startup_seconds:.2f}/{load.startup_seconds:.2f}s, "
        f"graph build={load.graph_build_seconds:.3f}s, "
        f"output tok/s native/load={native_rate:.2f}/{load_rate:.2f}"
    )

    # LOAD should preserve steady-state behavior. Keep this relative threshold
    # loose enough for a shared nightly GPU while catching replay regressions.
    assert load_rate >= native_rate * 0.85
