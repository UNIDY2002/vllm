# CUDA graph persistence with Foundry

The optional Foundry adapter archives CUDA graphs during SAVE and reconstructs
them in a fresh LOAD process. LOAD uses the archived decode graphs without
recapturing them.

The current adapter targets the V1 model runner and FULL decode CUDA graphs.
It disables Torch compilation in both SAVE and LOAD so that uncaptured forwards,
including prefill, use the same kernels. CUDA graph capture and replay remain
enabled. Use the same model, parallelism, software stack, and capture sizes for
SAVE and LOAD. Foundry requires a wheel matching the Torch minor version and
CUDA major version; see [Foundry installation](https://github.com/foundry-org/foundry#installation).

## Regression and benchmark

Run the dedicated GPU test with Foundry installed:

```bash
VLLM_RUN_FOUNDRY_PERSISTENCE=1 \
VLLM_FOUNDRY_LOG_DIR=/tmp/foundry-persistence \
python -m pytest --optional -v -s \
    tests/v1/cudagraph/test_foundry_persistence.py
```

The test starts Native, SAVE pass 1, SAVE pass 2, and LOAD in independent
processes. It uses Qwen3-0.6B in BF16, checks the first LOAD request, requires
exact output equality on a repeated fixed workload, and runs `vllm bench serve`
with 64 requests of 32 input and 32 output tokens at concurrency 2. The two
decode graph sizes are 1 and 2. The test rejects recapture fallback and requires
LOAD throughput to reach at least 85% of the matched native reference.

Logs and benchmark JSON files are retained in `VLLM_FOUNDRY_LOG_DIR`.
`results.json` records startup time, graph construction time, exact outputs,
and benchmark duration/token counts. Startup time measures server creation
through readiness; graph construction time measures Foundry's completed
background builds. This small-model regression reports startup savings without
enforcing a noisy startup-speed threshold. Measure production startup savings
with the intended model and capture-size set.

The optional Buildkite nightly job in `.buildkite/test_areas/cuda.yaml` runs
the same test on a single H200 partition with a matching Foundry dependency.
