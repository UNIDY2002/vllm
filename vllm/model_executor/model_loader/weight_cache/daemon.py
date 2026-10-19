# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight cache daemon for fast engine restarts.

One daemon process per GPU holds the post-quantized, TP-sharded weights of its
rank in GPU memory and serves CUDA IPC handles to vLLM engines over a Unix
domain socket. Restarting engines map the weights via zero-copy IPC instead of
reloading from disk.

Launch one daemon per TP rank with a single command:

    vllm preload \\
        --model /path/to/model --tensor-parallel-size 4

Engines then load from the daemons with:

    vllm serve /path/to/model --tensor-parallel-size 4 \\
        --load-format ipc_cache

Tensor, expert, data and pipeline parallelism are supported.

For multi-node tensor parallelism, run one launcher per node with a shared
rendezvous so the global TP group forms across nodes (CUDA IPC handles are
node-local, so each node serves only its local GPUs' shards). Reuse the same
``--nnodes``/``--node-rank``/``--master-addr`` flags you pass the engine, plus a
``--weight-cache-master-port`` distinct from the engine's ``--master-port``:

    # node 0 (8 local GPUs)
    vllm preload \\
        --model /path/to/model --tensor-parallel-size 16 \\
        --nnodes 2 --node-rank 0 --master-addr 10.0.0.1 \\
        --weight-cache-master-port 29600
    # node 1 (8 local GPUs)
    vllm preload \\
        --model /path/to/model --tensor-parallel-size 16 \\
        --nnodes 2 --node-rank 1 --master-addr 10.0.0.1 \\
        --weight-cache-master-port 29600

The global TP rank of local GPU ``i`` on node ``r`` is
``r * (tp_size // nnodes) + i``, matching vLLM's contiguous per-node rank
assignment, so each engine worker maps its shard from the daemon on its own
node. With pipeline parallelism each node additionally runs one daemon per PP
stage; the global rank is ``pp_rank * tp_size + tp_rank``.

For data parallelism (e.g. a TP1 x DP16 x EP decode fleet) run one launcher per
node with the engine's DP placement flags. Local GPU ``i`` serves DP rank
``start_rank + i // tp_size`` and TP rank ``i % tp_size``, and all
``dp_size * tp_size`` daemons form one world group on
``--data-parallel-address``/``--weight-cache-master-port`` so the expert
shards are laid out exactly as in the engine:

    # node r (4 local GPUs)
    vllm preload \\
        --model /path/to/model --tensor-parallel-size 1 --enable-expert-parallel \\
        --data-parallel-size 16 --data-parallel-size-local 4 \\
        --data-parallel-start-rank 4r --data-parallel-address 10.0.0.1 \\
        --weight-cache-master-port 29600

Data parallelism also combines with multi-node tensor parallelism: pass both
flag sets (``--nnodes``/``--node-rank``/``--master-addr`` and the DP flags with
a reachable ``--data-parallel-address``). Each node then serves a contiguous
block of the ``dp_size * tp_size`` global ranks, e.g. TP8 x DP2 on 4 nodes:

    # node r (4 local GPUs)
    vllm preload \\
        --model /path/to/model --tensor-parallel-size 8 --enable-expert-parallel \\
        --nnodes 4 --node-rank r --master-addr 10.0.0.1 \\
        --data-parallel-size 2 --data-parallel-address 10.0.0.1 \\
        --weight-cache-master-port 29600

With MTP, EAGLE or EAGLE3 speculative decoding the launcher additionally
starts a draft daemon group that caches the draft model. It uses its own cache
key, Unix sockets (``*_draft.sock``) and rendezvous port
(``--weight-cache-draft-master-port``, default ``--weight-cache-master-port +
1``), so each process serves exactly one model role. Other draft types are not
cached and keep loading from disk in the engine.
"""

import contextlib
import ctypes
import fcntl
import gc
import hmac
import multiprocessing
import os
import socket
import struct
import threading
import time
from collections.abc import Callable

import torch

from vllm.config import (
    ParallelConfig,
    VllmConfig,
    replace,
    set_current_vllm_config,
)
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_cache.artifact_cache import (
    ArtifactStore,
)
from vllm.model_executor.model_loader.weight_cache.compile_artifact_cache import (
    CompileCacheManifest,
    compile_cache_manifest_from_dict,
    create_compile_cache_bundle,
    get_default_compile_cache_roots,
    install_compile_cache_bundle,
)
from vllm.model_executor.model_loader.weight_cache.protocol import (
    TensorEntry,
    WeightCacheKey,
    WeightCacheUnavailableError,
    check_ipc_quant_support,
    ensure_private_socket_dir,
    get_current_device_uuid,
    get_socket_path,
    recv_json,
    recv_msg,
    send_json,
    send_msg,
    verify_peer_is_owner,
)
from vllm.model_executor.model_loader.weight_cache.utils import (
    build_warmup_runner,
    export_model_attrs,
    format_daemon_role,
    is_draft_model_cacheable,
)
from vllm.model_executor.warmup.flashinfer_autotune_cache import (
    flashinfer_autotune_artifact_key,
    resolve_rank_flashinfer_autotune_file,
    write_flashinfer_autotune_cache,
)
from vllm.model_executor.warmup.kernel_warmup import flashinfer_autotune
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.worker.workspace import init_workspace_manager

logger = init_logger("vllm.model_executor.model_loader.weight_cache.daemon")


def _send_bytes(conn: socket.socket, payload: bytes) -> None:
    conn.sendall(struct.pack("!Q", len(payload)))
    view = memoryview(payload)
    while view:
        sent = conn.send(view[: 1 << 20])
        if sent <= 0:
            raise ConnectionError("compile cache peer closed the connection")
        view = view[sent:]


def _recv_bytes(conn: socket.socket, size: int, *, deadline: float) -> bytes:
    if size < 0 or size > 8 << 30:
        raise ValueError("invalid compile cache payload size")
    header = bytearray()
    while len(header) < 8:
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out reading compile cache payload")
        chunk = conn.recv(8 - len(header))
        if not chunk:
            raise ConnectionError("compile cache peer closed the connection")
        header.extend(chunk)
    announced = struct.unpack("!Q", header)[0]
    if announced != size:
        raise ValueError("compile cache payload size mismatch")
    payload = bytearray()
    while len(payload) < size:
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out reading compile cache payload")
        chunk = conn.recv(min(1 << 20, size - len(payload)))
        if not chunk:
            raise ConnectionError("compile cache peer closed the connection")
        payload.extend(chunk)
    return bytes(payload)


def export_entries(
    model: torch.nn.Module,
) -> tuple[dict[str, TensorEntry], dict[str, str]]:
    """Export a model's tensors, preserving tied-parameter aliases.

    ``named_parameters``/``named_buffers`` are iterated with
    ``remove_duplicate=False`` so tied weights (e.g. ``lm_head.weight`` sharing
    storage with ``embed_tokens.weight``) are not silently dropped. Each unique
    tensor is exported once per call; every additional name that refers to the same
    tensor object is recorded in the returned alias map so the client can
    re-establish the shared identity instead of allocating uninitialized
    memory for it.

    CUDA reduction arguments must be exported separately for each consumer so
    that PyTorch registers a reference for each IPC mapping's lifetime.

    Returns:
        A ``(entries, aliases)`` pair where ``entries`` maps a canonical name to
        its ``TensorEntry`` and ``aliases`` maps each duplicate name to its
        canonical name.

    """
    entries: dict[str, TensorEntry] = {}
    aliases: dict[str, str] = {}
    canonical_by_id: dict[int, str] = {}

    def _add(name: str, tensor: torch.Tensor, kind: str) -> None:
        canonical = canonical_by_id.get(id(tensor))
        if canonical is not None:
            aliases[name] = canonical
            return
        canonical_by_id[id(tensor)] = name
        entries[name] = TensorEntry.from_tensor(tensor, kind)

    for name, param in model.named_parameters(remove_duplicate=False):
        _add(name, param, "param")
    # named_buffers includes non-persistent buffers (e.g. rotary embedding
    # caches) that state_dict would miss.
    for name, buffer in model.named_buffers(remove_duplicate=False):
        if name in entries or name in aliases:
            continue
        _add(name, buffer, "buffer")
    return entries, aliases


class WeightCacheDaemon:
    """Per-GPU process that loads one TP/PP shard and serves CUDA IPC handles."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        global_rank: int,
        local_rank: int,
        distributed_init_method: str,
        socket_dir: str | None = None,
        is_draft: bool = False,
        dp_rank: int = 0,
        pp_rank: int = 0,
        artifact_source: tuple[str, int] | None = None,
        artifact_listen: tuple[str, int] | None = None,
        artifact_token: str | None = None,
        compile_cache_backend: str = "none",
        compile_cache_mooncake_protocol: str = "rdma",
    ):
        parallel_config = vllm_config.parallel_config
        if parallel_config.enable_eplb:
            raise ValueError("The weight cache daemon does not support EPLB.")
        self.tp_size = parallel_config.tensor_parallel_size
        self.pp_size = parallel_config.pipeline_parallel_size
        self.dp_size = parallel_config.data_parallel_size
        if self.dp_size > 1:
            # The engine sets these on each DP rank's workers; the MoE/EP
            # layers read them to place their expert shards.
            if parallel_config.nnodes > 1:
                world_size = self.dp_size * self.tp_size
                local_world = world_size // parallel_config.nnodes
                start_dp_rank = parallel_config.node_rank * local_world // self.tp_size
            else:
                start_dp_rank = parallel_config.data_parallel_rank
            parallel_config = replace(
                parallel_config,
                data_parallel_rank=dp_rank,
                data_parallel_rank_local=dp_rank - start_dp_rank,
            )
            vllm_config = replace(vllm_config, parallel_config=parallel_config)
        self.vllm_config = vllm_config
        # A draft daemon builds the draft with the target's VllmConfig, like
        # the engine does; only the ModelConfig differs.
        if is_draft:
            spec = vllm_config.speculative_config
            assert spec is not None and spec.draft_model_config is not None
            self.model_config = spec.draft_model_config
        else:
            self.model_config = vllm_config.model_config
        self.global_rank = global_rank
        rank_in_dp = global_rank - dp_rank * self.pp_size * self.tp_size
        self.tp_rank = rank_in_dp % self.tp_size
        self.pp_rank = rank_in_dp // self.tp_size
        self.dp_rank = dp_rank
        self.world_size = self.dp_size * self.pp_size * self.tp_size
        self.local_rank = local_rank
        self.distributed_init_method = distributed_init_method
        self.socket_dir = socket_dir
        self.is_draft = is_draft
        self.role = format_daemon_role(is_draft)
        self.model: torch.nn.Module | None = None
        key_dp_size, key_dp_rank = get_engine_dp_placement(vllm_config, dp_rank)
        self.artifacts = ArtifactStore()
        self.artifact_source = artifact_source
        self.artifact_listen = artifact_listen
        self.artifact_token = artifact_token
        if compile_cache_backend not in ("none", "tcp", "mooncake"):
            raise ValueError(
                f"unsupported compile-cache backend: {compile_cache_backend!r}"
            )
        self.compile_cache_backend = compile_cache_backend
        if compile_cache_mooncake_protocol not in ("tcp", "rdma"):
            raise ValueError(
                "unsupported Mooncake compile-cache protocol: "
                f"{compile_cache_mooncake_protocol!r}"
            )
        self.compile_cache_mooncake_protocol = compile_cache_mooncake_protocol
        self.compile_cache_roots = get_default_compile_cache_roots()
        self.compile_cache_bundle: tuple[CompileCacheManifest, bytes] | None = None
        # (TransferEngine, registered buffer address, buffer size, session,
        # source buffer owner).
        self._mooncake_source: tuple[object, int, int, str, object] | None = None
        # Fingerprint before loading: process_weights_after_loading may
        # mutate hf_config.quantization_config.
        self.cache_config = WeightCacheKey.from_model_config(
            self.model_config,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            pp_size=self.pp_size,
            pp_rank=self.pp_rank,
            dp_size=key_dp_size,
            dp_rank=key_dp_rank,
            is_draft=is_draft,
            enable_expert_parallel=parallel_config.enable_expert_parallel,
            expert_placement_strategy=parallel_config.expert_placement_strategy,
        )

    def warmup(self) -> None:
        """Run the FlashInfer autotune pass against the cached model."""
        vllm_config = self.vllm_config
        # Mirrors the gate in the engine's kernel warmup.
        if (
            vllm_config.kernel_config.enable_flashinfer_autotune is False
            or not has_flashinfer()
            or not current_platform.has_device_capability(90)
            or is_draft_model_cacheable(vllm_config.speculative_config)
        ):
            return
        try:
            current_platform.update_block_size_for_backend(vllm_config)
            runner = build_warmup_runner(
                vllm_config,
                self.local_rank,
                is_draft=self.is_draft,
                model_config=self.model_config,
            )
            assert self.model is not None, "warmup ran before load_model"
            runner.load_model(model=self.model)
            skip_attn = vllm_config.use_v2_model_runner
            if self._uses_remote_artifacts:
                self._load_remote_autotune_artifact(runner, skip_attn=skip_attn)
            flashinfer_autotune(runner, skip_attn=skip_attn)
            if self._uses_remote_artifacts:
                self._publish_autotune_artifact(runner, skip_attn=skip_attn)
            logger.info(
                "Weight cache %s daemon rank %d tuned FlashInfer; the tuned "
                "configs are in the on-disk autotune cache",
                self.role,
                self.global_rank,
            )
        finally:
            # Free the runner and the dummy-run activations; the daemon's
            # weights are untouched.
            gc.collect()
            torch.accelerator.empty_cache()

    @property
    def _uses_remote_artifacts(self) -> bool:
        return self.artifact_source is not None or self.artifact_listen is not None

    def _load_remote_autotune_artifact(self, runner, *, skip_attn: bool) -> None:
        key = flashinfer_autotune_artifact_key(
            runner,
            skip_attn=skip_attn,
        )
        cache_path = resolve_rank_flashinfer_autotune_file(runner)
        self.fetch_artifacts()
        table = self.artifacts.get(key)
        if table is not None:
            write_flashinfer_autotune_cache(cache_path, table)
            logger.info("Installed remote FlashInfer autotune table: %s", cache_path)

    def _publish_autotune_artifact(self, runner, *, skip_attn: bool) -> None:
        cache_path = resolve_rank_flashinfer_autotune_file(runner)
        if not cache_path.is_file():
            return
        key = flashinfer_autotune_artifact_key(
            runner,
            skip_attn=skip_attn,
        )
        self.artifacts.put(key, cache_path.read_bytes())

    def fetch_artifacts(self) -> None:
        """Best-effort import of autotune artifacts from another daemon."""
        if self.artifact_source is None:
            return
        if not self.artifact_token:
            raise ValueError("artifact source requires an artifact token")
        try:
            with socket.create_connection(self.artifact_source, timeout=30.0) as sock:
                send_json(
                    sock,
                    {
                        "cmd": "fetch_artifacts",
                        "token": self.artifact_token,
                    },
                )
                response = recv_json(sock)
            if not isinstance(response, dict) or response.get("status") != "ok":
                raise RuntimeError(f"artifact source refused request: {response!r}")
            adopted = self.artifacts.merge_json(response.get("artifacts"))
            logger.info(
                "Weight cache %s daemon rank %d adopted %d autotune artifact(s)",
                self.role,
                self.global_rank,
                adopted,
            )
        except (OSError, ConnectionError, ValueError, RuntimeError) as error:
            logger.warning(
                "Cannot fetch autotune artifacts from %s; using local warmup: %s",
                self.artifact_source,
                error,
            )

    def fetch_compile_cache(self) -> None:
        """Install a validated compiled-cache bundle before model loading."""
        if self.compile_cache_backend == "none" or self.artifact_source is None:
            return
        if not self.artifact_token:
            raise ValueError("compile cache source requires an artifact token")
        deadline = time.monotonic() + 300.0
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self._fetch_compile_cache()
                return
            except (OSError, ConnectionError, ValueError, RuntimeError) as error:
                last_error = error
                time.sleep(1.0)
        raise RuntimeError(
            f"timed out fetching compile cache from {self.artifact_source}"
        ) from last_error

    def _fetch_compile_cache(self) -> None:
        fetch_started = time.monotonic()
        assert self.artifact_source is not None
        with socket.create_connection(self.artifact_source, timeout=10.0) as sock:
            send_json(
                sock,
                {
                    "cmd": "fetch_compile_cache",
                    "token": self.artifact_token,
                },
            )
            response = recv_json(sock)
            if not isinstance(response, dict) or response.get("status") != "ok":
                raise RuntimeError(
                    f"compile cache source refused request: {response!r}"
                )
            manifest = compile_cache_manifest_from_dict(response.get("manifest"))
            bundle_size = response.get("bundle_size")
            if not isinstance(bundle_size, int) or bundle_size < 0:
                raise ValueError("invalid compile cache bundle size")
            transport = response.get("transport", "tcp")
            payload_started = time.monotonic()
            if transport == "tcp":
                bundle = _recv_bytes(
                    sock, bundle_size, deadline=time.monotonic() + 300.0
                )
            elif transport == "mooncake":
                bundle = self._fetch_compile_cache_mooncake(response, bundle_size)
            else:
                raise ValueError(f"unsupported compile-cache transport: {transport!r}")
            payload_elapsed = time.monotonic() - payload_started
        install_compile_cache_bundle(bundle, manifest, self.compile_cache_roots)
        total_elapsed = time.monotonic() - fetch_started
        logger.info(
            "Weight cache %s daemon rank %d installed %d compiled-cache files "
            "(%d bytes) in %.3f s (payload %.3f s)",
            self.role,
            self.global_rank,
            len(manifest.entries),
            manifest.total_size,
            total_elapsed,
            payload_elapsed,
        )

    def _fetch_compile_cache_mooncake(
        self, response: dict[str, object], bundle_size: int
    ) -> bytes:
        session = response.get("session")
        remote_address = response.get("address")
        protocol = response.get("protocol", "rdma")
        if (
            not isinstance(session, str)
            or not isinstance(remote_address, int)
            or protocol not in ("tcp", "rdma")
        ):
            raise ValueError("malformed Mooncake compile-cache descriptor")
        try:
            from mooncake.engine import TransferEngine
        except ImportError as error:
            raise RuntimeError(
                "Mooncake compile-cache backend requires mooncake.engine"
            ) from error
        from vllm.utils.network_utils import get_ip

        engine = TransferEngine()
        if engine.initialize(get_ip(), "P2PHANDSHAKE", protocol, "") != 0:
            raise RuntimeError("Mooncake TransferEngine initialization failed")
        local_buffer = ctypes.create_string_buffer(bundle_size)
        local_address = ctypes.addressof(local_buffer)
        try:
            if engine.register_memory(local_address, bundle_size) != 0:
                raise RuntimeError("Mooncake failed to register compile-cache buffer")
            try:
                status = engine.batch_transfer_sync_read(
                    session, [local_address], [remote_address], [bundle_size]
                )
            finally:
                engine.unregister_memory(local_address)
            if status != 0:
                raise RuntimeError(
                    f"Mooncake compile-cache transfer failed with status {status}"
                )
            return local_buffer.raw[:bundle_size]
        finally:
            del local_buffer

    def load_model(self) -> None:
        torch.accelerator.set_device_index(self.local_rank)
        # Outside the config context so the daemon keeps its own rendezvous
        # instead of the engine's; model parallel groups (TP/DP/EP) are then
        # carved out of this world group like in the engine.
        init_distributed_environment(
            world_size=self.world_size,
            rank=self.global_rank,
            distributed_init_method=self.distributed_init_method,
            local_rank=self.local_rank,
            backend=current_platform.dist_backend,
        )
        with set_current_vllm_config(self.vllm_config):
            ensure_model_parallel_initialized(self.tp_size, self.pp_size)
            self.model = self.get_model()
        # Loading and post-processing leave freed transients in the caching
        # allocator; return them so engines sharing the GPU can use them.
        gc.collect()
        torch.accelerator.empty_cache()
        logger.info(
            "Weight cache %s daemon rank %d loaded model (%s GiB allocated, "
            "%s GiB reserved)",
            self.role,
            self.global_rank,
            format_gib(torch.accelerator.memory_allocated()),
            format_gib(torch.accelerator.memory_reserved()),
        )

    def get_model(self) -> torch.nn.Module:
        """Load the daemon's model, composed from the configured loader.

        Runs the quantization check after model creation but before the slow
        weight load, so an unsupported method fails fast. Online quantization
        always fails the check, so load_model's finalize step for it is
        unnecessary here.
        """
        vllm_config = self.vllm_config
        model_config = self.model_config
        load_config = vllm_config.load_config
        loader = get_model_loader(load_config)
        device_config = vllm_config.device_config
        target_device = torch.device(
            device_config.device if load_config.device is None else load_config.device
        )
        # Attention backends may take workspace at construction, like in the
        # engine's worker init.
        init_workspace_manager(target_device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = loader.create_model(vllm_config, model_config)
            check_ipc_quant_support(model)
            loader.load_weights(model, model_config)
            process_weights_after_loading(model, model_config, target_device)
        return model.eval()

    def serve_forever(self, ready_callback: Callable[[], None] | None = None) -> None:
        """Serve requests until terminated.

        The socket is only bound once the model is fully cached, so clients
        get a connection error (and fall back to disk) until the daemon is
        ready.

        Args:
            ready_callback: Invoked once the socket is bound and listening,
                so the launcher can report overall readiness.

        """
        socket_path = self._socket_path
        ensure_private_socket_dir(
            os.path.dirname(socket_path), strict_perms=self.socket_dir is None
        )
        # Hold an exclusive per-GPU lock for the daemon's lifetime so a second
        # daemon cannot remove this daemon's live socket and hijack the path.
        lock_fd = self._acquire_gpu_lock(socket_path)
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(socket_path)
        os.chmod(socket_path, 0o600)
        server.listen()
        remote_server = None
        remote_listen = self.artifact_listen
        if remote_listen is not None:
            if self.compile_cache_backend != "none":
                self.compile_cache_bundle = create_compile_cache_bundle(
                    self.compile_cache_roots
                )
                logger.info(
                    "Prepared compiled-cache bundle with %d bytes",
                    len(self.compile_cache_bundle[1]),
                )
            family = socket.AF_INET6 if ":" in remote_listen[0] else socket.AF_INET
            remote_server = socket.create_server(remote_listen, family=family)
            threading.Thread(
                target=self._serve_artifact_remote,
                args=(remote_server,),
                name=f"vllm-weight-cache-artifacts-{self.global_rank}",
                daemon=True,
            ).start()
        logger.info(
            "Weight cache %s daemon rank %d serving on %s",
            self.role,
            self.global_rank,
            socket_path,
        )
        if ready_callback is not None:
            ready_callback()
        try:
            while True:
                conn, _ = server.accept()
                with conn:
                    try:
                        verify_peer_is_owner(conn)
                        self._handle_connection(conn)
                    except (ConnectionError, EOFError):
                        logger.warning("Client disconnected mid-request")
                    except Exception as e:
                        # Report the error back instead of just closing the
                        # socket, but don't let it take the daemon down.
                        logger.exception(
                            "Error handling weight cache client; continuing"
                        )
                        with contextlib.suppress(OSError):
                            send_msg(conn, {"status": "error", "message": str(e)})
        finally:
            server.close()
            if remote_server is not None:
                remote_server.close()
            if os.path.exists(socket_path):
                os.unlink(socket_path)
            os.close(lock_fd)

    def _serve_artifact_remote(self, server: socket.socket) -> None:
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            with conn:
                try:
                    request = recv_json(conn, max_size=4096)
                    presented = (
                        request.get("token") if isinstance(request, dict) else None
                    )
                    if (
                        not self.artifact_token
                        or not isinstance(presented, str)
                        or not hmac.compare_digest(presented, self.artifact_token)
                    ):
                        send_json(
                            conn,
                            {
                                "status": "error",
                                "message": "Unauthorized artifact request",
                            },
                        )
                        continue
                    if not isinstance(request, dict) or request.get("cmd") not in (
                        "fetch_artifacts",
                        "fetch_compile_cache",
                    ):
                        send_json(
                            conn,
                            {
                                "status": "error",
                                "message": "Unsupported artifact command",
                            },
                        )
                        continue
                    if request["cmd"] == "fetch_artifacts":
                        send_json(
                            conn,
                            {
                                "status": "ok",
                                "artifacts": self.artifacts.to_json(),
                            },
                        )
                    else:
                        if self.compile_cache_bundle is None:
                            raise RuntimeError("compiled-cache bundle is not ready")
                        manifest, bundle = self.compile_cache_bundle
                        if self.compile_cache_backend == "tcp":
                            send_json(
                                conn,
                                {
                                    "status": "ok",
                                    "manifest": manifest.as_dict(),
                                    "bundle_size": len(bundle),
                                    "transport": "tcp",
                                },
                            )
                            _send_bytes(conn, bundle)
                        else:
                            self._ensure_mooncake_source(bundle)
                            assert self._mooncake_source is not None
                            _, address, _, session, _ = self._mooncake_source
                            send_json(
                                conn,
                                {
                                    "status": "ok",
                                    "manifest": manifest.as_dict(),
                                    "bundle_size": len(bundle),
                                    "transport": "mooncake",
                                    "protocol": self.compile_cache_mooncake_protocol,
                                    "address": address,
                                    "session": session,
                                },
                            )
                except (OSError, ConnectionError, RuntimeError, ValueError) as e:
                    with contextlib.suppress(OSError):
                        send_json(conn, {"status": "error", "message": str(e)})

    def _ensure_mooncake_source(self, bundle: bytes) -> None:
        if self._mooncake_source is not None:
            return
        try:
            from mooncake.engine import TransferEngine
        except ImportError as error:
            raise RuntimeError(
                "Mooncake compile-cache backend requires mooncake.engine"
            ) from error
        from vllm.utils.network_utils import get_ip

        engine = TransferEngine()
        if (
            engine.initialize(
                get_ip(),
                "P2PHANDSHAKE",
                self.compile_cache_mooncake_protocol,
                "",
            )
            != 0
        ):
            raise RuntimeError("Mooncake TransferEngine initialization failed")
        source_buffer = ctypes.create_string_buffer(bundle)
        address = ctypes.addressof(source_buffer)
        if engine.register_memory(address, len(bundle)) != 0:
            raise RuntimeError("Mooncake failed to register compile-cache buffer")
        self._mooncake_source = (
            engine,
            address,
            len(bundle),
            f"{get_ip()}:{engine.get_rpc_port()}",
            source_buffer,
        )

    def _acquire_gpu_lock(self, socket_path: str) -> int:
        """Take an exclusive lock guarding this GPU's socket path.

        The lock is advisory and released automatically when the daemon exits
        (or crashes), so a stale socket is only ever removed by whoever owns
        the lock. A running daemon holding it makes a second daemon fail fast
        instead of clobbering the live socket.
        """
        lock_fd = os.open(f"{socket_path}.lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            os.close(lock_fd)
            raise WeightCacheUnavailableError(
                f"Another weight cache daemon already owns {socket_path}"
            ) from e
        return lock_fd

    @property
    def _socket_path(self) -> str:
        return get_socket_path(
            get_current_device_uuid(),
            self.socket_dir,
            is_draft=self.is_draft,
        )

    def _handle_connection(self, conn: socket.socket) -> None:
        request = recv_msg(conn)
        cmd = request.get("cmd")
        if cmd == "get_state":
            self._handle_get_state(conn, request)
        elif cmd == "get_memory":
            send_msg(
                conn,
                {
                    "status": "ok",
                    "memory_bytes": torch.accelerator.memory_allocated(),
                },
            )
        elif cmd == "release":
            self._handle_release(conn)
        else:
            send_msg(conn, {"status": "error", "message": f"Unknown command {cmd!r}"})

    def _handle_get_state(self, conn: socket.socket, request: dict) -> None:
        client_config = request.get("cache_config")
        if not isinstance(client_config, WeightCacheKey):
            send_msg(conn, {"status": "error", "message": "Missing cache_config"})
            return
        mismatched = self.cache_config.mismatched_fields(client_config)
        if mismatched:
            logger.warning("WeightCacheKey mismatch on fields: %s", mismatched)
            send_msg(conn, {"status": "mismatch", "fields": mismatched})
            return
        if self.model is None:
            send_msg(conn, {"status": "error", "message": "Weights were released"})
            return
        gpu_uuid = get_current_device_uuid()
        entries, aliases = export_entries(self.model)
        send_msg(
            conn,
            {
                "status": "ok",
                "entries": entries,
                "aliases": aliases,
                "attrs": export_model_attrs(self.model),
                "gpu_uuid": gpu_uuid,
            },
        )
        logger.info_once(
            "Weight cache %s daemon rank %d sent %d tensors (+%d aliases) to engine",
            self.role,
            self.global_rank,
            len(entries),
            len(aliases),
        )

    def _handle_release(self, conn: socket.socket) -> None:
        self.model = None
        torch.accelerator.empty_cache()
        logger.info(
            "Weight cache %s daemon rank %d released cached weights",
            self.role,
            self.global_rank,
        )
        send_msg(conn, {"status": "ok"})


def _run_daemon(
    global_rank: int,
    local_rank: int,
    vllm_config: VllmConfig,
    distributed_init_method: str,
    socket_dir: str | None,
    ready_queue: "multiprocessing.Queue[tuple[str, int]]",
    is_draft: bool = False,
    dp_rank: int = 0,
    pp_rank: int = 0,
    artifact_source: tuple[str, int] | None = None,
    artifact_listen: tuple[str, int] | None = None,
    artifact_token: str | None = None,
    compile_cache_backend: str = "none",
    compile_cache_mooncake_protocol: str = "rdma",
) -> None:
    daemon = WeightCacheDaemon(
        vllm_config,
        global_rank,
        local_rank,
        distributed_init_method,
        socket_dir,
        is_draft,
        dp_rank,
        pp_rank,
        artifact_source,
        artifact_listen,
        artifact_token,
        compile_cache_backend,
        compile_cache_mooncake_protocol,
    )
    daemon.fetch_compile_cache()
    daemon.load_model()
    daemon.warmup()
    daemon.serve_forever(
        ready_callback=lambda: ready_queue.put((daemon.role, daemon.global_rank))
    )


def plan_local_ranks(
    parallel_config: ParallelConfig,
) -> list[tuple[int, int, int, int]]:
    """``(local_rank, dp_rank, pp_rank, tp_rank)`` for every local daemon.

    Global ranks enumerate DP, PP, then TP: ``global = dp_rank * pp_size *
    tp_size + pp_rank * tp_size + tp_rank``.
    With ``--nnodes`` the engine hands each node a contiguous block of global
    ranks, so node ``r`` serves ``node_rank * local + i``; without it a
    launcher's block starts at ``--data-parallel-start-rank * tp_size``.
    """
    tp_size = parallel_config.tensor_parallel_size
    pp_size = parallel_config.pipeline_parallel_size
    world_size = parallel_config.data_parallel_size * pp_size * tp_size
    if parallel_config.nnodes > 1:
        local_world_size = world_size // parallel_config.nnodes
        base = parallel_config.node_rank * local_world_size
    else:
        local_world_size = parallel_config.data_parallel_size_local * pp_size * tp_size
        base = parallel_config.data_parallel_rank * pp_size * tp_size
    placements = []
    for local_rank in range(local_world_size):
        global_rank = base + local_rank
        rank_in_dp = global_rank % (pp_size * tp_size)
        dp_rank = global_rank // (pp_size * tp_size)
        pp_rank, tp_rank = divmod(rank_in_dp, tp_size)
        placements.append((local_rank, dp_rank, pp_rank, tp_rank))
    return placements


def get_engine_dp_placement(vllm_config: VllmConfig, dp_rank: int) -> tuple[int, int]:
    """``(dp_size, dp_rank)`` as the engine fingerprints this DP rank.

    The engine runs the DP ranks of a non-MoE target as independent DP=1
    replicas, so their workers all report ``(1, 0)``.
    """
    if vllm_config.model_config.is_moe:
        return vllm_config.parallel_config.data_parallel_size, dp_rank
    return 1, 0


def get_draft_daemon_config(vllm_config: VllmConfig) -> VllmConfig | None:
    """VllmConfig for the draft daemon group"""
    if not is_draft_model_cacheable(vllm_config.speculative_config):
        return None
    assert vllm_config.speculative_config is not None
    return vllm_config.speculative_config.apply_draft_overrides(vllm_config)


def _reject_unsupported_parallelism(parallel_config: ParallelConfig) -> None:
    """Reject placements the daemon cannot map."""
    dp_size = parallel_config.data_parallel_size
    tp_size = parallel_config.tensor_parallel_size
    pp_size = parallel_config.pipeline_parallel_size
    if parallel_config.nnodes > 1:
        world_size = dp_size * pp_size * tp_size
        if world_size % parallel_config.nnodes != 0:
            raise ValueError(
                f"--nnodes ({parallel_config.nnodes}) must evenly divide the "
                f"daemon world size ({world_size} = data-parallel-size "
                f"{dp_size} x pipeline-parallel-size {pp_size} x "
                f"tensor-parallel-size {tp_size})"
            )
        return
    if dp_size == 1:
        return
    end_rank = (
        parallel_config.data_parallel_rank + parallel_config.data_parallel_size_local
    )
    if end_rank > dp_size:
        raise ValueError(
            "--data-parallel-start-rank + --data-parallel-size-local "
            f"({end_rank}) exceeds --data-parallel-size ({dp_size})"
        )
