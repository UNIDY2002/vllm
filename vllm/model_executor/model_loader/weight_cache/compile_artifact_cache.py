# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable manifests and bundles for compiled runtime artifacts.

The weight-cache daemon's JSON artifact protocol is intentionally bounded and
is suitable for autotune tables, not hundreds of megabytes of compiler cache.
This module describes the cache independently of the transport so a future
Mooncake data plane can move one validated bundle without exposing arbitrary
filesystem paths.
"""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from vllm.logger import init_logger

logger = init_logger(__name__)

_CHUNK_SIZE = 1 << 20
_MAX_BUNDLE_BYTES = 8 << 30


@dataclass(frozen=True)
class CompileCacheRoot:
    """A named cache root included in a portable bundle."""

    name: str
    path: Path


@dataclass(frozen=True)
class CompileCacheEntry:
    """One regular file in a compile cache."""

    root: str
    relative_path: str
    size: int
    sha256: str
    mtime_ns: int


@dataclass(frozen=True)
class CompileCacheManifest:
    """Checksummed inventory used to validate a cache before installation."""

    entries: tuple[CompileCacheEntry, ...]
    total_size: int
    root_paths: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "total_size": self.total_size,
            "root_paths": dict(self.root_paths),
            "entries": [
                {
                    "root": entry.root,
                    "path": entry.relative_path,
                    "size": entry.size,
                    "sha256": entry.sha256,
                    "mtime_ns": entry.mtime_ns,
                }
                for entry in self.entries
            ],
        }


def compile_cache_manifest_from_dict(value: object) -> CompileCacheManifest:
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("unsupported compile cache manifest")
    raw_entries = value.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("compile cache manifest entries must be a list")
    entries: list[CompileCacheEntry] = []
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise ValueError("malformed compile cache manifest entry")
        root = raw.get("root")
        relative_path = raw.get("path")
        size = raw.get("size")
        sha256 = raw.get("sha256")
        mtime_ns = raw.get("mtime_ns")
        if (
            not isinstance(root, str)
            or not isinstance(relative_path, str)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or not isinstance(mtime_ns, int)
            or mtime_ns < 0
        ):
            raise ValueError("malformed compile cache manifest entry")
        _safe_relative_path(relative_path)
        entries.append(CompileCacheEntry(root, relative_path, size, sha256, mtime_ns))
    total_size = value.get("total_size")
    raw_root_paths = value.get("root_paths")
    if not isinstance(raw_root_paths, dict) or not all(
        isinstance(name, str) and isinstance(path, str)
        for name, path in raw_root_paths.items()
    ):
        raise ValueError("malformed compile cache root paths")
    if not isinstance(total_size, int) or total_size != sum(
        entry.size for entry in entries
    ):
        raise ValueError("compile cache manifest size mismatch")
    return CompileCacheManifest(
        tuple(entries), total_size, tuple(sorted(raw_root_paths.items()))
    )


def get_default_compile_cache_roots() -> tuple[CompileCacheRoot, ...]:
    """Return explicitly configured FlashInfer cache roots.

    The roots are discovered without importing a model or creating a CUDA
    context. Empty roots are omitted; callers can therefore use this before
    daemon model loading to install a source node's bundle.
    """
    roots: list[CompileCacheRoot] = []
    seen: set[Path] = set()
    for name, variable in (
        ("flashinfer-cubins", "FLASHINFER_CUBIN_DIR"),
        ("flashinfer-workspace", "FLASHINFER_WORKSPACE_BASE"),
    ):
        value = os.environ.get(variable)
        if not value:
            continue
        path = Path(value).expanduser()
        if path in seen:
            continue
        seen.add(path)
        roots.append(CompileCacheRoot(name, path))
    return tuple(roots)


def _safe_relative_path(path: str) -> PurePosixPath:
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe cache path: {path!r}")
    return candidate


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def build_compile_cache_manifest(
    roots: Iterable[CompileCacheRoot],
) -> CompileCacheManifest:
    """Build a deterministic manifest of regular files under ``roots``.

    Symlinks, sockets, lock files, and temporary files are excluded. The
    compiler state files such as ``.ninja_log`` remain included because they
    are required for a restored cache to be a real cache hit.
    """
    roots = tuple(roots)
    entries: list[CompileCacheEntry] = []
    total_size = 0
    seen_roots: set[str] = set()
    for root in roots:
        if not root.name or root.name in seen_roots:
            raise ValueError(f"duplicate or empty cache root name: {root.name!r}")
        _safe_relative_path(root.name)
        if "/" in root.name:
            raise ValueError(
                f"cache root name must be one path component: {root.name!r}"
            )
        seen_roots.add(root.name)
        if not root.path.is_dir():
            continue
        for path in sorted(root.path.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            if path.name.endswith(".lock") or path.name.endswith(".tmp"):
                continue
            relative = path.relative_to(root.path).as_posix()
            _safe_relative_path(relative)
            stat = path.stat()
            size = stat.st_size
            total_size += size
            if total_size > _MAX_BUNDLE_BYTES:
                raise ValueError(f"compile cache exceeds {_MAX_BUNDLE_BYTES} bytes")
            entries.append(
                CompileCacheEntry(
                    root.name, relative, size, _file_sha256(path), stat.st_mtime_ns
                )
            )
    return CompileCacheManifest(
        tuple(entries),
        total_size,
        tuple(sorted((root.name, str(root.path)) for root in roots)),
    )


def _archive_name(entry: CompileCacheEntry) -> str:
    _safe_relative_path(entry.relative_path)
    return f"{entry.root}/{entry.relative_path}"


def create_compile_cache_bundle(
    roots: Iterable[CompileCacheRoot],
    manifest: CompileCacheManifest | None = None,
) -> tuple[CompileCacheManifest, bytes]:
    """Create a deterministic tar bundle and its manifest."""
    roots_by_name = {root.name: root for root in roots}
    if manifest is None:
        manifest = build_compile_cache_manifest(roots_by_name.values())
    if dict(manifest.root_paths) != {
        name: str(root.path) for name, root in roots_by_name.items()
    }:
        raise ValueError("compile cache root paths do not match manifest")
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for entry in manifest.entries:
            root = roots_by_name.get(entry.root)
            if root is None:
                raise ValueError(f"manifest references unknown root {entry.root!r}")
            path = root.path / _safe_relative_path(entry.relative_path)
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"manifest file is missing: {path}")
            if path.stat().st_size != entry.size:
                raise ValueError(f"manifest size mismatch: {path}")
            info = tarfile.TarInfo(_archive_name(entry))
            info.size = entry.size
            info.mode = 0o600
            info.mtime = entry.mtime_ns / 1e9
            with path.open("rb") as stream:
                archive.addfile(info, stream)
    bundle = output.getvalue()
    if len(bundle) > _MAX_BUNDLE_BYTES:
        raise ValueError(f"bundle exceeds {_MAX_BUNDLE_BYTES} bytes")
    return manifest, bundle


def install_compile_cache_bundle(
    bundle: bytes | BinaryIO,
    manifest: CompileCacheManifest,
    roots: Iterable[CompileCacheRoot],
) -> None:
    """Validate and atomically install a bundle into destination cache roots."""
    roots_by_name = {root.name: root for root in roots}
    if dict(manifest.root_paths) != {
        name: str(root.path) for name, root in roots_by_name.items()
    }:
        raise ValueError("compile cache root paths do not match manifest")
    expected = {_archive_name(entry): entry for entry in manifest.entries}
    staged: list[tuple[Path, Path, CompileCacheEntry]] = []
    stream: BinaryIO = io.BytesIO(bundle) if isinstance(bundle, bytes) else bundle
    try:
        with tarfile.open(fileobj=stream, mode="r:") as archive:
            members = archive.getmembers()
            if len(members) != len(expected):
                raise ValueError("compile cache bundle entry count mismatch")
            seen_members: set[str] = set()
            for member in members:
                entry = expected.get(member.name)
                if entry is None or member.name in seen_members or not member.isfile():
                    raise ValueError(f"unexpected bundle member: {member.name!r}")
                seen_members.add(member.name)
                root_name, relative = member.name.split("/", 1)
                root = roots_by_name.get(root_name)
                if root is None:
                    raise ValueError(f"unknown bundle root: {root_name!r}")
                destination = root.path / _safe_relative_path(relative)
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"cannot read bundle member: {member.name!r}")
                data = extracted.read()
                if len(data) != entry.size:
                    raise ValueError(f"bundle size mismatch: {member.name!r}")
                if hashlib.sha256(data).hexdigest() != entry.sha256:
                    raise ValueError(f"bundle checksum mismatch: {member.name!r}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + ".tmp")
                temporary.write_bytes(data)
                staged.append((temporary, destination, entry))
            if seen_members != set(expected):
                raise ValueError("compile cache bundle entries do not match manifest")
        for temporary, destination, entry in staged:
            os.replace(temporary, destination)
            os.utime(destination, ns=(entry.mtime_ns, entry.mtime_ns))
    except Exception:
        for temporary, _, _ in staged:
            temporary.unlink(missing_ok=True)
        raise
