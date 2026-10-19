# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest

from vllm.model_executor.model_loader.weight_cache.compile_artifact_cache import (
    CompileCacheRoot,
    build_compile_cache_manifest,
    create_compile_cache_bundle,
    install_compile_cache_bundle,
)


def test_compile_cache_bundle_round_trip_preserves_manifest_files(tmp_path: Path):
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "nested" / ".ninja_log").write_bytes(b"build state")
    (source / "kernel.so").write_bytes(b"compiled kernel")
    (source / "ignored.lock").write_bytes(b"lock")

    roots = [CompileCacheRoot("flashinfer", source)]
    manifest, bundle = create_compile_cache_bundle(roots)
    assert manifest.total_size == len(b"build state") + len(b"compiled kernel")
    assert [entry.relative_path for entry in manifest.entries] == [
        "kernel.so",
        "nested/.ninja_log",
    ]

    (source / "kernel.so").unlink()
    (source / "nested" / ".ninja_log").unlink()
    install_compile_cache_bundle(
        bundle, manifest, [CompileCacheRoot("flashinfer", source)]
    )
    assert (source / "kernel.so").read_bytes() == b"compiled kernel"
    assert (source / "nested" / ".ninja_log").read_bytes() == b"build state"
    assert not (source / "ignored.lock").exists()


def test_compile_cache_bundle_requires_the_same_absolute_root(tmp_path: Path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "kernel.so").write_bytes(b"compiled kernel")
    manifest, bundle = create_compile_cache_bundle(
        [CompileCacheRoot("flashinfer", source)]
    )

    with pytest.raises(ValueError, match="root paths do not match"):
        install_compile_cache_bundle(
            bundle, manifest, [CompileCacheRoot("flashinfer", target)]
        )


def test_compile_cache_bundle_rejects_checksum_mismatch(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "kernel.so").write_bytes(b"compiled kernel")
    manifest, bundle = create_compile_cache_bundle(
        [CompileCacheRoot("flashinfer", source)]
    )
    broken = bundle.replace(b"compiled kernel", b"broken kernel")
    with pytest.raises(ValueError, match="checksum mismatch"):
        install_compile_cache_bundle(
            broken,
            manifest,
            [CompileCacheRoot("flashinfer", source)],
        )


def test_compile_cache_manifest_skips_symlinks_and_lock_files(tmp_path: Path):
    root = tmp_path / "cache"
    root.mkdir()
    (root / "kernel.so").write_bytes(b"kernel")
    (root / "kernel.lock").write_bytes(b"lock")
    (root / "link").symlink_to(root / "kernel.so")
    manifest = build_compile_cache_manifest([CompileCacheRoot("root", root)])
    assert [entry.relative_path for entry in manifest.entries] == ["kernel.so"]
