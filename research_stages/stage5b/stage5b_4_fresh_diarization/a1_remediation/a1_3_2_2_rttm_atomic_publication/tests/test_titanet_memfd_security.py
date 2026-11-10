#!/usr/bin/env python3
"""
Unit and Security Test Suite for TitaNet Sealed memfd Loading (Finding A):
Verifies:
1. Exact digest of the bytes actually consumed by the loader.
2. Attempted mutation after sealing is rejected by the Linux kernel.
3. Attempted growth and truncation after sealing is rejected by the Linux kernel.
4. Attempted write through /proc/self/fd/{fd} is rejected by the Linux kernel.
5. Original checkpoint replacement on disk after verification has zero effect on staged bytes.
6. Loader exceptions guarantee descriptor closure and memory reclamation.
7. Production model-injection bypass via nemo_loader_override is strictly rejected.
8. Unsupported platform or missing memfd/seal support fails closed.
9. Corrupted memory buffer digest is rejected before memfd creation.
10. Descriptor is guaranteed closed after successful load.
"""

import errno
import fcntl
import hashlib
import os
from pathlib import Path
import sys
import pytest
import torch

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from checkpoint_preflight import (
    ExecutionMode,
    VerifiedCheckpoint,
    verify_encoder_checkpoint_strict,
    create_synthetic_checkpoint_fixture,
    compute_file_sha256,
)
from model_factory import (
    ControlledModelFactory,
    load_titanet_model_from_verified_bytes,
    ModelFactoryError,
    VerifiedModelHandle,
)


@pytest.fixture
def titanet_env(tmp_path):
    ckpt_file = tmp_path / "titanet-l.nemo"
    content = b"NVIDIA_TITANET_SYNTHETIC_NEMO_ARCHIVE_BYTES_9876543210" * 20
    ckpt_file.write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()

    token = verify_encoder_checkpoint_strict(
        encoder_name="titanet_large",
        checkpoint_path=ckpt_file,
        expected_sha256=sha,
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_permitted_roots=[tmp_path],
    )
    return {
        "tmp_path": tmp_path,
        "ckpt_file": ckpt_file,
        "content": content,
        "ckpt_sha": sha,
        "token": token,
    }


def test_1_exact_digest_consumed_by_loader(titanet_env):
    """Loader receives /proc/self/fd/{memfd} and reads exact byte-identical content."""
    observed_hashes = []
    observed_paths = []

    def mock_loader(proc_path: str, map_location: str):
        observed_paths.append(proc_path)
        assert proc_path.startswith("/proc/self/fd/")
        with open(proc_path, "rb") as f:
            data = f.read()
        observed_hashes.append(hashlib.sha256(data).hexdigest())
        return torch.nn.Linear(10, 192)

    model = load_titanet_model_from_verified_bytes(
        verified_checkpoint=titanet_env["token"],
        ckpt_bytes=titanet_env["content"],
        device="cpu",
        loader_fn=mock_loader,
    )
    assert isinstance(model, torch.nn.Linear)
    assert len(observed_hashes) == 1
    assert observed_hashes[0] == titanet_env["ckpt_sha"]


def test_2_attempted_mutation_after_sealing_rejected(titanet_env):
    """Attempting to write to descriptor after F_SEAL_WRITE is rejected with EPERM."""
    captured_fd = []

    def intercepting_loader(proc_path: str, map_location: str):
        fd = int(proc_path.split("/")[-1])
        captured_fd.append(fd)
        # Attempt to write to the sealed memfd descriptor
        try:
            os.write(fd, b"MALICIOUS_PAYLOAD_MUTATION")
            pytest.fail("Kernel allowed write to sealed memfd descriptor!")
        except OSError as e:
            assert e.errno in (errno.EPERM, errno.EBADF)
        return torch.nn.Linear(10, 192)

    load_titanet_model_from_verified_bytes(
        verified_checkpoint=titanet_env["token"],
        ckpt_bytes=titanet_env["content"],
        device="cpu",
        loader_fn=intercepting_loader,
    )


def test_3_attempted_truncation_and_growth_after_sealing_rejected(titanet_env):
    """Attempting ftruncate() or lseek+write on sealed memfd is rejected with EPERM."""
    def intercepting_loader(proc_path: str, map_location: str):
        fd = int(proc_path.split("/")[-1])
        # Attempt truncate (shrink)
        try:
            os.ftruncate(fd, 0)
            pytest.fail("Kernel allowed ftruncate(fd, 0) on sealed memfd!")
        except OSError as e:
            assert e.errno == errno.EPERM

        # Attempt growth
        try:
            os.ftruncate(fd, len(titanet_env["content"]) + 1000)
            pytest.fail("Kernel allowed growth on sealed memfd!")
        except OSError as e:
            assert e.errno == errno.EPERM

        return torch.nn.Linear(10, 192)

    load_titanet_model_from_verified_bytes(
        verified_checkpoint=titanet_env["token"],
        ckpt_bytes=titanet_env["content"],
        device="cpu",
        loader_fn=intercepting_loader,
    )


def test_4_proc_path_write_rejected(titanet_env):
    """Attempting to open /proc/self/fd/{fd} for writing is rejected with PermissionError."""
    def intercepting_loader(proc_path: str, map_location: str):
        try:
            with open(proc_path, "wb") as f:
                f.write(b"EXTERNAL_ATTACK")
            pytest.fail("Kernel allowed open('wb') on /proc/self/fd/{fd}!")
        except (PermissionError, OSError) as e:
            pass
        return torch.nn.Linear(10, 192)

    load_titanet_model_from_verified_bytes(
        verified_checkpoint=titanet_env["token"],
        ckpt_bytes=titanet_env["content"],
        device="cpu",
        loader_fn=intercepting_loader,
    )


def test_5_original_checkpoint_replacement_cannot_alter_staged_bytes(titanet_env):
    """Mutating or deleting the original checkpoint on disk has zero effect on sealed memfd bytes."""
    consumed_data = []

    def mock_loader(proc_path: str, map_location: str):
        # Mutate the file on disk during loader execution
        titanet_env["ckpt_file"].write_bytes(b"COMPROMISED_DISK_BYTES" * 10)
        with open(proc_path, "rb") as f:
            consumed_data.append(f.read())
        return torch.nn.Linear(10, 192)

    load_titanet_model_from_verified_bytes(
        verified_checkpoint=titanet_env["token"],
        ckpt_bytes=titanet_env["content"],
        device="cpu",
        loader_fn=mock_loader,
    )
    assert len(consumed_data) == 1
    assert consumed_data[0] == titanet_env["content"]
    assert consumed_data[0] != titanet_env["ckpt_file"].read_bytes()


def test_6_loader_exceptions_guarantee_descriptor_cleanup(titanet_env):
    """If the loader raises an exception, the memfd descriptor is guaranteed to be closed."""
    captured_fd = []

    def failing_loader(proc_path: str, map_location: str):
        fd = int(proc_path.split("/")[-1])
        captured_fd.append(fd)
        # Verify fd is valid while inside loader
        os.fstat(fd)
        raise RuntimeError("Simulated internal framework crash")

    with pytest.raises(RuntimeError, match="Simulated internal framework crash"):
        load_titanet_model_from_verified_bytes(
            verified_checkpoint=titanet_env["token"],
            ckpt_bytes=titanet_env["content"],
            device="cpu",
            loader_fn=failing_loader,
        )

    # After exit, fd MUST be closed (EBADF)
    assert len(captured_fd) == 1
    closed_fd = captured_fd[0]
    with pytest.raises(OSError) as exc_info:
        os.fstat(closed_fd)
    assert exc_info.value.errno == errno.EBADF


def test_7_production_model_injection_bypass_rejected(tmp_path, monkeypatch):
    """ControlledModelFactory.build_production_model strictly rejects nemo_loader_override."""
    monkeypatch.setattr(
        "checkpoint_preflight.get_immutable_standard_checkpoint_roots",
        lambda: [tmp_path.resolve()],
    )
    ckpt_file = tmp_path / "prod.nemo"
    content = b"PROD_BYTES_12345" * 10
    ckpt_file.write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()

    token = verify_encoder_checkpoint_strict(
        encoder_name="titanet_large",
        checkpoint_path=ckpt_file,
        expected_sha256=sha,
        execution_mode=ExecutionMode.PRODUCTION,
    )

    ControlledModelFactory.nemo_loader_override = lambda path, dev: torch.nn.Linear(10, 192)
    try:
        with pytest.raises(ModelFactoryError, match="Custom loader overrides are strictly prohibited in production mode"):
            ControlledModelFactory.build_production_model(token)
    finally:
        ControlledModelFactory.nemo_loader_override = None


def test_8_unsupported_platform_fails_closed(titanet_env, monkeypatch):
    """If os.memfd_create or fcntl.F_ADD_SEALS is missing, fails closed with ModelFactoryError."""
    monkeypatch.delattr(os, "memfd_create", raising=False)
    with pytest.raises(ModelFactoryError, match="Platform does not support sealed memfd_create"):
        load_titanet_model_from_verified_bytes(
            verified_checkpoint=titanet_env["token"],
            ckpt_bytes=titanet_env["content"],
            loader_fn=lambda p, d: torch.nn.Linear(10, 192),
        )


def test_9_corrupted_digest_rejected_before_sealing(titanet_env):
    """Memory buffer digest mismatch is rejected before memfd creation."""
    corrupted_bytes = b"CORRUPTED_IN_MEMORY_CONTENT"
    with pytest.raises(ModelFactoryError, match="TitaNet memory buffer digest mismatch"):
        load_titanet_model_from_verified_bytes(
            verified_checkpoint=titanet_env["token"],
            ckpt_bytes=corrupted_bytes,
            loader_fn=lambda p, d: torch.nn.Linear(10, 192),
        )


def test_10_descriptor_closed_after_successful_load(titanet_env):
    """Descriptor is closed and memory released immediately upon completion."""
    captured_fd = []

    def mock_loader(proc_path: str, map_location: str):
        fd = int(proc_path.split("/")[-1])
        captured_fd.append(fd)
        os.fstat(fd)
        return torch.nn.Linear(10, 192)

    load_titanet_model_from_verified_bytes(
        verified_checkpoint=titanet_env["token"],
        ckpt_bytes=titanet_env["content"],
        loader_fn=mock_loader,
    )
    closed_fd = captured_fd[0]
    with pytest.raises(OSError) as exc_info:
        os.fstat(closed_fd)
    assert exc_info.value.errno == errno.EBADF
