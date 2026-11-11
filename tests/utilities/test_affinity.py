import ctypes
import pickle
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

from litdata.utilities import affinity

_WORD_BITS = ctypes.sizeof(ctypes.c_ulong) * 8
_MASK_WORDS = 1024 // _WORD_BITS


@pytest.fixture
def topology(tmp_path, monkeypatch):
    nodes = tmp_path / "nodes"
    for node, cpus in [(0, "0-3"), (1, "4-7")]:
        path = nodes / f"node{node}"
        path.mkdir(parents=True)
        (path / "cpulist").write_text(cpus)
    (nodes / "possible").write_text("0-1")
    pci = tmp_path / "pci_device"
    pci.mkdir(parents=True)
    (pci / "numa_node").write_text("1")
    monkeypatch.setattr(affinity, "_SYS_NODES", nodes)
    # PCI addresses contain colons, which Windows cannot create as directory names.
    pci_root = MagicMock()
    pci_root.__truediv__.side_effect = {"0000:81:00.0": pci}.__getitem__
    monkeypatch.setattr(affinity, "_SYS_PCI", pci_root)
    monkeypatch.setattr(affinity.sys, "platform", "linux")
    state = SimpleNamespace(cpus=set(range(8)), policy=(0, (0,) * _MASK_WORDS))
    monkeypatch.setattr(affinity.os, "sched_getaffinity", lambda pid: state.cpus.copy(), raising=False)
    monkeypatch.setattr(
        affinity.os, "sched_setaffinity", lambda pid, cpus: setattr(state, "cpus", set(cpus)), raising=False
    )
    monkeypatch.setattr(affinity, "_allowed_memory_nodes", lambda: {0, 1})
    monkeypatch.setattr(affinity, "_memory_library", Mock())
    monkeypatch.setattr(affinity, "_read_policy", lambda lib: state.policy)
    monkeypatch.setattr(affinity, "_write_policy", lambda lib, policy: setattr(state, "policy", policy))
    return state


@pytest.mark.parametrize(("value", "expected"), [("0-3,6,8-9\n", {0, 1, 2, 3, 6, 8, 9}), ("", set()), ("7", {7})])
def test_parse_list(value, expected):
    assert affinity._parse_list(value) == expected


@pytest.mark.parametrize("value", ["-1", "4-2", "1-2-3", "x"])
def test_invalid_ranges(value):
    with pytest.raises(ValueError, match="Invalid|invalid literal"):
        affinity._parse_list(value)


def test_plan_respects_current_mask_without_binding(topology):
    topology.cpus = {0, 4, 6}
    plan = affinity.get_numa_affinity(1)
    assert plan == affinity.NumaAffinity(1, (4, 6), True)
    assert topology.cpus == {0, 4, 6}
    assert topology.policy[0] == 0


def test_no_remote_cpu_fallback(topology):
    topology.cpus = {0, 1}
    with pytest.raises(RuntimeError, match="no CPUs"):
        affinity.get_numa_affinity(1)


@pytest.mark.parametrize("node", [-1, True, 1.5])
def test_invalid_node(topology, node):
    with pytest.raises(ValueError, match="nonnegative integer"):
        affinity.get_numa_affinity(node)


def test_missing_topology(topology):
    with pytest.raises(RuntimeError, match="topology is unavailable"):
        affinity.get_numa_affinity(9)


def test_non_linux_is_explicit(monkeypatch):
    monkeypatch.setattr(affinity.sys, "platform", "darwin")
    with pytest.raises(RuntimeError, match="requires Linux"):
        affinity.get_numa_affinity(0)


def test_bind_both_and_inherited_narrow_cpu_mask(topology):
    plan = affinity.get_numa_affinity(1)
    topology.cpus = {0}
    plan.bind()
    assert topology.cpus == {4, 5, 6, 7}
    assert topology.policy == (2, (2,) + (0,) * (_MASK_WORDS - 1))


def test_cpu_only_needs_no_libnuma(topology):
    affinity.get_numa_affinity(1, bind_memory=False).bind()
    assert topology.cpus == {4, 5, 6, 7}
    assert topology.policy[0] == 0
    affinity._memory_library.assert_not_called()


def test_worker_plan_is_picklable_and_never_calls_cuda(topology, monkeypatch):
    query = Mock(side_effect=AssertionError("CUDA must not be queried in the worker"))
    monkeypatch.setattr(torch.cuda, "get_device_properties", query)
    plan = pickle.loads(pickle.dumps(affinity.get_numa_affinity(1)))  # noqa: S301
    plan(3)
    assert topology.cpus == {4, 5, 6, 7}
    query.assert_not_called()


def test_memory_cpuset_denial_does_not_change_cpu_mask(topology, monkeypatch):
    monkeypatch.setattr(affinity, "_allowed_memory_nodes", lambda: {0})
    with pytest.raises(RuntimeError, match="outside the allowed"):
        affinity.get_numa_affinity(1).bind()
    assert topology.cpus == set(range(8))
    assert topology.policy[0] == 0


def test_memory_failure_restores_cpu_and_policy(topology, monkeypatch):
    original = topology.policy

    def write_policy(lib, policy):
        if policy[0] == 2:
            raise OSError("binding denied")
        topology.policy = policy

    monkeypatch.setattr(affinity, "_write_policy", write_policy)
    with pytest.raises(OSError, match="binding denied"):
        affinity.get_numa_affinity(1).bind()
    assert topology.cpus == set(range(8))
    assert topology.policy == original


def test_policy_verification_failure_restores_state(topology, monkeypatch):
    monkeypatch.setattr(affinity, "_write_policy", lambda lib, policy: None)
    with pytest.raises(RuntimeError, match="effective NUMA memory policy"):
        affinity.get_numa_affinity(1).bind()
    assert topology.cpus == set(range(8))
    assert topology.policy[0] == 0


def test_partial_cpu_binding_restores_state(topology, monkeypatch):
    original = topology.cpus.copy()

    def set_affinity(pid, cpus):
        topology.cpus = set(cpus) if set(cpus) == original else {4}

    monkeypatch.setattr(affinity.os, "sched_setaffinity", set_affinity)
    with pytest.raises(RuntimeError, match="complete CPU affinity mask"):
        affinity.get_numa_affinity(1).bind()
    assert topology.cpus == original
    assert topology.policy[0] == 0


def test_plan_rejects_nonlocal_cpus(topology):
    with pytest.raises(ValueError, match="Every selected CPU"):
        affinity.NumaAffinity(1, (0, 4)).bind()


def test_gpu_lookup_uses_logical_device_properties(topology, monkeypatch):
    query = Mock(return_value=SimpleNamespace(pci_domain_id=0, pci_bus_id=129, pci_device_id=0))
    monkeypatch.setattr(torch.cuda, "get_device_properties", query)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
    assert affinity.get_gpu_affinity(1) == affinity.NumaAffinity(1, (4, 5, 6, 7))
    query.assert_called_once_with(1)
    affinity._SYS_PCI.__truediv__.assert_called_once_with("0000:81:00.0")


def test_older_torch_has_explicit_node_fallback(topology, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace())
    with pytest.raises(RuntimeError, match="get_numa_affinity"):
        affinity.get_gpu_affinity(0)


def test_unknown_gpu_numa_node(topology, monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(pci_domain_id=0, pci_bus_id=129, pci_device_id=0),
    )
    (affinity._SYS_PCI / "0000:81:00.0" / "numa_node").write_text("-1")
    with pytest.raises(RuntimeError, match="no NUMA node exposed"):
        affinity.get_gpu_affinity(0)


def test_gpu_discovery_is_rejected_in_worker(topology, monkeypatch):
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: object())
    query = Mock()
    monkeypatch.setattr(torch.cuda, "get_device_properties", query)
    with pytest.raises(RuntimeError, match="rank process"):
        affinity.get_gpu_affinity(0)
    query.assert_not_called()


@pytest.mark.parametrize("word_type", [ctypes.c_uint32, ctypes.c_uint64])
def test_ctypes_policy_roundtrip(tmp_path, monkeypatch, word_type):
    monkeypatch.setattr(affinity, "ctypes", SimpleNamespace(**(vars(ctypes) | {"c_ulong": word_type})))
    word_bits = ctypes.sizeof(word_type) * 8
    mask_words = 1024 // word_bits
    (tmp_path / "possible").write_text("0-130")
    monkeypatch.setattr(affinity, "_SYS_NODES", tmp_path)
    saved = {}

    def set_policy(mode, mask, maxnode):
        saved.update(
            mode=mode,
            words=tuple(mask[i] for i in range(maxnode // word_bits)) if mask else (0,) * (maxnode // word_bits),
        )
        return 0

    def get_policy(mode, mask, maxnode, address, flags):
        ctypes.cast(mode, ctypes.POINTER(ctypes.c_int))[0] = saved["mode"]
        for i, word in enumerate(saved["words"]):
            mask[i] = word
        return 0

    lib = SimpleNamespace(set_mempolicy=set_policy, get_mempolicy=get_policy)
    words = [0] * mask_words
    words[130 // word_bits] = 1 << (130 % word_bits)
    policy = (2, tuple(words))
    affinity._write_policy(lib, policy)
    assert affinity._read_policy(lib) == policy
    affinity._write_policy(lib, (0, (0,) * mask_words))
    assert affinity._read_policy(lib) == (0, (0,) * mask_words)


@pytest.mark.parametrize("operation", ["read", "write"])
def test_ctypes_errno_is_preserved(tmp_path, monkeypatch, operation):
    (tmp_path / "possible").write_text("0-1")
    monkeypatch.setattr(affinity, "_SYS_NODES", tmp_path)
    ctypes.set_errno(1)
    lib = SimpleNamespace(get_mempolicy=lambda *args: -1, set_mempolicy=lambda *args: -1)
    call = (
        (lambda: affinity._read_policy(lib))
        if operation == "read"
        else (lambda: affinity._write_policy(lib, (2, (1,) * _MASK_WORDS)))
    )
    with pytest.raises(OSError, match="NUMA memory policy") as error:
        call()
    assert error.value.errno == 1
