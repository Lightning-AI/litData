# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

_SYS_NODES = Path("/sys/devices/system/node")
_SYS_PCI = Path("/sys/bus/pci/devices")


def _require_linux() -> None:
    if sys.platform != "linux" or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("NUMA affinity requires Linux with sched_setaffinity support.")


def _parse_list(value: str) -> set[int]:
    result: set[int] = set()
    for part in value.strip().split(","):
        if not part:
            continue
        bounds = part.split("-")
        if len(bounds) > 2:
            raise ValueError(f"Invalid CPU or NUMA node range: {part!r}")
        start, end = int(bounds[0]), int(bounds[-1])
        if start < 0 or end < start:
            raise ValueError(f"Invalid CPU or NUMA node range: {part!r}")
        result.update(range(start, end + 1))
    return result


def _node_cpus(node: int) -> set[int]:
    if isinstance(node, bool) or not isinstance(node, int) or node < 0:
        raise ValueError("numa_node must be a nonnegative integer.")
    try:
        return _parse_list((_SYS_NODES / f"node{node}" / "cpulist").read_text())
    except OSError as ex:
        raise RuntimeError(f"CPU topology is unavailable for NUMA node {node}.") from ex


def _allowed_memory_nodes() -> set[int]:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("Mems_allowed_list:"):
            return _parse_list(line.split(":", 1)[1])
    raise RuntimeError("Cannot determine the allowed NUMA memory nodes from /proc/self/status.")


def _memory_library() -> Any:
    try:
        library = ctypes.CDLL("libnuma.so.1", use_errno=True)
    except OSError as ex:
        raise RuntimeError("NUMA memory binding requires libnuma.so.1 (the libnuma1 system package).") from ex
    pointer = ctypes.POINTER(ctypes.c_ulong)
    library.get_mempolicy.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        pointer,
        ctypes.c_ulong,
        ctypes.c_void_p,
        ctypes.c_ulong,
    ]
    library.get_mempolicy.restype = ctypes.c_int
    library.set_mempolicy.argtypes = [ctypes.c_int, pointer, ctypes.c_ulong]
    library.set_mempolicy.restype = ctypes.c_int
    return library


def _node_mask() -> Any:
    bits = ctypes.sizeof(ctypes.c_ulong) * 8
    possible = _parse_list((_SYS_NODES / "possible").read_text())
    count = max(1024, max(possible, default=0) + 1)
    return (ctypes.c_ulong * ((count + bits - 1) // bits))()


def _read_policy(library: Any) -> tuple[int, tuple[int, ...]]:
    mask = _node_mask()
    mode = ctypes.c_int()
    if library.get_mempolicy(ctypes.byref(mode), mask, ctypes.sizeof(mask) * 8, None, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, f"Cannot read NUMA memory policy: {os.strerror(error)}")
    return mode.value, tuple(mask)


def _write_policy(library: Any, policy: tuple[int, tuple[int, ...]]) -> None:
    mode, words = policy
    mask = (ctypes.c_ulong * len(words))(*words)
    pointer = mask if any(words) else None
    if library.set_mempolicy(mode, pointer, ctypes.sizeof(mask) * 8) != 0:
        error = ctypes.get_errno()
        raise OSError(error, f"Cannot set NUMA memory policy: {os.strerror(error)}")


@dataclass(frozen=True)
class NumaAffinity:
    """A picklable CPU and optional memory policy for one NUMA node.

    Obtain a plan with :func:`get_gpu_affinity` or :func:`get_numa_affinity`.
    Call :meth:`bind` in a rank before creating its workers, and pass the same
    object as ``worker_init_fn`` to either PyTorch's or LitData's DataLoader.
    Binding affects the calling thread and future child threads/processes.
    Existing sibling threads and previously allocated pages are unchanged.
    """

    numa_node: int
    cpus: tuple[int, ...]
    bind_memory: bool = True

    def bind(self) -> None:
        """Apply and verify the policy; raise if binding is unavailable or denied."""
        _require_linux()
        local_cpus = _node_cpus(self.numa_node)
        if not self.cpus or any(isinstance(cpu, bool) or not isinstance(cpu, int) for cpu in self.cpus):
            raise ValueError("cpus must contain integer CPU identifiers.")
        if not set(self.cpus) <= local_cpus:
            raise ValueError("Every selected CPU must belong to the selected NUMA node.")
        previous_cpus = os.sched_getaffinity(0)
        library = None
        previous_policy = None
        desired_policy = None
        if self.bind_memory:
            if self.numa_node not in _allowed_memory_nodes():
                raise RuntimeError(f"NUMA node {self.numa_node} is outside the allowed memory nodes.")
            library = _memory_library()
            previous_policy = _read_policy(library)
            mask = _node_mask()
            bits = ctypes.sizeof(ctypes.c_ulong) * 8
            mask[self.numa_node // bits] |= 1 << (self.numa_node % bits)
            desired_policy = (2, tuple(mask))  # MPOL_BIND
        try:
            os.sched_setaffinity(0, self.cpus)
            if os.sched_getaffinity(0) != set(self.cpus):
                raise RuntimeError("The kernel could not apply the complete CPU affinity mask.")
            if library is not None and desired_policy is not None:
                _write_policy(library, desired_policy)
                if _read_policy(library) != desired_policy:
                    raise RuntimeError("The effective NUMA memory policy does not match the requested node.")
        except Exception:
            try:
                if library is not None and previous_policy is not None:
                    _write_policy(library, previous_policy)
            finally:
                os.sched_setaffinity(0, previous_cpus)
            raise

    def __call__(self, worker_id: int) -> None:
        """Bind a DataLoader worker without querying or initializing CUDA."""
        self.bind()


def get_numa_affinity(numa_node: int, *, bind_memory: bool = True) -> NumaAffinity:
    """Plan binding to an explicit node, restricted to the calling thread's allowed CPUs.

    This performs no binding and does not initialize CUDA. Nodes may serve more
    than one GPU. An empty intersection raises rather than choosing remote CPUs.
    """
    _require_linux()
    cpus = _node_cpus(numa_node) & os.sched_getaffinity(0)
    if not cpus:
        raise RuntimeError(f"NUMA node {numa_node} has no CPUs in the current affinity mask.")
    return NumaAffinity(numa_node, tuple(sorted(cpus)), bind_memory)


def get_gpu_affinity(device: "torch.device | str | int | None" = None, *, bind_memory: bool = True) -> NumaAffinity:
    """Plan NUMA binding for a logical CUDA device using its PCI topology.

    Resolve this in the rank process, before creating DataLoader workers. The
    PyTorch device lookup honors CUDA visibility and initializes CUDA. Pass the
    returned object to workers instead of calling this function inside them.

    Requires a PyTorch build exposing PCI identifiers in CUDA device properties.
    With older builds, use :func:`get_numa_affinity` and an explicit node instead.
    """
    _require_linux()
    import torch

    if torch.utils.data.get_worker_info() is not None:
        raise RuntimeError("Resolve GPU affinity in the rank process and pass the resulting plan to workers.")
    properties = torch.cuda.get_device_properties(device)
    try:
        pci = f"{properties.pci_domain_id:04x}:{properties.pci_bus_id:02x}:{properties.pci_device_id:02x}.0"
    except AttributeError as ex:
        raise RuntimeError(
            "This PyTorch build does not expose GPU PCI identifiers. "
            "Use get_numa_affinity(numa_node) with an explicit node or upgrade PyTorch."
        ) from ex
    try:
        node = int((_SYS_PCI / pci / "numa_node").read_text())
    except OSError as ex:
        raise RuntimeError(f"NUMA topology is unavailable for GPU PCI device {pci}.") from ex
    if node < 0:
        raise RuntimeError(f"GPU PCI device {pci} has no NUMA node exposed by the kernel.")
    return get_numa_affinity(node, bind_memory=bind_memory)
