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
import os

import torch.distributed as dist

from litdata.utilities.env import _DistributedEnv


def maybe_barrier() -> None:
    """Synchronizes all processes in distributed training if PyTorch's distributed package is available and initialized.

    This function checks if the PyTorch distributed package is both available and initialized.

    If so, it calls `dist.barrier()` to synchronize all processes.

    This is useful in distributed training to ensure that all processes reach a certain point in the code before
    proceeding.
    """
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def is_local_rank_0() -> bool:
    """Checks if the current process has local rank 0."""
    local_rank = os.environ.get("LOCAL_RANK", None)  # env variable set by torchrun
    if local_rank is not None:
        return int(local_rank) == 0

    env = _DistributedEnv.detect()

    # condition might not work as expected if num of processes is not equal on each nodes
    return (env.num_nodes == 1 and env.global_rank == 0) or (env.num_nodes > 1 and env.global_rank % env.num_nodes == 0)
