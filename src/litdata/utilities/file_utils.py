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
from contextlib import suppress

from filelock import FileLock, Timeout


def increment_file_count(file_path: str) -> int:
    """Increment the file count in the index file."""
    countpath = file_path + ".cnt"
    with suppress(Timeout), FileLock(countpath + ".lock", timeout=1):
        try:
            with open(countpath) as count_f:
                curr_count = int(count_f.read().strip())
        except Exception:
            curr_count = 0
        curr_count += 1
        with open(countpath, "w+") as count_f:
            count_f.write(str(curr_count))

    return curr_count


def decrement_file_count(file_path: str) -> int:
    """Decrement the file count in the index file."""
    countpath = file_path + ".cnt"

    with suppress(Timeout), FileLock(countpath + ".lock", timeout=1):
        try:
            with open(countpath) as count_f:
                curr_count = int(count_f.read().strip())
        except Exception as e:
            raise ValueError(f"Count file not found when trying to decrement_file_count: {countpath}.") from e
        curr_count -= 1

        if curr_count <= 0:
            # remove the count file if it reaches zero
            with suppress(FileNotFoundError):
                os.remove(countpath)
            return 0
        with open(countpath, "w+") as count_f:
            count_f.write(str(curr_count))

    return curr_count
