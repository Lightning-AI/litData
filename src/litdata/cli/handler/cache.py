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

import shutil
from argparse import Namespace

from litdata.utilities.dataset_utilities import get_default_cache_dir


def clear_cache(args: Namespace) -> None:
    """Clear default cache used for StreamingDataset and other utilities."""
    streaming_default_cache_dir = get_default_cache_dir()

    shutil.rmtree(streaming_default_cache_dir, ignore_errors=True)

    print(f"Cache directory '{streaming_default_cache_dir}' cleared.")


def show_cache_path(args: Namespace) -> None:
    """Show the path to the cache directory."""
    streaming_default_cache_dir = get_default_cache_dir()
    print(f"Default cache directory: {streaming_default_cache_dir}")
