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

from argparse import _SubParsersAction

from litdata.cli.handler.cache import clear_cache, show_cache_path
from litdata.cli.handler.optimize import optimize_dataset
from litdata.cli.parser import LitFormatter


def register_cache_subcommand(subparser: _SubParsersAction) -> None:
    """Add the cache subcommand to the main parser."""
    cache_parser = subparser.add_parser("cache", help="Cache related commands.", formatter_class=LitFormatter)
    cache_parser.description = (
        "Cache related commands for managing the default cache used by StreamingDataset and other utilities."
    )

    # register the `list` & `clear` commands under the cache subparser

    # Clear cache command
    cache_subparser = cache_parser.add_subparsers(
        dest="cache_command",
        title="Cache Commands",
    )
    clear_parser = cache_subparser.add_parser(
        "clear",
        help="Clear the default cache used for StreamingDataset and other utilities.",
        description="Clear the default cache used for StreamingDataset and other utilities.",
        formatter_class=LitFormatter,
    )
    clear_parser.set_defaults(func=clear_cache)

    # Show cache path command
    path_parser = cache_subparser.add_parser(
        "path",
        help="Show the path to the default cache directory.",
        description="Show the path to the default cache directory.",
        formatter_class=LitFormatter,
    )
    path_parser.set_defaults(func=show_cache_path)


def register_optimize_subcommand(subparser: _SubParsersAction) -> None:
    """Add the optimize subcommand to the main parser."""
    optimize_parser = subparser.add_parser("optimize", help="Optimize related commands.", formatter_class=LitFormatter)
    optimize_parser.description = "Optimize your dataset for faster streaming and AI model training."

    optimize_parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset to optimize.",
    )
    optimize_parser.set_defaults(func=optimize_dataset)


# List containing references to all functions that register subcommands
COMMAND_REGISTRY = [
    register_cache_subcommand,
    register_optimize_subcommand,
]
