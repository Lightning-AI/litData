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

from argparse import ArgumentParser, Namespace

from litdata.cli import LitFormatter
from litdata.cli.commands import COMMAND_REGISTRY


def parse_args() -> tuple[Namespace, ArgumentParser]:
    parser = ArgumentParser(
        prog="litdata",
        usage="%(prog)s [command] [options]",
        epilog="For more information, visit https://github.com/lightning-ai/litdata/",
        description="""LitData CLI – Transform, Optimize & Stream data for AI at scale.

LitData simplifies and accelerates data workflows for machine learning.
Easily scale data processing tasks—like scraping, resizing, inference, or embedding creation
across local or cloud environments.

Optimize datasets to boost model training speed and handle large remote datasets efficiently,
without full local downloads.""",
        formatter_class=LitFormatter,
    )

    # create a subparsers object to handle different commands
    subparsers = parser.add_subparsers(
        dest="command",
        title="Commands",
    )

    # register all commands
    for cmd_fn in COMMAND_REGISTRY:
        cmd_fn(subparsers)

    return parser.parse_args(), parser
