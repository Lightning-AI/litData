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

import typer

from litdata.utilities.dataset_utilities import get_default_cache_dir

app = typer.Typer(
    help="""
⚡ LitData CLI – Transform, Optimize & Stream data for AI at scale.

LitData simplifies and accelerates data workflows for machine learning.
Easily scale data processing tasks—like scraping, resizing, inference, or embedding creation
across local or cloud environments.

Optimize datasets to boost model training speed and handle large remote datasets efficiently,
without full local downloads.
"""
)

cache_app = typer.Typer()
app.add_typer(cache_app, name="cache")


@cache_app.callback()
def cache(ctx: typer.Context) -> None:
    """Subcommand group for cache-related operations."""


@cache_app.command("clear")
def clear_cache() -> None:
    """Clear default cache used for StreamingDataset and other utilities."""
    streaming_default_cache_dir = get_default_cache_dir()

    shutil.rmtree(streaming_default_cache_dir, ignore_errors=True)

    typer.echo(f"Cache directory '{streaming_default_cache_dir}' cleared.")


@cache_app.command("path")
def show_cache_path() -> None:
    """Show the path to the cache directory."""
    streaming_default_cache_dir = get_default_cache_dir()
    typer.echo(f"Default cache directory: {streaming_default_cache_dir}")


if __name__ == "__main__":
    app()
