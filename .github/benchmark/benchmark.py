"""This script runs LitData benchmarks in Lightning Studio for a given PR number."""

import argparse
from dataclasses import dataclass
from typing import Optional

from lightning_sdk import Machine, Studio

# Constants
DEFAULT_TEAMSPACE = "litdata"
DEFAULT_USER = None
DEFAULT_ORG = "lightning-ai"
DEFAULT_MACHINE = "A10G"


@dataclass
class BenchmarkArgs:
    """Arguments for the LitData benchmark."""

    pr_number: int
    org: Optional[str]
    user: Optional[str]
    teamspace: str
    machine: Machine


def parse_args() -> BenchmarkArgs:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark LitData in CI.")
    parser.add_argument("--pr", type=int, required=True, help="GitHub PR number")
    parser.add_argument("--user", default=DEFAULT_USER, type=str, help="Lightning Studio username")
    parser.add_argument("--org", default=DEFAULT_ORG, type=str, help="Lightning Studio org")
    parser.add_argument("--teamspace", default=DEFAULT_TEAMSPACE, type=str, help="Lightning Studio teamspace")
    parser.add_argument(
        "--machine", type=str, default=DEFAULT_MACHINE, choices=["A10G", "T4", "CPU"], help="Machine type"
    )

    args = parser.parse_args()

    machine_map = {
        "A10G": Machine.A10G,
        "T4": Machine.T4,
        "CPU": Machine.CPU,
    }

    return BenchmarkArgs(
        pr_number=args.pr,
        user=args.user,
        org=args.org,
        teamspace=args.teamspace,
        machine=machine_map[args.machine],
    )


class LitDataBenchmark:
    """Class to benchmark LitData in Lightning Studio."""

    def __init__(self, config: BenchmarkArgs):
        """Initialize the LitData benchmark with the given configuration."""
        if config.user is None and config.org is None:
            raise ValueError("Either user or org must be provided.")
        if config.user is not None and config.org is not None:
            raise ValueError("Only one of user or org must be provided.")

        self.pr = config.pr_number
        self.teamspace = config.teamspace
        self.user = config.user
        self.org = config.org
        self.machine = config.machine
        self.studio: Optional[Studio] = None

    def run(self) -> None:
        """Run the LitData benchmark."""
        assert self.pr is not None, "PR number is required"
        self.setup_studio()
        self.setup_litdata_pr()
        self.setup_and_run_litdata_benchmark()
        self.download_result_file()

    def setup_studio(self) -> None:
        """Set up the Lightning Studio."""
        assert self.studio is None, "Studio is already set up"

        self.studio = Studio(
            name=f"benchmark_litdata_pr_{self.pr}",
            teamspace=self.teamspace,
            user=self.user,
            org=self.org,
        )
        self.studio.start(self.machine)

    def setup_litdata_pr(self) -> None:
        """Set up the LitData PR in the studio."""
        assert self.studio is not None, "Studio is not set up"
        commands = [
            "rm -rf lit*",
            "git clone https://github.com/Lightning-AI/litData.git",
            "cd litData",
            f"gh pr checkout {self.pr}",
            "make setup",
        ]
        final_command = " && ".join(commands)
        print(f"Running command: {final_command}")
        output, output_code = self.studio.run_with_exit_code(final_command)
        if output_code != 0:
            raise RuntimeError(f"Command failed:\n{final_command}\nExit code {output_code}:\n{output}")

    def setup_and_run_litdata_benchmark(self) -> None:
        """Set up and run the LitData benchmark code and run the benchmarking."""
        assert self.studio is not None, "Studio is not set up"
        commands = [
            "git clone https://github.com/bhimrazy/litdata-benchmark.git",
            "cd litdata-benchmark",
            "make benchmark",
        ]
        final_command = " && ".join(commands)
        print(f"Running command: {final_command}")
        output, output_code = self.studio.run_with_exit_code(final_command)
        if output_code != 0:
            raise RuntimeError(f"Benchmark failed:\n{final_command}\nExit code {output_code}:\n{output}")

    def download_result_file(self) -> None:
        """Download the result file from the studio."""
        assert self.studio is not None, "Studio is not set up"
        filename = "litdata-benchmark/result.md"
        output_filename = "result.md"
        print(f"Downloading file: {filename} to {output_filename}")
        self.studio.download_file(filename, output_filename)


def main():
    """Main function to run the benchmark."""
    config = parse_args()
    print(f"Running LitData benchmark for PR #{config}")
    benchmark = LitDataBenchmark(config)
    benchmark.run()
    benchmark.studio.stop()
    print(f"✅ Benchmark completed for PR #{config.pr_number}")


if __name__ == "__main__":
    main()
