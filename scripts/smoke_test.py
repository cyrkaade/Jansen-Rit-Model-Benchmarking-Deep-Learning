"""Run the fast deterministic simulator/noise/Transformer smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

from deepjr.baseline import write_json
from deepjr.smoke import run_smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output", type=Path, default=Path("results/smoke/status.json")
    )
    arguments = parser.parse_args()
    result = run_smoke(arguments.seed)
    write_json(arguments.output, result)
    print(f"smoke test passed: {result}")


if __name__ == "__main__":
    main()
