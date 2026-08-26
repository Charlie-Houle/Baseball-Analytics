"""
WIP: runs the full "+" pipeline end-to-end, adding each metric's columns to
the raw Statcast data in order (Stuff+ -> Location+ -> Pitching+ ->
bestPitch+; see docs/purpose.md). Only Stuff+ is implemented so far.
"""

from pathlib import Path

import pandas as pd

from location import add_location_plus
from stuff import add_stuff_plus

DEFAULT_INPUT = Path(__file__).resolve().parent.parent.parent / "data" / "MLB_2021-2025.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent.parent / "data" / "MLB_2021-2025_plus.csv"


def run_pipeline(input_path=DEFAULT_INPUT, output_path=DEFAULT_OUTPUT):
    df = pd.read_csv(input_path)

    df = add_stuff_plus(df)
    df = add_location_plus(df)

    # TODO: Pitching+ (Stuff+ weighted by Location+)
    # TODO: bestPitch+ (Pitching+ vs. the optimal pitch for that arsenal/situation)

    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the full pitching '+' pipeline.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    run_pipeline(args.input, args.output)
