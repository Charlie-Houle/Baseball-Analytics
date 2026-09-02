"""
Runs the full "+" pipeline end-to-end, adding each metric's columns to the
raw Statcast data in order (Stuff+ -> Location+ -> Pitching+ -> bestPitch+;
see docs/purpose.md).
"""

from pathlib import Path

import pandas as pd

try:
    from .bestpitch import add_bestpitch_plus
    from .location import DEFAULT_MODELS_DIR, add_location_plus
    from .pitching import add_pitching_plus
    from .stuff import add_stuff_plus
except ImportError:
    from bestpitch import add_bestpitch_plus
    from location import DEFAULT_MODELS_DIR, add_location_plus
    from pitching import add_pitching_plus
    from stuff import add_stuff_plus

DEFAULT_INPUT = Path(__file__).resolve().parent.parent.parent / "data" / "MLB_2021-2025.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent.parent / "data" / "MLB_2021-2025_plus.csv"


def run_pipeline(input_path=DEFAULT_INPUT, output_path=DEFAULT_OUTPUT, models_dir=DEFAULT_MODELS_DIR, retrain=False):
    df = pd.read_csv(input_path)

    df = add_stuff_plus(df)
    df = add_location_plus(df, models_dir=models_dir, retrain=retrain)
    df = add_pitching_plus(df, models_dir=models_dir, retrain=retrain)
    df = add_bestpitch_plus(df, models_dir=models_dir, retrain=retrain)

    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the full pitching '+' pipeline.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--retrain", action="store_true", help="Retrain Location+ models instead of using the cache")
    args = parser.parse_args()

    run_pipeline(args.input, args.output, models_dir=args.models_dir, retrain=args.retrain)
