"""
Runs the full "+" pipeline end-to-end, adding each metric's columns to the
raw Statcast data in order (Stuff+ -> Location+ -> Pitching+ -> bestPitch+;
see docs/purpose.md).

Only the raw input CSV is ever read: every stage's `add_*_plus` function
takes and returns a dataframe in memory, so a bare `run_pipeline()` call
(from a notebook, a REPL, another script) never touches disk beyond that one
read, unless you explicitly pass `output_path`. That's also true of each
stage's own module run standalone (`python stuff.py`, `python location.py`,
etc.) -- their CLI wrappers write their own output CSV, but nothing else in
the pipeline reads that file back in; every later stage is always computed
from the raw data via in-memory chaining. Writing one intermediate CSV per
stage is a debugging convenience the CLI offers, not something the pipeline
depends on -- for normal use, `run_pipeline(output_path=...)` alone is
sufficient, and if you don't need the CSV, omitting `output_path` skips the
write entirely (see docs/dev_log.md's 9/2 entry).
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


def run_pipeline(input_path=DEFAULT_INPUT, output_path=None, models_dir=DEFAULT_MODELS_DIR, retrain=False):
    """
    Returns the fully-scored dataframe. Writes it to `output_path` too, but
    only if one is given -- default is None (in-memory only, no write),
    so calling this from a notebook/REPL doesn't silently spend a couple
    minutes writing a multi-GB CSV nobody asked for. The CLI below always
    passes an explicit output path, matching its documented behavior.
    """

    df = pd.read_csv(input_path)

    df = add_stuff_plus(df)
    df = add_location_plus(df, models_dir=models_dir, retrain=retrain)
    df = add_pitching_plus(df, models_dir=models_dir, retrain=retrain)
    df = add_bestpitch_plus(df, models_dir=models_dir, retrain=retrain)

    if output_path is not None:
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
