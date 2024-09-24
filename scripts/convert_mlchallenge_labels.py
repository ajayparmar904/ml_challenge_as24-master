"""
This script reads MLChallenge submissions and convert `fname` to the actual log file name.
REMARK: This script is maintained for completeness. It is not needed for the MLChallenge task or to be used by participants.
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from ml_challenge.utils import repo_path


def main(args):
    """
    Renames fname column in the input CSV file to the actual log file name and saves the output to a new CSV file.
    and changes the label column to cluster column. This format is the one desired by the DV engineers.

    Args:
        args (_type_): _description_
    """
    with open(args.mapping_path, "r") as file:
        fname_mapping = json.load(file)

    inverse_mapping = {v: k for k, v in fname_mapping.items()}
    input_df = pd.read_csv(args.input_path)
    output_df = pd.DataFrame()
    output_df["fname"] = input_df["fname"].apply(lambda x: inverse_mapping[x])
    output_df["cluster"] = input_df["label"]
    output_df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert MLChallenge submissions to format readable by DV engineers."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to the input file (CSV)",
        default=str(
            Path(
                repo_path / "data" / "mlchallenge_submissions" / "random_submission.csv"
            )
        ),
    )
    parser.add_argument(
        "--mapping_path",
        type=str,
        help="Path to the input file (JSON)",
        default=str(Path(repo_path / "data" / "log_metadata" / "log_mapping.json")),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the output file (CSV)",
        default=str(
            Path(
                repo_path / "data" / "log_metadata" / "random_submission_converted.csv"
            )
        ),
    )
    args = parser.parse_args()

    main(args)
