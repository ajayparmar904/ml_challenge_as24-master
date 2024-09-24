"""
A script to prepare the data for the MLChallenge.

This script accepts a csv file and json file as input, and generates a new csv file with the labels in the required format for the MLChallenge.
REMARK: This script is maintained for completeness. It is not needed for the MLChallenge task.
"""

import json
import pandas as pd
from pathlib import Path
import argparse


def prepare_mlchallenge_labels(input_csv, input_json, output_csv):
    # Read the input CSV file
    df = pd.read_csv(input_csv)

    # Read the input JSON file
    with open(input_json, "r") as file:
        log_mapping = json.load(file)

    # Create a new DataFrame to store the labels in the required format
    new_df = pd.DataFrame(columns=["fname", "label"])

    # Iterate over the rows of the input DataFrame
    fnames = []
    labels = []
    for index, row in df.iterrows():
        # Get the log file name
        log_fname = log_mapping[str(Path(row.fname).parent.parent / "latest.log")]
        label = row.cluster
        # Get the label for the log file from the JSON file

        # Append the log file name and label to the new DataFrame
        fnames.append(log_fname)
        labels.append(label)

    new_df = pd.DataFrame({"fname": fnames, "label": labels})
    # Save the new DataFrame to a CSV file
    print(f"Saving the labels to {output_csv}")
    new_df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Prepare data for MLChallenge")
    # Add arguments for input CSV, input JSON, and output CSV
    parser.add_argument(
        "--input_csv",
        type=str,
        help="Path to the input CSV file with log file labels",
        default="/Users/Ash.AlDujaili/Downloads/mlchallenge_data/triage_prelim_clusters.csv",
    )
    parser.add_argument(
        "--input_json",
        type=str,
        help="Path to the input JSON file which contains mapping bw actual log file and mlchallenge log file names",
        default="/Users/Ash.AlDujaili/Downloads/mlchallenge_data/log_mapping.json",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        help="Path to the output CSV file that contains the labels info for mlchallenge log files.",
        default="/Users/Ash.AlDujaili/Downloads/mlchallenge_data/mlchallenge_labels.csv",
    )
    # Parse the command line arguments
    args = parser.parse_args()

    # Call the prepare_mlchallenge_labels function with the provided arguments
    prepare_mlchallenge_labels(args.input_csv, args.input_json, args.output_csv)
