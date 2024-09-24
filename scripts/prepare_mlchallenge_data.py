import os
import shutil
import argparse
import json
from pathlib import Path

"""
A script to prepare the data for the MLChallenge.

This script copies `latest.log` files recursively from a given directory to a new directory, and renames them with an incremental number.
It also creates a dictionary where each key is the original log file path and the value is the new log file name.
REMARK: This script is maintained for completeness. It is not needed for the MLChallenge task.
"""


def prepare_mlchallenge_data(source_dir, destination_dir):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Initialize a dictionary to store the mapping of original log file paths to new log file names
    log_file_mapping = {}

    # Traverse the source directory recursively
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file == "latest.log":
                # Get the full path of the original log file
                original_file_path = os.path.join(root, file)

                # Generate the new log file name with an incremental number
                new_file_name = f"log_{len(log_file_mapping) + 1}.log"

                # Copy the original log file to the destination directory with the new name
                new_file_path = os.path.join(destination_dir, new_file_name)
                shutil.copy2(original_file_path, new_file_path)

                # Store the mapping of original log file path to new log file name
                log_file_mapping[
                    str(Path(original_file_path).relative_to(Path(source_dir).parent))
                ] = new_file_name

    return log_file_mapping


# Create an argument parser
parser = argparse.ArgumentParser(
    description="Script to prepare the data for the MLChallenge."
)

# Add arguments
parser.add_argument(
    "--source_directory",
    type=str,
    help="Path to the source directory",
    default="/Users/Ash.AlDujaili/Downloads/triage_data/NIGHTLY_regr_for_central_ai/",
)
parser.add_argument(
    "--destination_directory",
    type=str,
    default="/Users/Ash.AlDujaili/Downloads/mlchallenge_data/",
)

# Parse the command-line arguments
args = parser.parse_args()

# Example usage
log_mapping = prepare_mlchallenge_data(
    args.source_directory, args.destination_directory
)
# Write log_mapping to a JSON file
with open("log_mapping.json", "w") as json_file:
    json.dump(log_mapping, json_file)
