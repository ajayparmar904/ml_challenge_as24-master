import os
import argparse
import pandas as pd
from ml_challenge.utils import evaluate_submission_file, repo_path
from pathlib import Path

""" 
This script is used to evaluate submissions for the MLChallenge.

The script takes the following arguments:
- submission_dir: The directory containing the submissions.
- labels_path: The file containing the ground truth labels.
- output_path: The file to write the evaluation results to.
"""


def main(args):
    submission_results = []
    for submission_fname in os.listdir(args.submission_dir):
        if submission_fname.endswith(".csv"):
            print(f"Evaluating submission: {submission_fname}")
            res = evaluate_submission_file(
                os.path.join(args.submission_dir, submission_fname),
                args.groundtruth_path,
            )
            submission_results.append(
                dict(submission=Path(submission_fname).stem, **res)
            )

    pd.DataFrame(submission_results).to_csv(args.output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MLChallenge submissions")
    parser.add_argument(
        "--submission_dir",
        type=str,
        help="Directory containing the submissions",
        default=str(repo_path / "data" / "mlchallenge_submissions"),
    )
    parser.add_argument(
        "--groundtruth_path",
        type=str,
        help="path to the CSV file containing the groundtruth labels",
        default=str(repo_path / "data" / "log_metadata" / "mlchallenge_labels.csv"),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="file path to write the evaluation results to",
        default=str(
            repo_path / "data" / "log_metadata" / "mlchallenge_evaluation_results.csv"
        ),
    )

    args = parser.parse_args()
    main(args)
