import pandas as pd
from typing import List, Dict
from pathlib import Path
from sklearn.metrics import (
    homogeneity_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
)


repo_path = Path(__file__).parent.parent.resolve()


def get_log_paths() -> List[str]:
    """Returns a list of log files' paths. This is helpful for embedding the log files."""
    logs_dir = Path(repo_path / "data" / "log_files")
    # load file names
    log_files = [str(f) for f in logs_dir.glob("*.log")]
    return log_files


def get_log_fnames() -> List[str]:
    """Returns a list of log file names. This is helpful for generating submission files."""
    return [Path(f).name for f in get_log_paths()]


def generate_submission_file(fnames: List[str], labels: List[int], output_path: str):
    """Generates a submission file for the MLChallenge.

    Args:
        fnames (List[str]): names of the log files (e.g., ["log_1.log", "log_2.log"])
        labels (List[int]): labels of the log files listed in fnames (e.g., [0, 1])
        output_path (str): path to the output CSV file
    """
    # Create a DataFrame from the log file names and labels
    submission_df = pd.DataFrame({"fname": fnames, "label": labels})
    # Save the DataFrame to a CSV file
    submission_df.to_csv(output_path, index=False)
    print(f"Submission file saved to {output_path}")


def evaluate_submission_file(
    submission_path: str, groundtruth_path: str
) -> Dict[str, float]:
    """Evaluates a submission file for the MLChallenge.


    Args:
        submission_file (str): path to the submission CSV file
        groundtruth_path (str): path to the ground truth labels CSV file
    """
    # Read the submission and ground truth labels
    submission_df = pd.read_csv(submission_path)
    gt_df = pd.read_csv(groundtruth_path)

    # Ensure that the submission and ground truth labels have the same format
    assert (
        submission_df.shape[1] == 2
    ), "Submission file should have two columns: 'fname' and 'label'"
    assert (
        gt_df.shape[1] == 2
    ), "Labels file should have two columns: 'fname' and 'label'"

    # Merge the submission and ground truth labels on the 'fname' column
    merged_df = submission_df.merge(
        right=gt_df, on="fname", suffixes=("_pred", "_true")
    )
    # Remove rows with missing ground truth labels
    merged_df = merged_df[merged_df["label_true"] > -1]

    # Calculate the accuracy of the submission
    hm_score = homogeneity_score(merged_df["label_true"], merged_df["label_pred"])

    # mutual score [0,1] -> 1 is perfect match
    mi_score = normalized_mutual_info_score(
        merged_df["label_true"], merged_df["label_pred"]
    )
    # adjusted rand score [0,1] -> 1 is perfect match
    ar_score = adjusted_rand_score(merged_df["label_true"], merged_df["label_pred"])

    # Save the evaluation results to a dictionary
    result_dict = {
        "homogeneity": hm_score,
        "normalized_mutual_info": mi_score,
        "adjusted_rand_index": ar_score,
    }
    return result_dict
