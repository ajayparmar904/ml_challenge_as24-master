import os
import numpy as np
import argparse
from fastembed import TextEmbedding
from pathlib import Path
import matplotlib.pyplot as plt
import sklearn
from pathlib import Path
import pandas as pd
import json

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from ml_challenge.utils import evaluate_submission_file, generate_submission_file
from ml_challenge.utils import get_log_paths, get_log_fnames, repo_path

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, pairwise_distances_argmin_min
"""
This script reads logs from a given directory and embeds the log lines using a pre-trained model. 
The embeddings are then stored to a numpy file.
For a log file of N lines, the output numpy file will have the shape (N, embedding_dim).
"""

intermediate_dir = Path(repo_path / "data" / "intermediate_embeddings")
os.makedirs(intermediate_dir, exist_ok=True)

def embed_lines(lines, model):
    print(f" Embedding {len(lines)} lines")
    # you can implement your chunking logic here
    # ==============================================
    chunking_size = 100
    chunking_overlap = 50
    embeddings = []
    for i in range(0, len(lines), chunking_size - chunking_overlap):
        chunk = "\n".join(lines[i : i + chunking_size])
        embeddings.append(list(model.embed([chunk]))[0])
    # ==============================================

    return np.array(embeddings)


def embed_logs(log_directory, output_directory, embedding_model_name):

    # Create the destination directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # Load pre-trained model
    model = TextEmbedding(model_name=embedding_model_name)

    # Read log files from directory
    for filename in os.listdir(log_directory):
        if filename.endswith(".log"):
            log_fname = os.path.join(log_directory, filename)
            print(f"-Reading {log_fname}")
            with open(log_fname, "r") as file:
                lines = [
                    line.strip() for line in file.readlines() if len(line.strip()) > 0
                ]
                embeddings = embed_lines(lines, model)

            # Save embeddings to numpy file
            embed_fname = os.path.join(
                output_directory, str(Path(filename).stem) + ".npy"
            )
            print(f"-Saving embeddings to {output_directory}")
            np.save(embed_fname, embeddings)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Embed log lines using a pre-trained model"
    )
    parser.add_argument(
        "--log_directory", type=str, help="Path to the directory containing log files"
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        help="Path to the directory to store the output numpy files",
        default="log_embeddings",
    )
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        help="Name of the pre-trained embedding model",
        default="BAAI/bge-small-en-v1.5",
    )
    return parser.parse_args()

def compute_rv_coefficient(A, B):
    X = A.T.dot(A)
    Y = B.T.dot(B)
    # equation 2 in the paper https://arxiv.org/pdf/1307.7383
    correlation = np.trace(X @ Y) / (np.trace(X) * np.trace(Y))
    # for distance, we return 1 - correlation
    # some algorithms accepts an affinity matrix, so we can use the correlation directly
    return correlation

def standardize_data(data):
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)
    return standardized_data

# Load ground truth labels from CSV
def load_groundtruth(groundtruth_path):
    df = pd.read_csv(groundtruth_path)
    # Assume the CSV has columns 'filename' and 'label'
    return dict(zip(df['fname'], df['label']))

# Function to reduce dimensionality (Stage 1 PCA for individual files)
def reduce_noise_per_file(embedding_file, target_dim=100):
    data = np.load(embedding_file)
    data = np.transpose(data)

    # PCA to remove noise from each individual file
    pca = PCA(n_components=target_dim)
    reduced_data = pca.fit_transform(data)

    return reduced_data

# Function to flatten embeddings for global PCA and clustering
def flatten_embeddings(embeddings):
    flattened_embeddings = []
    for embedding in embeddings:
        flattened_embeddings.append(embedding.flatten())
    return flattened_embeddings

# Function to apply global PCA (Stage 2 PCA across all files)
def apply_global_pca(embeddings, target_dim=100):
    pca = PCA(n_components=target_dim)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings

# Function to reduce dimensionality (Stage 1 PCA for individual files) and save to disk
def save_reduced_embeddings_per_file(embedding_file, target_dim=100):
    data = np.load(embedding_file)
    data = np.transpose(data)

    # Apply PCA to reduce noise from each individual file
    #pca = IncrementalPCA(n_components=target_dim)
    pca = PCA(n_components=target_dim)
    reduced_data = pca.fit_transform(data)

    # Save reduced embeddings to disk
    reduced_filepath = os.path.join(intermediate_dir, embedding_file.stem + '_reduced.npy')
    np.save(reduced_filepath, reduced_data)

    return reduced_filepath

# Function to flatten embeddings for global PCA and clustering
#def flatten_embeddings(reduced_data):
#    return reduced_data.flatten()

# Function to apply global Incremental PCA (Stage 2)
def apply_global_incremental_pca(batch_size=50, target_dim=100):
    ipca = IncrementalPCA(n_components=target_dim)
    reduced_files = list(Path(intermediate_dir).glob('*.npy'))
    reduced_global_embeddings = []

    # Fit the incremental PCA in batches
    for i in range(0, len(reduced_files), batch_size):
        batch_data = [np.load(f) for f in reduced_files[i:i + batch_size]]
        flattened_batch = [flatten_embeddings(data) for data in batch_data]
        ipca.partial_fit(flattened_batch)
        transformed_batch = ipca.transform(flattened_batch)
        reduced_global_embeddings.extend(transformed_batch)

    return reduced_global_embeddings


# Function to cluster the embeddings
def cluster_embeddings(embeddings, num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # Get distances of each sample from its assigned cluster center
    distances = kmeans.transform(embeddings)  # Distance to each cluster center
    assigned_distances = np.min(distances, axis=1)  # Distance to the closest (assigned) cluster center

    # Convert distances into confidence scores (the smaller the distance, the higher the confidence)
    max_distance = np.max(assigned_distances)  # Max distance for normalization
    confidences = 1 - (assigned_distances / max_distance)  # Confidence: 1 means high confidence, 0 means low

    return labels, confidences


# Confidence-based filtering function
def filter_noisy_labels(fnames, labels, groundtruth, model_confidences, cluster_mapping, confidence_threshold=0.7):
    filtered_fnames = []
    filtered_labels = []

    for fname, label, confidence in zip(fnames, labels, model_confidences):
        groundtruth_label = groundtruth.get(fname, -1)

        if groundtruth_label == -1:
            # Keep the model prediction if the ground truth is unknown
            filtered_fnames.append(fname)
            filtered_labels.append(label)
        else:
            # Get the mapped label for the current model cluster
            #mapped_label = cluster_mapping.get(label, None)
            if label == groundtruth_label:
                filtered_fnames.append(fname)
                filtered_labels.append(label)  # Keep the model prediction
            else:
                # Update the model prediction if the confidence is above the threshold
                if confidence < confidence_threshold:
                    filtered_fnames.append(fname)
                    filtered_labels.append(groundtruth_label)  # Match the known label
                else:
                    filtered_fnames.append(fname)
                    filtered_labels.append(label)  # Keep the model prediction

    return filtered_fnames, filtered_labels

def create_cluster_mapping(labels, groundtruth, fnames, threshold=0.1):
    cluster_mapping = {}
    cluster_to_labels = defaultdict(list)

    # Build a mapping of model clusters to their corresponding known labels
    for fname, label in zip(fnames, labels):
        groundtruth_label = groundtruth.get(fname, -1)
        if groundtruth_label != -1:  # Only consider known labels
            cluster_to_labels[label].append(groundtruth_label)

    # Track assigned labels
    assigned_labels = set()

    # Determine the most common groundtruth label for each model cluster with frequency threshold
    for cluster, known_labels in cluster_to_labels.items():
        label_counts = pd.Series(known_labels).value_counts(normalize=True)  # Get relative frequencies

        # Sort labels by frequency and filter based on the threshold
        #sorted_labels = label_counts[label_counts >= threshold].index.tolist()
        sorted_labels = label_counts.index.tolist()

        for potential_label in sorted_labels:
            if potential_label not in assigned_labels:
                cluster_mapping[cluster] = potential_label
                assigned_labels.add(potential_label)  # Mark this label as assigned
                break  # Move to the next cluster once a label is assigned

        #most_common_label = label_counts.idxmax()  # Get the most common label
        #if label_counts.max() >= threshold:  # Check if the max frequency meets the threshold
        #cluster_mapping[cluster] = most_common_label

    return cluster_mapping

# Step 2: Function to Replace Labels Based on Mapping
def replace_labels_with_mapping(labels, cluster_mapping):
    # Apply the mapping to replace labels, keeping unchanged if no mapping found
    new_labels = np.array([
        cluster_mapping[cluster] if cluster in cluster_mapping and cluster_mapping[cluster] != -1 else cluster
        for cluster in labels
    ])
    return new_labels



def main():

    embeddings_dir = Path(repo_path / "data" / "log_embeddings")
    #submission_path = str(Path(repo_path / "data" / "mlchallenge_submissions" / "spectral_clustering_submission.csv"))
    submission_path = str(Path(repo_path / "data" / "mlchallenge_submissions" / "kmean_clustering_submission.csv"))
    groundtruth_path = str(Path(repo_path / "data" / "log_metadata" / "mlchallenge_labels.csv"))

    fnames = []

    reduced_embeddings_per_file = []

    # Step 1: Apply PCA to each file individually to remove noise
    for embedding_file in tqdm(embeddings_dir.iterdir()):
        reduced_data = reduce_noise_per_file(embedding_file, target_dim=100)
        reduced_embeddings_per_file.append(reduced_data)
        fnames.append(embedding_file.stem + '.log')

    # Step 2: Flatten the embeddings to prepare them for global PCA
    flattened_embeddings = flatten_embeddings(reduced_embeddings_per_file)

    # Step 3: Apply global PCA to find essential features across all files
    reduced_global_embeddings = apply_global_pca(flattened_embeddings, target_dim=100)

    # Step 4: Perform clustering on the globally reduced embeddings
    labels, model_confidences  = cluster_embeddings(reduced_global_embeddings, num_clusters=5)

    # Load groundtruth labels (noisy labels)
    groundtruth = load_groundtruth(groundtruth_path)

    # Create cluster mapping based on noisy labels
    cluster_mapping = create_cluster_mapping(labels, groundtruth, fnames)

    labels = replace_labels_with_mapping(labels, cluster_mapping)

    # Step 5: Confidence-based filtering of noisy labels
    filtered_fnames, filtered_labels = filter_noisy_labels(fnames, labels, groundtruth, model_confidences, cluster_mapping,confidence_threshold=0.9)

    # Step 6: Generate a new submission file with filtered data
    generate_submission_file(filtered_fnames, filtered_labels, submission_path)

    #generate_submission_file(fnames, labels, submission_path)

    # let's evaluate the submission
    sc_result = evaluate_submission_file(submission_path, groundtruth_path)
    print(sc_result)

if __name__ == "__main__":
    main()
