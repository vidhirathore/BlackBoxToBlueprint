# src/utils/helpers.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def plot_clusters(data: np.ndarray, labels: np.ndarray, centroids: np.ndarray, title: str, save_path: Path):
    """Plots clustered data points and centroids."""
    if data.shape[1] != 2:
        print("Plotting only supported for 2D data.")
        # Could add PCA/UMAP for higher dimensions if needed
        return

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        if label == -1: # Noise points in DBSCAN etc.
            color = 'gray'
        cluster_points = data[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, label=f'Cluster {label}', alpha=0.6, s=50)

    # Plot centroids
    if centroids is not None and centroids.shape[1] == 2:
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='red', label='Centroids')

    plt.title(title)
    plt.xlabel("Input Dimension 1")
    plt.ylabel("Input Dimension 2")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Cluster plot saved to {save_path}")
    # plt.show() # Uncomment to display plot interactively


def save_results(rules: list, cluster_plot_path: Path, system_name: str, results_dir: Path):
    """Saves extracted rules to a text file."""
    rules_path = results_dir / f"{system_name}_rules.txt"
    with open(rules_path, 'w') as f:
        f.write(f"Extracted Decision Rules for: {system_name}\n")
        f.write("=" * 40 + "\n\n")
        if rules:
             for i, rule in enumerate(rules):
                 f.write(f"Rule Cluster {i}:\n{rule}\n\n")
        else:
             f.write("No significant decision rules could be extracted.\n")
        f.write("\n" + "=" * 40 + "\n")
        f.write(f"Cluster visualization saved to: {cluster_plot_path.name}\n")

    print(f"Extracted rules saved to {rules_path}")


def save_trajectories(trajectories: list, filepath: Path):
    """Saves collected trajectories to a CSV file."""
    df = pd.DataFrame(trajectories)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Trajectories saved to {filepath}")

def load_trajectories(filepath: Path) -> pd.DataFrame:
    """Loads trajectories from a CSV file."""
    if not filepath.exists():
        raise FileNotFoundError(f"Trajectory file not found: {filepath}")
    df = pd.read_csv(filepath)
    # Convert string representations of numpy arrays back to arrays if needed
    # This simple save/load assumes basic types or requires post-processing
    # For numpy arrays stored as strings:
    # df['state'] = df['state'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
    # df['next_state'] = df['next_state'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
    # df['action'] = df['action'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
    print(f"Trajectories loaded from {filepath}")
    return df