# src/analysis/extract.py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_text
from pathlib import Path

from utils.helpers import plot_clusters, save_trajectories, load_trajectories

def collect_counterfactual_trajectories(env, agent, num_episodes: int, deterministic: bool = True) -> list:
    """Run the trained agent in the environment to collect interaction data."""
    trajectories = []
    print(f"Collecting trajectories for {num_episodes} episodes...")

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_steps = 0
        while not (done or truncated):
            action, _ = agent.predict(obs, deterministic=deterministic)
            next_obs, reward, done, truncated, info = env.step(action)

            # Record transitions where the output actually changed (counterfactuals)
            if info.get("output_changed", False):
                trajectory_point = {
                    "state": obs.copy(), # State *before* the change
                    "action": action.copy(),
                    "next_state": next_obs.copy(), # State *after* the change
                    "previous_output": info.get("previous_output"),
                    "current_output": info.get("current_output"),
                    "reward": reward,
                    "episode": episode,
                    "step": info.get("step")
                }
                trajectories.append(trajectory_point)

            obs = next_obs
            episode_steps += 1
        # print(f"Episode {episode+1} finished after {episode_steps} steps.") # Debug

    print(f"Collected {len(trajectories)} counterfactual transitions.")
    return trajectories


def cluster_transitions(trajectories_df: pd.DataFrame, n_clusters: int = 5, random_state: int = 42) -> tuple[pd.DataFrame, KMeans]:
    """
    Cluster the states where output changes occurred using KMeans.
    Returns the DataFrame with added cluster labels and the fitted KMeans model.
    """
    if trajectories_df.empty:
        print("No counterfactual transitions found to cluster.")
        return trajectories_df, None

    # Use the 'state' (input just before the change) as features for clustering
    # Convert list of arrays into a 2D numpy array
    states_for_clustering = np.array(trajectories_df['state'].tolist())

    if states_for_clustering.shape[0] < n_clusters:
        print(f"Warning: Number of data points ({states_for_clustering.shape[0]}) is less than n_clusters ({n_clusters}). Adjusting n_clusters.")
        n_clusters = max(1, states_for_clustering.shape[0]) # Ensure at least 1 cluster

    print(f"Clustering {states_for_clustering.shape[0]} states into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10) # n_init suppresses warning
    cluster_labels = kmeans.fit_predict(states_for_clustering)

    trajectories_df['cluster'] = cluster_labels
    print("Clustering complete.")
    return trajectories_df, kmeans


def extract_rules_from_clusters(clustered_df: pd.DataFrame, kmeans_model: KMeans, input_dim: int) -> list[str]:
    """
    Extracts interpretable rules from cluster centroids or using a Decision Tree.
    """
    if clustered_df.empty or kmeans_model is None:
        return ["No clusters or data available for rule extraction."]

    rules = []
    centroids = kmeans_model.cluster_centers_

    # --- Method 1: Simple Centroid Interpretation (less robust) ---
    # for i, centroid in enumerate(centroids):
    #     # You might query the legacy system near the centroid to understand behavior
    #     rule = f"Cluster {i} Centroid: {np.round(centroid, 2)}. Behavior near this point likely represents a decision boundary region."
    #     rules.append(rule)

    # --- Method 2: Decision Tree on Counterfactual States ---
    print("Extracting rules using Decision Tree on states causing output changes...")
    X = np.array(clustered_df['state'].tolist())
    # Target could be the cluster label, or the 'next_output', or simply 'output changed' (binary)
    # Let's use cluster label as target to describe the input regions
    y = clustered_df['cluster'].values

    if len(np.unique(y)) < 2:
         print("Only one cluster found. Decision tree requires multiple classes.")
         # Fallback to centroid description if only one cluster
         rules.append(f"Only one cluster found. Centroid at {np.round(centroids[0], 2)} represents the main region where output changes occur.")
         return rules


    # Simple decision tree for interpretability
    dt = DecisionTreeClassifier(max_depth=max(3, input_dim + 1), random_state=42) # Limit depth
    dt.fit(X, y)

    # Generate feature names dynamically
    feature_names = [f'input_{i}' for i in range(input_dim)]
    class_names = [f'Cluster_{i}' for i in range(kmeans_model.n_clusters)]

    try:
        tree_rules = export_text(dt, feature_names=feature_names, class_names=class_names)
        rules.append("Decision Tree Rules (describing input regions leading to different clusters of change):")
        rules.append(tree_rules)
    except Exception as e:
        rules.append(f"Could not generate text rules from Decision Tree: {e}")
        # Fallback to centroid interpretation if export_text fails
        for i, centroid in enumerate(centroids):
             rule = f"Cluster {i} Centroid: {np.round(centroid, 2)}. Behavior near this point likely represents a decision boundary region."
             rules.append(rule)

    print("Rule extraction complete.")
    return rules


def analyze_legacy_system(
    system_name: str,
    env,
    agent,
    results_dir: Path,
    data_dir: Path,
    num_episodes_collect: int = 50,
    n_clusters: int = 4,
    force_recollect: bool = False
):
    """
    Full analysis pipeline: collect data, cluster, extract rules, visualize.
    """
    print(f"\n--- Starting Analysis for: {system_name} ---")
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    trajectory_file = data_dir / f"{system_name}_trajectories.csv"
    cluster_plot_file = results_dir / f"{system_name}_clusters.png"

    # --- 1. Collect Counterfactual Trajectories ---
    if force_recollect or not trajectory_file.exists():
        trajectories = collect_counterfactual_trajectories(env, agent, num_episodes_collect)
        if not trajectories:
            print("No counterfactual trajectories collected. Skipping analysis.")
            return
        trajectories_df = pd.DataFrame(trajectories)
        # Need to handle numpy arrays for CSV saving/loading if not done in helpers
        # For simplicity here, we might lose array structure if not careful
        # Let's save/load directly within the function if not reloading
        save_trajectories(trajectories, trajectory_file) # Save raw list for now
    else:
        print(f"Loading existing trajectories from {trajectory_file}")
        # This load needs refinement if arrays were stored as strings
        trajectories_df = load_trajectories(trajectory_file)
         # Convert string representations back to numpy arrays after loading
        for col in ['state', 'action', 'next_state']:
             # Basic string parsing - assumes space separation and numeric values
             trajectories_df[col] = trajectories_df[col].apply(
                 lambda x: np.fromstring(str(x).strip('[]').replace('\n', ''), sep=' ', dtype=np.float32) if isinstance(x, str) else x
                 )


    if trajectories_df.empty:
        print("No counterfactual trajectories found in data. Skipping analysis.")
        return

    # Filter again just to be sure (might not be needed if collection is correct)
    counterfactual_df = trajectories_df[trajectories_df['reward'] > 0].copy() # Use reward as indicator
    if counterfactual_df.empty:
         print("No transitions with output changes found in loaded data.")
         return


    # --- 2. Cluster Transitions ---
    clustered_df, kmeans_model = cluster_transitions(counterfactual_df, n_clusters=n_clusters)

    # --- 3. Visualize Clusters ---
    if kmeans_model and env.legacy_wrapper.get_input_dim() == 2 : # Only plot if 2D input
        states_to_plot = np.array(clustered_df['state'].tolist())
        labels = clustered_df['cluster'].values
        centroids = kmeans_model.cluster_centers_
        plot_clusters(
            data=states_to_plot,
            labels=labels,
            centroids=centroids,
            title=f"Clusters of Input States Causing Output Changes ({system_name})",
            save_path=cluster_plot_file
        )
    elif kmeans_model:
        print(f"Skipping cluster plot for {env.legacy_wrapper.get_input_dim()}D input (only 2D supported).")
        # Mark plot as not generated
        cluster_plot_file = Path(results_dir / f"{system_name}_clusters_plot_skipped.txt")
        cluster_plot_file.touch()


    # --- 4. Extract Semantic Rules ---
    rules = extract_rules_from_clusters(clustered_df, kmeans_model, env.legacy_wrapper.get_input_dim())

    # --- 5. Save Results ---
    from utils.helpers import save_results # Local import to avoid circularity if helpers imports analysis
    save_results(rules, cluster_plot_file, system_name, results_dir)

    print(f"--- Analysis Complete for: {system_name} ---")