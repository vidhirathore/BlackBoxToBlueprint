# Black Box to Blueprint: Extracting Legacy System Logic

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Replace with your actual license -->
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)]() <!-- Adjust Python version if needed -->

This project implements an automated pipeline to extract interpretable decision-making logic from legacy software systems, treating them as black boxes. It leverages Reinforcement Learning (RL) for targeted exploration, identifies critical decision boundaries via counterfactual analysis, clusters boundary-crossing interactions, and extracts human-readable rules using decision trees. The ultimate goal is to generate a "blueprint" of the legacy system's core logic, significantly aiding understanding, modernization, and migration efforts.

## Table of Contents

- [Core Idea](#core-idea)
- [Pipeline Overview](#pipeline-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup Steps](#setup-steps)
- [Usage](#usage)
  - [Running Experiments](#running-experiments)
  - [Output Files](#output-files)
- [Interpreting Results](#interpreting-results)
  - [Rules (`_rules.txt`)](#rules-_rulestxt)
  - [Cluster Plot (`_clusters.png`)](#cluster-plot-_clusterspng)
  - [Trajectories (`_trajectories.csv`)](#trajectories-_trajectoriescsv)
- [Example Findings (Dummy Systems)](#example-findings-dummy-systems)
- [Applications in Legacy Migration](#applications-in-legacy-migration)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Core Idea

Migrating or modernizing legacy systems is often plagued by incomplete documentation and a poor grasp of the embedded business rules. Naive behavioral cloning (simply replicating input-output behavior) is often insufficient because it fails to capture the underlying *decision logic* or *intent*.

This project tackles this challenge by:

1.  **Interacting Programmatically:** Wrapping the legacy system (even if it's only accessible via CLI, API, or requires UI automation) to enable automated interaction.
2.  **Intelligent Exploration (RL):** Training a Reinforcement Learning agent to specifically find inputs that cause the system's output to *change*. This focuses exploration efforts on the critical decision boundaries.
3.  **Counterfactual Analysis:** Collecting data points (state, action, next state, previous output, current output) precisely where the system's output changed ($y_{prev} \neq y_{curr}$). These represent "counterfactuals" near decision boundaries.
4.  **Pattern Discovery (Clustering):** Grouping the input states ($s$) that led to these output changes using clustering algorithms (K-Means) to identify recurring patterns associated with decisions.
5.  **Rule Extraction (Decision Trees):** Training a decision tree on the clustered states to translate these patterns into interpretable IF-THEN rules, effectively approximating the system's logic near the identified boundaries.

## Pipeline Overview

The implemented pipeline consists of the following automated steps:

1.  **Black Box Wrapping:** Encapsulate the target legacy system within a Python wrapper interface (`src/legacy_systems/definitions.py`).
2.  **RL Agent Training:** Train an RL agent (e.g., PPO from Stable-Baselines3) within a custom Gymnasium environment (`src/rl_environment/env.py`). The environment rewards the agent for finding inputs that cause the wrapped system's output to change.
3.  **Counterfactual Collection:** Deploy the trained RL agent to interact with the environment and log the transitions where the output changes (`src/analysis/extract.py`).
4.  **Clustering:** Apply K-Means clustering to the input states (`s`) gathered during the counterfactual collection phase (`src/analysis/extract.py`).
5.  **Visualization (Optional):** Plot the clustered states and their centroids if the input system has 2 dimensions (`src/utils/helpers.py`).
6.  **Rule Extraction:** Train a Decision Tree classifier on the input states, using the cluster labels as targets, and extract the learned rules in a textual format (`src/analysis/extract.py`).
7.  **Orchestration:** A main script (`src/run_experiment.py`) manages the configuration and execution of the training, collection, and analysis steps.

## Installation

### Prerequisites

*   Python 3 (tested with 3.10, likely compatible with 3.8+)
*   `pip` (Python package installer)
*   `venv` (Recommended for managing virtual environments)

### Setup Steps

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url> # Replace with your repository URL
    cd BlackBoxToBlueprint
    ```

2.  **Create and activate a virtual environment** (Recommended):

    *   **Linux/macOS:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   **Windows (Command Prompt):**
        ```bash
        python -m venv venv
        venv\Scripts\activate.bat
        ```
    *   **Windows (PowerShell):**
        ```bash
        python -m venv venv
        venv\Scripts\Activate.ps1
        # If script execution is disabled, you might need to run:
        # Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **(Optional) Install TensorBoard for monitoring RL training:**
    ```bash
    pip install tensorboard
    ```
    To view logs during or after training (run from the `BlackBoxToBlueprint` root directory):
    ```bash
    tensorboard --logdir src/tensorboard_logs/
    ```
    *(Note: If you skip this, ensure `tensorboard_log=None` is set or handled appropriately in `src/run_experiment.py`)*

## Usage

**Important:** Due to the current import structure (using relative imports like `from legacy_systems...`), the main script **must be run from within the `src` directory.**

1.  **Navigate into the `src` directory:**
    ```bash
    cd src
    ```

### Running Experiments

Run the main experiment script, specifying the target system defined in `legacy_systems/definitions.py`. The script will automatically train the RL agent if a saved model doesn't exist (unless forced), collect counterfactuals, and perform the analysis.

```bash
# Analyze system_1_threshold (trains RL model if not found)
python run_experiment.py --system system_1_threshold

# Analyze system_2_combined (2D system, will generate a cluster plot)
python run_experiment.py --system system_2_combined

# Analyze system_3_nonlinear
python run_experiment.py --system system_3_nonlinear

# Force retraining of the RL agent for system_1, even if a model exists
python run_experiment.py --system system_1_threshold --force_retrain

# Force recollection of trajectories for system_2, even if data exists
python run_experiment.py --system system_2_combined --force_recollect

# Run with custom parameters (overriding defaults in run_experiment.py)
python run_experiment.py --system system_3_nonlinear --train_steps 30000 --collect_episodes 150 --n_clusters 5
```
### Output Files

The pipeline generates the following outputs:

*   **Console:** Prints progress information during RL training, data collection, and analysis phases.
*   **`models/`:** Saves the trained PPO RL agent as a `.zip` file for each system (e.g., `system_1_threshold_ppo_agent.zip`).
*   **`data/`:** Saves the collected counterfactual trajectories (state, action, next_state, outputs) as a `.csv` file for each system (e.g., `system_1_threshold_trajectories.csv`).
*   **`results/`:** Stores the analysis outputs:
    *   `*_rules.txt`: Contains the interpretable IF-THEN rules extracted by the Decision Tree, approximating the logic at the boundaries.
    *   `*_clusters.png`: A scatter plot visualizing the clustered states and centroids (generated *only* for systems with 2D input spaces).
    *   `*_clusters_plot_skipped.txt`: An empty file indicating that plotting was skipped because the input space was not 2D.
*   **`src/tensorboard_logs/`:** Contains logs for monitoring RL training progress (if TensorBoard is enabled).

## Interpreting Results

### Rules (`_rules.txt`)

These files contain the core output: human-readable rules approximating the legacy system's decision logic *at the boundaries* where its output changes. The rules show conditions (e.g., `input_0 <= 4.98`) learned by the decision tree to separate the different clusters of boundary-crossing points.

*   **Focus on the threshold values:** These values (e.g., `4.98` in the example) indicate the approximate locations of the decision boundaries discovered by the pipeline.
*   **Interpret the structure:** The tree structure shows how different input features combine to define decision regions near the boundaries.

### Cluster Plot (`_clusters.png`)

For 2D input systems, this plot provides a visual confirmation of where the pipeline found decision boundaries.

*   **Points:** Each point represents an input state (`s`) where the legacy system's output changed during exploration.
*   **Colors:** Different colors group points into clusters found by K-Means, suggesting distinct boundary regions or patterns.
*   **Red 'X' Marks:** Indicate the calculated centroids of each cluster.
*   **Patterns:** Look for points clustering along specific axes, thresholds, or other geometric shapes, which can visually suggest the type of decision logic (e.g., thresholds, linear separators, quadrants).

### Trajectories (`_trajectories.csv`)

This raw data file contains the detailed records of each collected counterfactual transition. It includes:

*   `state`: The input state(s) just *before* the output changed.
*   `action`: The perturbation applied by the RL agent.
*   `next_state`: The input state(s) *after* the output changed.
*   `prev_output`, `current_output`: The system outputs before and after the change.

This data is useful for detailed debugging, alternative analysis methods, or gaining deeper insight into specific boundary crossings.

## Example Findings (Dummy Systems)

The included dummy systems demonstrate the pipeline's ability:

*   **System 1 (Simple Threshold):** The RL agent effectively identifies the boundary region near `input_0 = 5.0`. The extracted rules exhibit splits very close to this value (e.g., `input_0 <= 4.96`), confirming the simple threshold logic.
*   **System 2 (Combined Conditions):** The agent focuses exploration near the `input_0 = 0` and `input_1 = 0` axes. The cluster plot visually confirms this, showing distinct groups near these axes. The extracted rules use thresholds near zero for both inputs, approximating the system's quadrant-based logic (e.g., `input_0 <= 0.02`, `input_1 > -0.05`).
*   **System 3 (Nonlinear Ranges):** The agent successfully identifies the two relevant boundaries near `input_0 = -2.0` and `input_0 = 2.0`. The extracted rules clearly show splits near these values (e.g., `input_0 <= -1.98`, `input_0 > 2.03`), correctly capturing the range-based decision logic.
   
