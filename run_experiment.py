# scripts/run_experiment.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from pathlib import Path
import argparse
import time

# Make sure src is in the Python path or adjust imports accordingly
# e.g., by running from the root `legacy-logic-extractor` directory:
# `python scripts/run_experiment.py --system system_1_threshold`
from rl_environment.env import LegacyExplorerEnv
from analysis.extract import analyze_legacy_system
from legacy_systems.definitions import LEGACY_SYSTEMS


def train_rl_agent(env_id: str, system_name: str, models_dir: Path, total_timesteps: int = 10000, force_retrain: bool = False) -> PPO:
    """Trains or loads an RL agent (PPO) for the given environment."""
    model_path = models_dir / f"{system_name}_ppo_agent.zip"
    models_dir.mkdir(parents=True, exist_ok=True)

    if not force_retrain and model_path.exists():
        print(f"Loading existing agent model from {model_path}")
        agent = PPO.load(model_path)
        # You might need to set the environment for the loaded model if you use it directly
        # agent.set_env(make_vec_env(lambda: LegacyExplorerEnv(system_name=system_name), n_envs=1)) # Example
        # However, for analysis, we often create a fresh env anyway.
    else:
        print(f"Training new agent for {system_name}...")
        # Create a vectorized environment for potentially faster training
        # Use a lambda to pass arguments to the environment constructor
        vec_env = make_vec_env(lambda: LegacyExplorerEnv(system_name=system_name), n_envs=4) # Use 4 parallel envs

        # Instantiate the agent
        agent = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./tensorboard_logs/")

        # Train the agent
        start_time = time.time()
        agent.learn(total_timesteps=total_timesteps, log_interval=10)
        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds.")

        # Save the agent
        agent.save(model_path)
        print(f"Agent model saved to {model_path}")

        # It's good practice to close the environment
        vec_env.close()

    return agent


def main():
    parser = argparse.ArgumentParser(description="Run Legacy Logic Extraction Pipeline")
    parser.add_argument(
        "--system",
        type=str,
        required=True,
        choices=LEGACY_SYSTEMS.keys(),
        help="Name of the legacy system to analyze."
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=20000, # Increased steps for better exploration
        help="Number of timesteps to train the RL agent."
    )
    parser.add_argument(
        "--collect_episodes",
        type=int,
        default=100, # More episodes for denser data
        help="Number of episodes to run for collecting counterfactual trajectories."
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=4,
        help="Number of clusters for K-Means."
    )
    parser.add_argument(
        "--force_retrain",
        action='store_true',
        help="Force retraining the RL agent even if a saved model exists."
    )
    parser.add_argument(
        "--force_recollect",
        action='store_true',
        help="Force recollection of trajectories even if saved data exists."
    )
    args = parser.parse_args()

    # Define project directories relative to this script's location
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent
    models_dir = root_dir / "models"
    data_dir = root_dir / "data"
    results_dir = root_dir / "results"

    print(f"Selected Legacy System: {args.system}")
    print(f"RL Training Steps: {args.train_steps}")
    print(f"Trajectory Collection Episodes: {args.collect_episodes}")
    print(f"Number of Clusters: {args.n_clusters}")

    # --- Step 1 & 2: Train/Load RL Agent ---
    # The environment is implicitly created via make_vec_env during training
    # We need a separate instance for analysis
    agent = train_rl_agent(
        env_id=f"LegacyExplorer-{args.system}-v0", # Gym env id format
        system_name=args.system,
        models_dir=models_dir,
        total_timesteps=args.train_steps,
        force_retrain=args.force_retrain
    )

    # --- Step 3-6: Analyze System ---
    # Create a single environment instance for analysis/collection
    analysis_env = LegacyExplorerEnv(system_name=args.system, max_steps=200) # Use more steps per ep for collection

    analyze_legacy_system(
        system_name=args.system,
        env=analysis_env,
        agent=agent,
        results_dir=results_dir,
        data_dir=data_dir,
        num_episodes_collect=args.collect_episodes,
        n_clusters=args.n_clusters,
        force_recollect=args.force_recollect
    )

    analysis_env.close()
    print("\nExperiment finished.")


if __name__ == "__main__":
    # Register the custom environment (optional but good practice if using gym.make directly)
    # gym.register(
    #     id=f"LegacyExplorer-{args.system}-v0", # This needs system name dynamically which is tricky here
    #     entry_point="src.rl_environment.env:LegacyExplorerEnv",
    #     # kwargs={'system_name': args.system} # Can't access args here easily
    # )
    # For now, we instantiate directly, so registration isn't strictly needed.

    main()