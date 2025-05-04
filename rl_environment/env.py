# src/rl_environment/env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Any

from legacy_systems.definitions import LegacySystemWrapper

class LegacyExplorerEnv(gym.Env):
    """
    A Gymnasium environment for exploring a legacy system black box.

    State: The current input vector to the legacy system.
    Action: A perturbation vector to be added to the current input state.
    Reward: High reward if the legacy system's output changes meaningfully
            after applying the perturbed input.
    """
    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(self, system_name: str, max_steps: int = 100, action_scale: float = 1.0):
        super().__init__()

        self.legacy_wrapper = LegacySystemWrapper(system_name)
        self.input_dim = self.legacy_wrapper.get_input_dim()
        self.min_bound, self.max_bound = self.legacy_wrapper.get_input_bounds()
        self.max_steps = max_steps
        self.action_scale = action_scale # Controls the magnitude of perturbations

        # Define observation space (the input to the legacy system)
        self.observation_space = spaces.Box(
            low=self.min_bound, high=self.max_bound, shape=(self.input_dim,), dtype=np.float32
        )

        # Define action space (the perturbation to apply)
        # Small perturbations are often desired
        self.action_space = spaces.Box(
            low=-self.action_scale, high=self.action_scale, shape=(self.input_dim,), dtype=np.float32
        )

        self.current_state = None
        self.current_output = None
        self.current_step = 0

        print(f"Environment configured for {system_name}:")
        print(f"  Input Dim: {self.input_dim}")
        print(f"  Input Bounds: ({self.min_bound}, {self.max_bound})")
        print(f"  Action Scale: {self.action_scale}")
        print(f"  Max Steps per Episode: {self.max_steps}")


    def _get_obs(self) -> np.ndarray:
        return self.current_state.astype(np.float32)

    def _get_info(self, output_changed: bool, prev_output: Any, next_output: Any) -> dict:
        return {
            "output_changed": output_changed,
            "previous_output": prev_output,
            "current_output": next_output,
            "step": self.current_step
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed) # Important for reproducibility

        # Initialize state with random input within bounds
        self.current_state = self.observation_space.sample()
        self.current_output = self.legacy_wrapper.run(self.current_state)
        self.current_step = 0

        # print(f"Env Reset: Initial state={self.current_state}, Initial output={self.current_output}") # Debug

        observation = self._get_obs()
        # Provide initial info, indicating no change yet
        info = self._get_info(False, self.current_output, self.current_output)
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Apply perturbation (action) to current state
        perturbation = action.astype(np.float32) # Ensure correct type
        next_state_raw = self.current_state + perturbation

        # Clamp the next state to be within the defined bounds
        next_state = np.clip(next_state_raw, self.min_bound, self.max_bound)

        # Get the output for the next state
        next_output = self.legacy_wrapper.run(next_state)

        # Calculate reward: +1 if output changed, 0 otherwise
        # Using simple equality check. Might need adjustment for numerical outputs.
        output_changed = (next_output != self.current_output)
        reward = 1.0 if output_changed else 0.0

        # Update internal state
        previous_output = self.current_output
        self.current_state = next_state
        self.current_output = next_output
        self.current_step += 1

        # Determine if episode is done
        terminated = False # No specific goal state, just exploration
        truncated = (self.current_step >= self.max_steps)
        done = terminated or truncated # Gymnasium uses terminated and truncated

        # Get observation and info
        observation = self._get_obs()
        info = self._get_info(output_changed, previous_output, self.current_output)

        # if output_changed: # Debug
        #     print(f"Step {self.current_step}: State={self.current_state}, Action={action}, Output changed: {previous_output} -> {next_output}, Reward={reward}")

        return observation, reward, terminated, truncated, info

    def close(self):
        # Clean up resources if needed (not necessary for these dummy systems)
        pass