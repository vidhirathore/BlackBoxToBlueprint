# src/legacy_systems/definitions.py
import numpy as np

"""
Define simple dummy legacy functions.
Each takes a numpy array as input and returns a predictable output
based on some internal logic (decision boundaries).
"""

def legacy_system_1_threshold(input_data: np.ndarray) -> str:
    """
    Simple threshold logic.
    Input: 1D numpy array [x]
    Output: 'Category A' or 'Category B'
    """
    if not isinstance(input_data, np.ndarray) or input_data.shape != (1,):
        raise ValueError("Input must be a 1D numpy array with shape (1,)")
    
    threshold = 5.0
    if input_data[0] > threshold:
        return "Category B"
    else:
        return "Category A"

def legacy_system_2_combined_conditions(input_data: np.ndarray) -> str:
    """
    Logic based on two combined conditions.
    Input: 2D numpy array [x, y]
    Output: 'High', 'Medium', 'Low'
    """
    if not isinstance(input_data, np.ndarray) or input_data.shape != (2,):
         raise ValueError("Input must be a 2D numpy array with shape (2,)")

    x, y = input_data
    if x > 0 and y > 0:
        return "High"
    elif x < 0 and y < 0:
        return "Low"
    else:
        return "Medium"

def legacy_system_3_nonlinear_ranges(input_data: np.ndarray) -> int:
    """
    Non-linear logic based on input ranges.
    Input: 1D numpy array [x]
    Output: Integer score (0, 10, 20)
    """
    if not isinstance(input_data, np.ndarray) or input_data.shape != (1,):
         raise ValueError("Input must be a 1D numpy array with shape (1,)")

    x = input_data[0]
    if -2 < x < 2:
        return 10 # Score 10 for central range
    elif x >= 2 or x <= -2:
        return 20 # Score 20 for outer ranges
    else: # Should not happen if logic covers all, but good practice
        return 0 # Default score


# Dictionary to easily access systems by name
LEGACY_SYSTEMS = {
    "system_1_threshold": {
        "function": legacy_system_1_threshold,
        "input_dim": 1,
        "input_bounds": (-10.0, 10.0) # Min, Max for input values
    },
    "system_2_combined": {
        "function": legacy_system_2_combined_conditions,
        "input_dim": 2,
        "input_bounds": (-5.0, 5.0)
    },
    "system_3_nonlinear": {
        "function": legacy_system_3_nonlinear_ranges,
        "input_dim": 1,
        "input_bounds": (-5.0, 5.0)
    }
}

# Simple wrapper class (as described in Step 1)
class LegacySystemWrapper:
    def __init__(self, system_name: str):
        if system_name not in LEGACY_SYSTEMS:
            raise ValueError(f"Unknown legacy system: {system_name}")
        self.system_info = LEGACY_SYSTEMS[system_name]
        self.system_func = self.system_info["function"]
        print(f"Initialized wrapper for: {system_name}")

    def run(self, input_data: np.ndarray):
        """Sends input x and receives output y = f(x)"""
        # Add any necessary input validation or transformation here if needed
        return self.system_func(input_data)

    def get_input_dim(self) -> int:
        return self.system_info["input_dim"]

    def get_input_bounds(self) -> tuple[float, float]:
        return self.system_info["input_bounds"]