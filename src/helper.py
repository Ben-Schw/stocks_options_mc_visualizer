import numpy as np


"""
Computes summary statistics (paths and returns) across simulated price paths.

Parameters:
S0 (float) - Initial price used to convert paths into returns
paths (np.ndarray) - Simulated price paths (time x simulations)

Return:
dict[str, np.ndarray] - Dictionary of summary arrays (per time step)
"""
def price_paths_macros(S0: float, paths: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "mean_path": np.mean(paths, axis=1),
            "median_path": np.median(paths, axis=1),
            "std_dev_path": np.std(paths, axis=1),
            "mean_return": np.mean(paths, axis=1) / S0 - 1,
            "median_return": np.median(paths, axis=1) / S0 - 1,
            "std_dev_return": np.std(paths, axis=1) / S0,
            "25th_percentile": np.percentile(paths, 25, axis=1),
            "75th_percentile": np.percentile(paths, 75, axis=1),
            "5th_percentile": np.percentile(paths, 5, axis=1),
            "95th_percentile": np.percentile(paths, 95, axis=1)
        }
