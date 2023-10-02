import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def set_filepath(file_path_string):
    """Creates a directory at the specified path if it does not already exist"""
    file_path = Path(file_path_string)
    file_path.mkdir(parents=True, exist_ok=True)
    return file_path


# TODO: come up with a better solution than this
def temp_initialise_log(spec_dict):
    """Temporary solution to generate a training log for plotting metrics. To be replaced with a more robust solution
    in the future"""

    num_training_episodes = spec_dict.get("training_episodes")
    metrics = [
        "loss",
        "total_reward",
        "solved",
    ]

    # Generate a pandas DataFrame. Column names are `metrics`. Number of rows is `num_training_episodes`. Cells are
    # initially empty
    training_log = pd.DataFrame(index=range(num_training_episodes), columns=metrics)

    return training_log


def set_random_seed():
    """Set random seeds for PyTorch and NumPy, for reproducibility"""
    seed = int(time.time())
    # TODO: random seed dependent on session and trial numbers too? (See SLM-Lab)
    # TODO: save seed to spec
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed
