import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import pydash as ps


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


def to_torch_batch(batch, is_episodic, device=None):
    """
    Converts a batch of experiences from a dictionary of lists to a dictionary of PyTorch tensors.

    Arguments:
    ----------
    batch: dict
        Batch of experiences. Keys are the names of the different elements of an experience, values are lists of the
        experiences. Elements may be nested into episodes.
        e.g.
        batch = {
            "states": [s_1, s_2, ...],
            "actions": [a_1, a_2, ...],
            "rewards": [r_1, r_2, ...],
            "next_states": [s_1, s_2, ...],
            "dones": [d_1, d_2, ...],
        }
    device: torch.device
        Device to which the batch should be moved
    is_episodic: bool
        Whether the batch is episodic or not. If True, the batch is nested into episodes. If False, the batch is not
        nested into episodes.
    """

    if is_episodic:
        # Flatten each value of the dictionary into a single numpy array
        batch = {k: np.array([item for sublist in v for item in sublist]) for k, v in batch.items()}
        # Then convert each element of the batch to a PyTorch tensor, and move to the specified device
        batch = {k: torch.tensor(v, device=device) for k, v in batch.items()}

    else:
        # Convert each element of the batch to a PyTorch tensor, and move to the specified device
        batch = {k: torch.tensor(np.array(v), device=device) for k, v in batch.items()}
    return batch


def set_attr_from_dict(obj, attr_dict, keys=None):
    """
    Sets attributes of an object from a dictionary. If keys is specified, only sets attributes for the keys in 'keys'.
    """

    if keys is not None:
        # Pydash (ps) is a utility library for Python with many convenience methods.
        # This one concisely filters a dict by a list of keys.
        attr_dict = ps.pick(attr_dict, keys)
    for attr, val in attr_dict.items():
        setattr(obj, attr, val)
