from pathlib import Path

import pandas as pd


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
