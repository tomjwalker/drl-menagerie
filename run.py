"""
Main entry point for the application.
"""

# import json

from tac.experiment.control import RUN_MODES
from tac.spec.spec_utils import analyse_spec


# spec_path = "drl_menagerie/spec/reinforce/reinforce_cartpole.json"
# with open(spec_path, "r") as f:
#     spec = json.load(f)

# spec_dict = {
#     "algorithm": "reinforce",
#     "gamma": 0.99,
#     "hidden_layer_units": [64],
#     "learning_rate": 0.01,
#     "environment": "CartPole-v1",
#     "training_episodes": 500,
#     "activation": "relu",
#     "optimiser": "adam",
#     "training_record_episodes": [0, 100, 499],
#     "data_directory": ".data",
#     "num_sessions": 2,
# }

spec_dict = {
    "algorithm": "reinforce",
    "gamma": 0.99,
    "hidden_layer_units": [64],
    "learning_rate": 0.01,
    "environment": "CartPole-v1",
    "training_episodes": 20,
    "activation": "relu",
    "optimiser": "adam",
    # "training_record_episodes": [0, 100, 499],
    "data_directory": ".data",
    "num_sessions": 2,
    "num_trials": 2,
    # "search": {
    #     "learning_rate__choice": [0.01, 0.001],
    #     "gamma__uniform": [0.5, 1.0],
    # }
}


def run_spec(spec):
    # Determine the run mode from the spec
    run_mode_name = analyse_spec(spec)
    if run_mode_name in RUN_MODES:
        run_mode = RUN_MODES[run_mode_name]
    else:
        raise ValueError(f"Invalid run mode: {run_mode_name}")
    # Instantiate Session, Trial, or Experiment object
    run = run_mode(spec)
    # Run the session, trial, or experiment
    run.run()


if __name__ == "__main__":
    run_spec(spec_dict)
