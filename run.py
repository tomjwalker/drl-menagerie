# import json

from tac.experiment.control import Session, Trial


# spec_path = "drl_menagerie/spec/reinforce/reinforce_cartpole.json"
# with open(spec_path, "r") as f:
#     spec = json.load(f)

# spec = {
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

spec = {
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
}

RUN_MODE = "trial"    # {"session", "trial", "experiment"}


if __name__ == "__main__":
    if RUN_MODE == "session":
        session = Session(spec)
        session.run()
    elif RUN_MODE == "trial":
        trial = Trial(spec)
        trial.run()
    elif RUN_MODE == "experiment":
        raise NotImplementedError("Experiment mode not yet implemented")
    else:
        raise ValueError(f"Invalid run mode: {RUN_MODE}")
