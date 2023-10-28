spec = {
    "algorithm": "sarsa",
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
    "epsilon": 0.1,
    "training_frequency": 10,
    "memory": "on_policy",
}
