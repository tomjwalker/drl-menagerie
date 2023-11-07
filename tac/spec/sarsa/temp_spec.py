spec = {
    "name": "sarsa",
    "algorithm_spec": {
        "name": "sarsa",
        "action_pd_type": "Argmax",
        "action_policy": "epsilon_greedy",
        "gamma": 0.99,
        "training_frequency": 10,
        "explore_var_spec": {
            "epsilon": 0.1,
        },
    },
    "memory_spec": {
        "name": "on_policy_batch",
    },
    "net_spec": {
        "type": "mlp",    # TODO: mapping for this is in tac/agent/net/__init__.py. Connect up to SLM Lab implementation
        "hidden_layer_units": [64],
        "hidden_layer_activation": "relu",
        # TODO: no clip_grad_val currently
        "loss_spec": {
            "name": "MSELoss",
        },
        "optim_spec": {
            "name": "adam",
            "learning_rate": 0.01,
        },
    },
    "environment_spec": {
        "name": "CartPole-v1",
    },
    # TODO: is this still required? "training_episodes": 20,
    # "training_record_episodes": [0, 100, 499],
    "meta_spec": {
        "data_directory": ".data",
        "num_sessions": 2,
        "num_trials": 2,
        "random_seed": 42,
        "max_frame": 1000,
    },
    # "search": {
    #     "learning_rate__choice": [0.01, 0.001],
    #     "gamma__uniform": [0.5, 1.0],
    # }
}
