# from tac.experiment.control import RUN_MODES


SPEC_TEMPLATE = {
    "required_fields": {
        "name": str,    # TODO: this will make more sense when algorithm/env/memory specs are sub-dicts of the spec
        "algorithm": str,
        "gamma": float,
        "hidden_layer_units": list,
        "learning_rate": float,
        "environment": str,
        "training_episodes": int,
        "activation": str,
        "optimiser": str,
        "data_directory": str,
        "training_frequency": int,
        "memory": str,
        "max_frame": int,
    },
    "optional_fields": {
        "num_sessions": int,
        "num_trials": int,
        "training_record_episodes": list,
        "search": dict,
        "random_seed": int,
    }
}


def _validate_spec(spec):
    """
    Validate the spec against the spec template. Check all required fields are present and of the correct type.
    Check all optional fields are of the correct type.
    """

    # Check required fields
    for field, field_type in SPEC_TEMPLATE["required_fields"].items():
        if field not in spec:
            raise ValueError(f"Spec is missing required field: {field}")
        if not isinstance(spec[field], field_type):
            raise TypeError(f"Spec field {field} has incorrect type: {type(spec[field])}, should be {field_type}")

    # Check optional fields
    for field, field_type in SPEC_TEMPLATE["optional_fields"].items():
        if field in spec:
            if not isinstance(spec[field], field_type):
                raise TypeError(f"Spec field {field} has incorrect type: {type(spec[field])}, should be {field_type}")

    # Check search field. If present, check it is a dict and that it contains at least one key. Check also that
    # "num_trials" exists, and its value is 1 or greater.
    if "search" in spec:
        if not isinstance(spec["search"], dict):
            raise TypeError(f"Spec field search has incorrect type: {type(spec['search'])}, should be dict")
        if len(spec["search"]) == 0:
            raise ValueError("Spec field search is empty")
        if "num_trials" not in spec:
            raise ValueError(
                "Spec field 'search' is present, indicating an experiment, but 'num_trials' is not specified"
            )


def _get_run_type(spec):
    """
    Get the run type from the spec. The run type is either "session", "trial", or "experiment". Logic:
    - If a dictionary exists for the "search" field, then the run type is "experiment".
    - If no "search" field exists, but a "num_sessions" field exists whose value is greater than 1, then the run type
      is "trial".
    """
    if "search" in spec:
        return "experiment"
    elif "num_sessions" in spec and spec["num_sessions"] > 1:
        return "trial"
    else:
        return "session"


def analyse_spec(spec):
    """Validates spec, then returns run type"""
    _validate_spec(spec)
    return _get_run_type(spec)


# Demonstrate the utility functions on an example spec
if __name__ == "__main__":

    example_reinforce_spec = {
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
        "search": {
            "learning_rate__choice": [0.01, 0.001],
            "gamma__uniform": [0.5, 1.0],
        }
    }

    _validate_spec(example_reinforce_spec)
    print("Spec validated successfully")

    run_type = _get_run_type(example_reinforce_spec)
    print(f"Run type: {run_type}")
