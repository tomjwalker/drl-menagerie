# from tac.experiment.control import RUN_MODES


SPEC_TEMPLATE = {
    "required_fields": {
        "name": str,
        "algorithm_spec.name": str,
        "algorithm_spec.gamma": float,
        "algorithm_spec.training_frequency": int,
        "memory_spec.name": str,
        "net_spec.type": str,
        "net_spec.hidden_layer_units": list,
        "net_spec.hidden_layer_activation": str,
        "net_spec.loss_spec.name": str,
        "net_spec.optim_spec.name": str,
        "net_spec.optim_spec.learning_rate": float,
        "environment_spec.name": str,
        "meta_spec.data_directory": str,
        "meta_spec.max_frame": int
    },
    "optional_fields": {
        "algorithm_spec.action_pd_type": str,
        "algorithm_spec.action_policy": str,
        "algorithm_spec.explore_var_spec.epsilon": float,
        "meta_spec.num_sessions": int,
        "meta_spec.num_trials": int,
        "meta_spec.random_seed": int,
    }
}


def flatten_dict(d, parent_key='', sep='.'):
    """
    Given a nested dictionary, return a flattened dictionary with keys containing hierarchy.

    Example
    -------
    d = {
        "a": 1,
        "b": {
            "c": 2,
            "d": 3,
        },
    }
    flatten_dict(d) = {
        "a": 1,
        "b.c": 2,
        "b.d": 3,
    }
    """
    items = {}
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep=sep))
        else:
            items[new_key] = value
    return items


def _validate_spec(spec, template, parent_key=""):
    """
    Validate the spec against the spec template. Check all required fields are present and of the correct type.
    Check all optional fields are of the correct type.
    """
    required_fields = template["required_fields"]
    optional_fields = template["optional_fields"]

    # Flatten the spec dictionary within this function, so that keys can be checked against the template
    spec = flatten_dict(spec)

    for field, field_type in required_fields.items():
        full_key = f"{parent_key}.{field}" if parent_key else field
        if field not in spec:
            raise ValueError(f"Spec is missing required field: {full_key}")
        if not isinstance(spec[field], field_type):
            raise TypeError(f"Spec field {full_key} has incorrect type: {type(spec[field])}, should be {field_type}")

    for field, field_type in optional_fields.items():
        full_key = f"{parent_key}.{field}" if parent_key else field
        if field in spec:
            if not isinstance(spec[field], field_type):
                raise TypeError(f"Spec field {full_key} has incorrect type: {type(spec[field])}, should be {field_type}")

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
    _validate_spec(spec, template=SPEC_TEMPLATE)
    return _get_run_type(spec)


# Demonstrate the utility functions on an example spec
if __name__ == "__main__":

    from tac.spec.sarsa.temp_spec import spec as example_sarsa_spec
    #
    # flattened_spec = flatten_dict(example_sarsa_spec)

    _validate_spec(example_sarsa_spec, template=SPEC_TEMPLATE)
    print("Spec validated successfully")

    run_type = _get_run_type(example_sarsa_spec)
    print(f"Run type: {run_type}")
