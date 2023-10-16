from .spec_utils import analyse_spec

# TODO: This will need renaming and replacing when final spec solution is ready
from .reinforce.temp_spec import spec as reinforce_spec
from .sarsa.temp_spec import spec as sarsa_spec


ALGORITHM_SPECS = {
    "reinforce": reinforce_spec,
    "sarsa": sarsa_spec,
}


def load_spec(algorithm_name):
    return ALGORITHM_SPECS[algorithm_name]
