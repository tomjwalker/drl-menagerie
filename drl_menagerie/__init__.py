from .utils import *

from .agent.algorithm.reinforce import Reinforce


algorithm_map = {
    "reinforce": Reinforce,
}
