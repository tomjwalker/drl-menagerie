from .utils import *
import os

# TODO: remove this comment block once Reinforce has been refactored to fit the same template as SARSA
# from .agent.algorithm.reinforce import Reinforce
#
#
# algorithm_map = {
#     "reinforce": Reinforce,
# }

# Get the environment variable, or default to "development"
os.environ["PY_ENV"] = os.environ.get("PY_ENV") or "development"

# Set the root directory in a platform-independent way, which is reliable, regardless of the current working directory
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
