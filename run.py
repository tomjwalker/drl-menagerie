"""
Main entry point for the application.
"""

# import json

from tac.experiment.control import RUN_MODES    # A list of available run modes; [Session, Trial, Experiment]
from tac.spec.spec_utils import analyse_spec    # Validates spec is in the correct format, and returns the run mode
from tac.spec import load_spec    # Loads the spec as a dictionary


# spec_path = "drl_menagerie/spec/reinforce/reinforce_cartpole.json"
# with open(spec_path, "r") as f:
#     spec = json.load(f)

# TODO: temp. Replace this pointer when temp spec python files replaced with the full spec json files
ALGORITHM = "reinforce"
spec_dict = load_spec(ALGORITHM)


def run_spec(spec):
    # Determine the run mode from the spec
    run_mode_name = analyse_spec(spec)

    # Get run mode class (Session, Trial, or Experiment)
    if run_mode_name in RUN_MODES:
        run_mode = RUN_MODES[run_mode_name]
    else:
        raise ValueError(f"Invalid run mode: {run_mode_name}")

    # Instantiate a Session, Trial, or Experiment object
    run = run_mode(spec)

    # Run the session, trial, or experiment
    run.run()


if __name__ == "__main__":
    run_spec(spec_dict)
