from tac.spec import load_spec
import pytest

from tac.spec.reinforce.temp_spec import spec as reinforce_spec
from tac.spec.sarsa.temp_spec import spec as sarsa_spec


def test_load_spec_reinforce():
    # Test if load_spec returns the correct spec for "reinforce"
    spec = load_spec("reinforce")
    expected_spec = {}  # Replace with the expected spec for "reinforce"
    assert spec == reinforce_spec


def test_load_spec_sarsa():
    # Test if load_spec returns the correct spec for "sarsa"
    spec = load_spec("sarsa")
    expected_spec = sarsa_spec  # Replace with the expected spec for "sarsa"
    assert spec == expected_spec


def test_load_spec_invalid_algorithm():
    # Test if load_spec raises a KeyError for an invalid algorithm
    with pytest.raises(KeyError):
        load_spec("invalid_algorithm")
