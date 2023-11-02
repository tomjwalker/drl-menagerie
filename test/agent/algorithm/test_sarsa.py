import numpy as np
import torch
from tac.agent.algorithm.sarsa import Sarsa
from tac.agent.memory.onpolicy import OnPolicyBatchReplay
from tac.utils.general import to_torch_batch
import pytest
import gymnasium as gym


SPEC_DICT = {
    "learning_rate": 0.001,
    "gamma": 0.99,
    "epsilon": 0.1,
    "environment": "CartPole-v1",
    "optimiser": "adam",
    "training_frequency": 1,
    "hidden_layer_units": [64],
    "activation": "relu",
}
INPUT_SIZE = 4
OUTPUT_SIZE = 2


class TestSarsa:

    def test_init(self):

        # Instantiate a Sarsa object
        sarsa = Sarsa(spec_dict=SPEC_DICT, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE)

        # Check that the Sarsa object has the correct attributes
        assert isinstance(sarsa.net, torch.nn.Module)
        assert isinstance(sarsa.optimiser, torch.optim.Adam)
        assert sarsa.discount_factor == 0.99
        assert sarsa.epsilon == 0.1
        assert sarsa.environment.spec.id == "CartPole-v1"
        assert sarsa.action_space == gym.make("CartPole-v1").action_space
        assert sarsa.ready_to_train is False
        assert sarsa.training_frequency == 1

    def test_on_policy_reset(self):
        raise NotImplementedError

    def test_calc_pdparam(self):
        # Create a mock network that returns predefined logits
        class MockNetwork:
            def forward(self, state):
                # Replace with example logits
                return torch.tensor([1.0, 2.0])

        # Set up inputs
        sarsa = Sarsa(spec_dict=SPEC_DICT, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE)
        sarsa.net = MockNetwork()
        state = torch.tensor([0.1, 0.2, 0.3, 0.4])

        # Function under test
        pdparam = sarsa.calc_pdparam(state)

        # Check that the output is as expected
        expected_logits = torch.tensor([1.0, 2.0])
        assert torch.allclose(pdparam, expected_logits, atol=1e-6)

    def test_act(self):
        raise NotImplementedError

    def test_sample(self):
        raise NotImplementedError

    def test_calc_q_loss(self):
        raise NotImplementedError

    def test_train(self):
        raise NotImplementedError

    def test_update(self):

        # In the SARSA class definition, this method is currently not implemented. This test checks that an error is
        # raised if the method is called.

        sarsa = Sarsa(spec_dict=SPEC_DICT, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE)
        state = np.array([1, 2, 3, 4])
        action = sarsa.act(state)
        reward = 1.0
        with pytest.raises(NotImplementedError):
            sarsa.update(state, action, reward, state, False)