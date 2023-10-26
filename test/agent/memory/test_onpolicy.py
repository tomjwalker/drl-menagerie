import pytest
from tac.agent.memory.onpolicy import OnPolicyReplay, OnPolicyBatchReplay
from tac.agent.algorithm.reinforce import Reinforce
from tac.spec.reinforce.temp_spec import spec


class TestOnPolicyReplay:

    def test_update(self):

        agent = Reinforce(spec_dict=spec, input_size=1, output_size=1)
        replay = OnPolicyReplay(agent)

        replay.update(0, 0, 0, 0, False)
        replay.update(1, 2, 3, 4, False)
        replay.update(5, 6, 7, 8, True)
        replay.update(20, 21, 22, 23, True)

        assert replay.size == 4
        assert replay.seen_size == 4
        assert replay.states == [[0, 1, 5], [20]]
        assert replay.actions == [[0, 2, 6], [21]]
        assert replay.rewards == [[0, 3, 7], [22]]
        assert replay.next_states == [[0, 4, 8], [23]]
        assert replay.dones == [[False, False, True], [True]]

