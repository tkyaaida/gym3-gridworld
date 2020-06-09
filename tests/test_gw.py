import pytest
import torch
from torch import Tensor

from gym3_gridworld.gw import GridWorld, UP, DOWN, LEFT, RIGHT


class TestGridWorld:
    @pytest.fixture
    def gw(self):
        gw = GridWorld(4, 5, torch.device('cpu'))
        return gw

    def test_get_next_state(self, gw):
        reward, state, first = gw.observe()
        assert torch.equal(state, Tensor([[2, 2], [2, 2], [2, 2], [2, 2]]).to(torch.long))
        ac = Tensor([UP, DOWN, LEFT, RIGHT]).long()
        gw.act(ac)
        reward, state, first = gw.observe()
        assert torch.equal(state, Tensor([[1, 2], [3, 2], [2, 1], [2, 3]]).to(torch.long))
        gw.act(ac)
        reward, state, first = gw.observe()
        assert torch.equal(state, Tensor([[0, 2], [4, 2], [2, 0], [2, 4]]).to(torch.long))
        gw.act(ac)
        reward, state, first = gw.observe()
        assert torch.equal(state, Tensor([[0, 2], [4, 2], [2, 0], [2, 4]]).to(torch.long))
