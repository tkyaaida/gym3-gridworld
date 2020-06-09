from typing import Tuple
import torch
from torch import Tensor
import gym3
from gym3.types import Discrete, TensorType


UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3


class GridWorld(gym3.Env):
    """Gridworld with no reward and no episode termination. Following gym3 interface"""
    def __init__(self, n_env: int, n: int, device: torch.device):
        ob_shape = (n_env, 2)  # n x n gridworld where each row indicates agent's position
        ac_shape = (n_env, )
        ob_space = TensorType(Discrete(n), ob_shape)
        ac_space = TensorType(Discrete(4), ac_shape)
        super(GridWorld, self).__init__(ob_space, ac_space, n_env)

        self.state = torch.ones(ob_shape, device=device, dtype=torch.long) * (n // 2)
        self.reward = torch.zeros(ac_shape, device=device, dtype=torch.float)
        self.first = torch.ones(ac_shape, device=device, dtype=torch.bool)

        self.n = n
        self.ob_shape = ob_shape
        self.ac_shape = ac_shape
        self.device = device

    def act(self, ac: Tensor) -> None:
        """act one environment step

        Args:
            ac (Tensor): action of shape (n_env, ) and each value is 0~3
        """
        next_state = self.get_next_state(self.state, ac)
        reward = self.compute_reward(next_state)
        first = self.is_done(next_state)
        next_state = self.reset_episode(next_state, first)

        self.state = next_state
        self.reward = reward
        self.first = first

    def observe(self) -> Tuple[Tensor, Tensor, Tensor]:
        return self.reward, self.state, self.first

    def get_next_state(self, state: Tensor, ac: Tensor) -> Tensor:
        up = -1 * (ac == UP)
        down = ac == DOWN
        left = -1 * (ac == LEFT)
        right = ac == RIGHT
        state[:, 0] = state[:, 0] + up + down
        state[:, 1] = state[:, 1] + left + right
        state = torch.clamp(state, 0, self.n-1)
        return state

    def compute_reward(self, state: Tensor) -> Tensor:
        return torch.zeros(self.ac_shape, device=self.device)

    def is_done(self, state: Tensor) -> Tensor:
        return torch.zeros(self.ac_shape, device=self.device)

    def reset_episode(self, next_state: Tensor, first: Tensor) -> Tensor:
        """reset episode if finished

        Args:
            next_state:
            first:

        Returns:

        """
        return next_state
