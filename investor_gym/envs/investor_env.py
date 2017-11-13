# pylint: disable=C0111
from typing import Optional, Tuple
from investor_gym.envs.custom_types import Days, Observation, Info, Done, Currency, Reward, Action, Proportion, Return, Weights
from investor_gym.envs.portfolio import Portfolio
from investor_gym.envs.startportfolio import StartPortfolio
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class InvestorEnv(gym.Env):  # type: ignore
    def __init__(self, num_days: Days = Days(100)) -> None:
        self.metadata = {"render.modes": ["human"]}
        self.reward_range = (-np.inf, np.inf)
        self._num_days = num_days
        self.portfolio: Portfolio
        self._day: Days
        self.np_random: np.random.RandomState  # pylint: disable=E1103
        self._seed()
        self._reset()  # Must come before space declarations!
        self.action_space = spaces.Box(
            Proportion(0.0), Proportion(1.0), self.portfolio.size)
        self.observation_space = spaces.Box(
            Return(-1.0), Return(1.0), self.portfolio.size)

    @property
    def done(self) -> Done:
        return Done(self.portfolio.value <= Currency(0.00)
                    or self._day >= self._num_days)

    @property
    def observation(self) -> Observation:
        return self.portfolio.asset_returns

    @property
    def reward(self) -> Reward:
        return Reward(self.portfolio.portfolio_return)

    @property
    def info(self) -> Info:
        return Info({})

    def _tick(self) -> None:
        self.portfolio.tick()
        self._day = Days(self._day + 1)

    def _step(self, action: Action) -> Tuple[Observation, Reward, Done, Info]:
        weights = self._weights(action)
        self.portfolio.rebalance(weights)
        self._tick()
        return self.observation, self.reward, self.done, self.info

    def _reset(self) -> Observation:
        self._day = Days(0)
        self.portfolio = StartPortfolio(self._num_days,
                                        self.np_random).portfolio
        return self.observation

    def _render(self, mode: str = "human", close: bool = False) -> None:
        if close:
            print("Closing gym environment.")
        else:
            print(
                f"day: {self._day:3d},\t"
                f"value: {self.portfolio.value:.2f},\t"
                f"weights: {tuple((f'{w:.2f}') for w in self.portfolio.weights)}"
            )

    def _seed(self, seed: Optional[int] = None) -> Tuple[int]:
        self.np_random, seed_prime = seeding.np_random(seed)
        return (seed_prime, )

    @staticmethod
    def _weights(action: Action) -> Weights:
        sum_action = sum(action)
        return Weights(tuple(Proportion(a / sum_action) for a in action))
