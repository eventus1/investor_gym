# pylint: disable=C0111
from typing import Iterator
from investor_gym.envs.custom_types import Currency, Return
import numpy as np


class Asset:
    def __init__(
            self,
            mu: float,
            sigma: float,
            initial_value: Currency,
            random_state: np.random.RandomState,  # pylint: disable=E1103
            name: str = "",
    ) -> None:
        self._name = name
        self._random_state = random_state
        self._mu = mu
        self._sigma = sigma
        self._initial_value = initial_value
        self._ticker = self._gen_ticker()
        self._last_tick = initial_value
        self._this_tick = initial_value
        self._value = initial_value

    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> Currency:
        return self._value

    @property
    def log_return(self) -> Return:
        """
        Using log returns makes the cumulative returns equal the total return.
        Expected log return of a GBM asset = mu * t - sigma**2 * t / 2
        """
        return Return(np.log(self._this_tick / self._last_tick))  # pylint: disable=E1103

    def tick(self) -> Currency:
        self._last_tick = self._this_tick
        self._this_tick = next(self._ticker)
        return self._this_tick

    def reset(self) -> None:
        self._ticker = self._gen_ticker()

    def _gen_ticker(self) -> Iterator[Currency]:
        '''
        Produces geometric brownian motion
        '''
        multiplier = 1.0
        while True:
            new_multiplier = multiplier * np.exp(
                (self._mu - self._sigma**2.0 / 2.0) +
                self._sigma * self._random_state.normal(0.0, 1.0))  # pylint: disable=E1103
            self._value = self._initial_value * new_multiplier
            yield Currency(self._value)
            multiplier = new_multiplier
