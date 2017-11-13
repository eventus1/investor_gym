# pylint: disable=C0111
from investor_gym.envs.custom_types import Currency, Return, Holdings, Weights, Proportion, Tuple, Observation
import numpy as np


class Portfolio:
    def __init__(self, holdings: Holdings) -> None:
        self._holdings = holdings
        self._last_value = self.value
        self._this_value = self._last_value

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(asset.name for asset, _ in self._holdings)

    @property
    def value(self) -> Currency:
        return Currency(
            sum(asset.value * amount for asset, amount in self._holdings))

    @property
    def asset_returns(self) -> Observation:
        return Observation(
            tuple(Return(asset.log_return) for asset, _ in self._holdings))

    @property
    def portfolio_return(self) -> Return:
        return Return(np.log(self._this_value / self._last_value))  # pylint: disable=E1103

    @property
    def ticks(self) -> Tuple[Currency, ...]:
        return tuple((Currency(asset.value) for asset, _ in self._holdings))

    @property
    def weights(self) -> Weights:
        value = self.value
        return Weights(
            tuple(
                Proportion(asset.value * amount / value)
                for asset, amount in self._holdings))

    @property
    def size(self) -> int:
        return len(self._holdings)

    def tick(self) -> None:
        for asset, _ in self._holdings:
            asset.tick()
        self._last_value = self._this_value
        self._this_value = self.value

    def rebalance(self, weights: Weights) -> None:
        value = self.value
        self._holdings = Holdings(
            tuple((asset, Currency(value * proportion / asset.value))
                  for ((asset, amount), proportion) in zip(
                      self._holdings, weights)))
