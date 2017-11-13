# pylint: disable=C0111
from investor_gym.envs.asset import Asset
from investor_gym.envs.portfolio import Portfolio
from investor_gym.envs.custom_types import Currency, Holdings, Days
import numpy as np


class StartPortfolio:
    def __init__(self, num_days: Days,
                 random_state: np.random.RandomState) -> None:  # pylint: disable=E1103

        mu_scale = 1 / num_days
        sigma_scale = np.sqrt(2 / num_days)
        start_cash = Currency(10_000.00)

        bank = Asset(
            name="bank",
            mu=0.1 * mu_scale,
            sigma=0.0 * sigma_scale,
            initial_value=Currency(1.00),
            random_state=random_state)

        good_stock = Asset(
            name="good_stock",
            mu=1.0 * mu_scale,
            sigma=0.1 * sigma_scale,
            initial_value=Currency(1.00),
            random_state=random_state)

        bad_stock = Asset(
            name="bad_stock",
            mu=-1.0 * mu_scale,
            sigma=0.1 * sigma_scale,
            initial_value=Currency(1.00),
            random_state=random_state)

        holdings_list = [(bank, start_cash), (good_stock, Currency(0.00)),
                         (bad_stock, Currency(0.00))]

        random_state.shuffle(
            holdings_list)  # TODO init too early to be seeded by env.seed

        holdings = Holdings(tuple(holdings_list))

        self._portfolio = Portfolio(holdings)

    @property
    def portfolio(self) -> Portfolio:
        return self._portfolio
