# pylint: disable=E1103
# pylint: disable=W,C,R

import sys
sys.path.append("..")

import unittest
import numpy as np
from investor_gym.envs.asset import Asset
from investor_gym.envs.portfolio import Portfolio
from investor_gym.envs.investor_env import InvestorEnv
from investor_gym.envs.custom_types import *


class TestAssetMethods(unittest.TestCase):
    def test_asset_tick(self) -> None:
        seed = np.random.randint(999999)
        n = 500
        mu = 0.15 / n
        dt = 1
        x0 = 1423
        sigma = 1.12 / n

        bank = Asset(mu, sigma, Currency(x0), np.random.RandomState(seed))
        bank.reset()
        x = [bank.tick() for t in range(0, n)]

        # Code below from https://en.wikipedia.org/wiki/Geometric_Brownian_motion#Simulating_sample_paths
        np.random.seed(seed)
        step = np.exp((mu - sigma**2 / 2) * dt) * np.exp(
            sigma * np.random.normal(0, dt, (1, n)))
        y = x0 * step.cumprod()

        self.assertAlmostEqual(np.linalg.norm(x - y), 0)


class TestPortfolioMethods(unittest.TestCase):
    def test_portfolio_returns(self) -> None:
        bank = Asset(0.0912, 0, Currency(1), np.random.RandomState())
        stock = Asset(-0.14, .02, Currency(23.23), np.random.RandomState())
        bond = Asset(0.1, 0.05, Currency(53.12), np.random.RandomState())
        other = Asset(0.02, 0.05, Currency(13.22), np.random.RandomState())
        holdings = Holdings(
            ((bank, Currency(22.12)), (stock, Currency(551.56)),
             (bond, Currency(121.11)), (other, Currency(93.13))))

        portfolio = Portfolio(holdings)
        start_value = portfolio.value
        rewards = []
        for _ in range(1243):
            portfolio.tick()
            rewards.append(portfolio.portfolio_return)

        self.assertAlmostEqual(
            sum(rewards), np.log(portfolio.value / start_value))

    def test_portfolio_rebalance(self) -> None:
        bank = Asset(0.05, 0, Currency(1), np.random.RandomState())
        stock = Asset(-0.1, .2, Currency(23.23), np.random.RandomState())
        bond = Asset(0.1, 0.05, Currency(53.12), np.random.RandomState())

        holdings = Holdings(((bank, Currency(12.12)), (stock, Currency(51.06)),
                             (bond, Currency(91.11))))

        portfolio = Portfolio(holdings)
        value_before = portfolio.value
        action = Action(tuple(np.random.uniform(0.0, 1.0, len(holdings))))
        weights = InvestorEnv._weights(action)
        portfolio.rebalance(weights)
        value_after = portfolio.value

        self.assertAlmostEqual(value_after, value_before)
        self.assertAlmostEqual(
            np.linalg.norm(np.array(weights) - np.array(portfolio.weights)), 0)


if __name__ == "__main__":
    unittest.main()