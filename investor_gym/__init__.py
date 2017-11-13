# pylint: disable=C0111
from gym.envs.registration import register

register(
    id='Investor-v0', entry_point='investor_gym.envs.investor_env:InvestorEnv')
