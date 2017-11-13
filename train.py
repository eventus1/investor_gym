# Copyright 2017 reinforce.io. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numpy as np

from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

import investor_gym as ig

#from tensorforce.core.preprocessing import Sequence  # http://tensorforce.readthedocs.io/en/latest/preprocessing.html

# Create an OpenAIgym environment
# env = OpenAIGym('CartPole-v0')
env = OpenAIGym('Investor-v0')

preprocessing_config = [{"type": "sequence", "length": 4}]
# Network as list of layers
network_spec = [
    dict(type='dense', size=32, activation='tanh'),
    dict(type='dense', size=32, activation='tanh')
]

# https://github.com/reinforceio/tensorforce/blob/master/tensorforce/agents/ppo_agent.py
# https://openai-public.s3-us-west-2.amazonaws.com/blog/2017-07/ppo/ppo-arxiv.pdf
agent = PPOAgent(
    states_spec=env.states,
    actions_spec=env.actions,
    network_spec=network_spec,
    batch_size=4096,
    # Agent
    preprocessing=preprocessing_config,
    exploration=None,
    reward_preprocessing=None,
    # BatchAgent
    keep_last_timestep=True,
    # PPOAgent
    step_optimizer=dict(type='adam', learning_rate=1e-3),
    optimization_steps=10,
    # Model
    scope='ppo',
    discount=0.99,
    # DistributionModel
    distributions_spec=None,
    entropy_regularization=0.01,
    # PGModel
    baseline_mode=None,
    baseline=None,
    baseline_optimizer=None,
    gae_lambda=None,
    normalize_rewards=False,
    # PGLRModel
    likelihood_ratio_clipping=0.2,
    summary_spec=None,
    distributed_spec=None)

# Create the runner
runner = Runner(agent=agent, environment=env)


# Callback function printing episode statistics
def episode_finished(r):
    n = 10
    if r.episode % n == 0:
        R = '\033[31m'  # red
        G = '\033[32m'  # green
        W = '\033[0m'  # white
        avg_reward = np.mean(r.episode_rewards[-n:])
        col = G if avg_reward > 0 else R
        print(
            f"Finished episode {r.episode} after {r.episode_timestep} timesteps, "
            f"{W}average of last {n} rewards: {col}{avg_reward: .4f}{W}")
    return True


# Start learning
runner.run(
    episodes=2000,
    max_episode_timesteps=1000,
    episode_finished=episode_finished)

# Print statistics
print(
    f"Learning finished. Total episodes: {runner.episode}."
    f" Average reward of last 100 episodes: {np.mean(runner.episode_rewards[-100:])}."
)
