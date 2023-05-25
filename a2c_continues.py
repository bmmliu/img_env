import random
import yaml
import time
from envs import make_env, read_yaml
import sys
import os
import time
import math
import ptan
import gym
#import gymnasium as gym
# import pybullet_envs
import argparse
from tensorboardX import SummaryWriter

from lib import common

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import openai

openai.api_key = "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"
HID_SIZE = 128
GAMMA = 0.99
REWARD_STEPS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.001
ENTROPY_BETA = 1e-4

TEST_ITERS = 10000

class ModelA2C(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ModelA2C, self).__init__()

        self.base = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
        )
        self.mu = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh(),
        )
        self.var = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Softplus(),
        )
        self.value = nn.Linear(HID_SIZE, 1)

    def forward(self, x):
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out), self.value(base_out)


class AgentA2C(ptan.agent.BaseAgent):
    def __init__(self, net, device="cpu"):
        self.net = net
        self.device = device

    def __call__(self, states, agent_states):
        concatenate_states = [np.concatenate((states[0][0][0][0].flatten(), states[0][1][0], states[0][2][0][0].flatten(), states[0][2][0][1].flatten(), states[0][2][0][2].flatten())).tolist()]


        states_v = ptan.agent.float32_preprocessor(concatenate_states).to(self.device)

        mu_v, var_v, _ = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        sigma = torch.sqrt(var_v).data.cpu().numpy()
        actions = np.random.normal(mu, sigma)

        actions[0][0] = np.clip(actions[0][0], 0, 0.6)
        actions[0][1] = np.clip(actions[0][1], -0.9, 0.9)

        return [actions], agent_states

def test_net(net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        obs = [np.concatenate((obs[0][0][0].flatten(), obs[1][0], obs[2][0][0].flatten(), obs[2][0][1].flatten(), obs[2][0][2].flatten())).tolist()]

        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)[0]
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action[0][0] = np.clip(action[0][0], 0, 0.6)
            action[0][1] = np.clip(action[0][1], -0.9, 0.9)
            obs, reward, done, _ = env.step(action)
            obs = [np.concatenate((obs[0][0][0].flatten(), obs[1][0], obs[2][0][0].flatten(), obs[2][0][1].flatten(),
                                   obs[2][0][2].flatten())).tolist()]
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


def calc_logprob(mu_v, var_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2*var_v.clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * var_v))
    return p1 + p2


def getResponse(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a human being walking on the road."},
            {"role": "user", "content": "Behind on your left!"},
            {"role": "assistant", "content": "I will move to the right."},
            {"role": "user", "content": "You can pass me."},
            {"role": "assistant", "content": "Thank you, I will go first"},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0]["message"]["content"]


def testResponseToHuman(logs_text, prompt):
    logs_text = logs_text + "Robot: {0}\n".format(prompt)
    logs_text = logs_text + "Human: {0}\n".format(getResponse(prompt))

    return logs_text

if __name__ == "__main__":

    common.mkdir('.', 'checkpoints')
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    save_path = os.path.join("./checkpoints/", "a2c-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    cfg = read_yaml('envs/cfg/circle.yaml')
    env = make_env(cfg)
    state = env.reset()



    net = ModelA2C(9225, 2).to(device)
    print(net)

    writer = SummaryWriter(comment="-a2c_" + args.name)
    agent = AgentA2C(net, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=REWARD_STEPS)
    print(exp_source)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    batch = []
    best_reward = None
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:

            for step_idx, exp in enumerate(exp_source):
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], step_idx)
                    tracker.reward(rewards[0], step_idx)

                if step_idx % TEST_ITERS == 0:
                    ts = time.time()
                    rewards, steps = test_net(net, env, device=device)
                    print("Test done is %.2f sec, reward %.3f, steps %d" % (
                        time.time() - ts, rewards, steps))
                    writer.add_scalar("test_reward", rewards, step_idx)
                    writer.add_scalar("test_steps", steps, step_idx)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                            name = "best_%+.3f_%d.dat" % (rewards, step_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(net.state_dict(), fname)
                        best_reward = rewards

                batch.append(exp)
                if len(batch) < BATCH_SIZE:
                    continue

                states_v, actions_v, vals_ref_v = \
                    common.unpack_batch_continuous(batch, net, last_val_gamma=GAMMA ** REWARD_STEPS, device=device)
                batch.clear()

                optimizer.zero_grad()
                mu_v, var_v, value_v = net(states_v)

                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
                log_prob_v = adv_v * calc_logprob(mu_v, var_v, actions_v)
                loss_policy_v = -log_prob_v.mean()
                entropy_loss_v = ENTROPY_BETA * (-(torch.log(2*math.pi*var_v) + 1)/2).mean()

                loss_v = loss_policy_v + entropy_loss_v + loss_value_v
                loss_v.backward()
                optimizer.step()

                tb_tracker.track("advantage", adv_v, step_idx)
                tb_tracker.track("values", value_v, step_idx)
                tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                tb_tracker.track("loss_value", loss_value_v, step_idx)
                tb_tracker.track("loss_total", loss_v, step_idx)

