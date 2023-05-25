import random
import yaml
import time
import argparse
from envs import make_env, read_yaml
from a2c_continues import ModelA2C
import numpy as np
import torch
import openai




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

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-r", "--record", help="If specified, sets the recording dir, default=Disabled")
    args = parser.parse_args()
    cfg = read_yaml('envs/cfg/circle.yaml')

    print(cfg)
    env = make_env(cfg)
    # env2 = make_env(cfg)
    # time.sleep(1)
    # test continuous action
    env.reset()

    net = ModelA2C(9225, 2)
    net.load_state_dict(torch.load(args.model))

    obs = env.reset()
    total_reward = 0.0
    total_steps = 0
    while True:

        obs = [np.concatenate((obs[0][0][0].flatten(), obs[1][0], obs[2][0][0].flatten(), obs[2][0][1].flatten(),
                               obs[2][0][2].flatten())).tolist()]
        obs_v = torch.FloatTensor([obs])
        mu_v, var_v, val_v = net(obs_v)
        action = mu_v.squeeze(dim=0).data.numpy()
        action[0][0] = np.clip(action[0][0], 0, 0.6)
        action[0][1] = np.clip(action[0][1], -0.9, 0.9)
        print(action)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            break
    print("In %d steps we got %.3f reward" % (total_steps, total_reward))
