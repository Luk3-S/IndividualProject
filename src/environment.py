import gym_super_mario_bros
from gym.spaces import Box
from gym import Wrapper
from nes_py.wrappers import JoypadSpace

import cv2
import numpy as np
import subprocess as sp

MOVEMENT_OPTIONS = [['NOOP'], ['right'], ['A'], ['left'], ['down'], ['up'],['B']]

def process_frame(frame):
    if frame is not None:
        #print('Frame shape before convertion: {}'.format(frame.shape))

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        #print('Frame shape between convertion: {}'.format(frame.shape))
        # changing the size to match auto encoder and reshaping from (128,128,3) to (3, 128, 128)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        frame = frame.reshape(1,84,84)
        #print('Frame shape after convertion: {}'.format(frame.shape))

        return frame
    else:
        return np.zeros((1, 84, 84))



class GetReward(Wrapper):
    def __init__(self, env=None):
        super(GetReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0
        

    def step(self, action):
        self.env.render()
        state, reward, done, info = self.env.step(action)
        state = process_frame(state)
        
        reward += (info["score"] - self.curr_score) / 40.
        self.curr_score = info["score"]
        if done:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50
        return state, reward / 10., done, info

    def reset(self):
        self.curr_score = 0
        #print('Frame env reset: {}'.format(env.reset.shape))
        return process_frame(self.env.reset())


class GetFrame(Wrapper):
    def __init__(self, env):
        super(GetFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(4, 84, 84)) #4*3,84,84

    def step(self, action):
        total_reward = 0
        states = []
        state, reward, done, info = self.env.step(action)
        for _ in range(4):
            if not done:
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                states.append(state)
            else:
                states.append(state)
        states = np.concatenate(states, 0)[None, :, :, :]
        return states.astype(np.float32), reward, done, info

    def reset(self):
        state = self.env.reset()
        states = np.concatenate([state for _ in range(4)], 0)[None, :, :, :]
        return states.astype(np.float32)


def create_env(world,stage):
    env_name = "SuperMarioBros-{}-{}-v0".format(world,stage)
    env= gym_super_mario_bros.make(env_name)
    env = JoypadSpace(env=env,actions=MOVEMENT_OPTIONS) # joypad space wants actions input to be list of lists, hence above reformatting when passing a singular button in
    env = GetReward(env)
    env = GetFrame(env)
    return env, env.observation_space.shape[0], len(MOVEMENT_OPTIONS) 
