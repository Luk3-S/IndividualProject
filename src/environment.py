import gym_super_mario_bros
from gym.spaces import Box
import gym
from gym import Wrapper,ObservationWrapper,spaces
from nes_py.wrappers import JoypadSpace
import numpy as np
import subprocess as sp
import torchvision
from gym.wrappers import FrameStack,LazyFrames
from torchvision import transforms, utils
import cv2
right_only = [['right'],['right','A'],['right','A','B']]

# inspiration taken from : https://github.com/openai/gym/blob/master/gym/wrappers/gray_scale_observation.py
def convert_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY) # convert colours to grayscale
        return np.expand_dims(cv2.resize(frame,(84,84)),axis=0) / 255 # resize image and add additional column
       
    else:
        return np.zeros((1, 84, 84))


class RewardHandler(Wrapper):
    def __init__(self, env=None):
        super(RewardHandler, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.current_score = 0
        self.lifetime_max_x = 0 ## futherest x pos that mario has achieved in its lifetime
        self.step_count = 0
        self.episode_max = 0 ## currently episode prior's max
        self.current_x = 0
       
        

    def step(self, action):
        #self.env.render()
        reward=0
        state, _, done, info = self.env.step(action)

        state = convert_frame(state)
        if (info["score"] == 100): ## 100 points awarded for killing a goomba, 200 for getting a coin. conly allocate rewards for
            reward += (info["score"] - self.current_score) /10.
        self.current_score = info["score"]

        reward += self.generate_reward(reward,info,done)
        return state, reward/10. , done, info


    def generate_reward(self,reward,info,done):
        self.current_x = info["x_pos"]
        self.step_count+=1

        ## episode-specific rewards
        if (self.current_x > self.episode_max): ## if mario progresses during an episode, allocate reward and update the episode's max x pos
            reward+=1
            self.episode_max = self.current_x

        if (self.episode_max > self.lifetime_max_x): ## if episode surpasses lifetime x pos
            reward+=1
            self.lifetime_max_x = self.episode_max
        
        if done:
            if info["flag_get"]:
                reward += 20 ## reward for reaching flag - this is low since 
            else: ## dead / time ran out
                reward -=8 
                
           
       
        if (self.step_count ==500): ## idea: and !done -> incentivise this only if not dead (done = False)
            print("Exceeded step count, resetting environment reward parameters")
            self.reset_ep_specifics()
        return reward

    def reset_ep_specifics(self): ## reset episode max only if 500 steps reached (reached end of episode)
        self.episode_max = 0
        self.step_count = 0
    
    
    def reset(self): ## called when agent dies / runs out of steps 
        self.current_score = 0
        self.lifetime_max_x = 0
        return convert_frame(self.env.reset())

#https://github.com/LecJackS/TP-Final-Procesos-Markovianos-para-el-Aprendizaje-Automatico-2019-1C/blob/32ae31a314f197afcad78ba6ee8ad60169868944/gym_pacman/src/env.py#
# code inspired by above link and ammended by me
class FrameHandler(Wrapper):
    def __init__(self, env):
        super(FrameHandler, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(4, 84, 84)) #4*3,84,84

    def step(self, action):
        states = []
        state, reward, done, info = self.env.step(action)
        for _ in range(4):
            if not done:
                state, reward, done, info = self.env.step(action)

            states.append(state)
        states = np.concatenate(states, 0)[None, :, :, :]
        return states.astype(np.float32), reward, done, info

    def reset(self):
        state = self.env.reset()
        states = np.concatenate([state for _ in range(4)], 0)[None, :, :, :]
        return states.astype(np.float32)

def instantiate_environment():
    env= gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = JoypadSpace(env=env,actions=right_only) 
    env = RewardHandler(env)
    env = FrameHandler(env)
    return env, env.observation_space.shape[0], len(right_only) 