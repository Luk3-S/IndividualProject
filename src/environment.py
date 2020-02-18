import gym_super_mario_bros
from gym.spaces import Box
from gym import Wrapper
from nes_py.wrappers import JoypadSpace
import cv2
import numpy as np
import subprocess as sp

MOVEMENT_OPTIONS = [['right'], ['A'], ['left'], ['down'], ['up'],['B'],['right','A'],['right','A','B']]

def process_frame(frame):
    if frame is not None:
        #print('Frame shape before convertion: {}'.format(frame.shape))

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        #print('Frame shape between convertion: {}'.format(frame.shape))
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
        self.lifetime_max = 0 ## futherest x pos that mario has achieved in its lifetime
        self.step_count = 0
        self.ep_max = 0 ## currently episode prior's max
        #self.surpassed = False
        #self.ep_prev_x = 0
        self.stationary_count = 0
        self.curr_x = 0
        self.timesCounted = 0
        self.training_max = 0

    def step(self, action):
        self.env.render()

        state, reward, done, info = self.env.step(action)
        state = process_frame(state)
        #reward += (info["score"] - self.curr_score) / 10.
        reward = self.generate_reward(reward,info,done)
        return state, reward , done, info # reward/ 10.


    def generate_reward(self,reward,info,done):
        self.timesCounted+=1
        self.curr_score = info["score"]
        self.curr_x = info["x_pos"]
        self.step_count+=1

        ## episode-specific rewards
        if (self.curr_x > self.ep_max): ## if mario progresses during an episode, allocate reward and update the episode's max x pos
            reward+=3
            self.stationary_count=0
            self.ep_max = self.curr_x
        
        if (self.curr_x == self.ep_max): ## if mario isn't progressing - i.e. stuck  against pipe, negatively reward staying still
            self.stationary_count+=1
            self.ep_max = self.curr_x
            if (self.stationary_count >= 40): ## if we've been stationary for 30 actions negatively reward. However if been standing still for a long time, and then eventually take
                #reward-=3     
                if (self.curr_x > self.ep_max):
                    print("surpassed!")
                    self.ep_max = self.curr_x
                    self.stationary_count = 0
                    reward+=15

        ## rewards in play while mario is currently playing (either not dead, or time hasn't run out yet)

        if (self.ep_max > self.lifetime_max): ## if episode surpasses lifetime x pos
            reward+=5
            self.lifetime_max = self.ep_max
        
        if done:
            if info["flag_get"]:
                reward += 100
            else: ## dead / time ran out
                reward -= 50 ## 30
        
        # if (self.ep_max > self.training_max):
        #     print("training max exceeded")
        #     self.training_max = self.ep_max
        #     reward+=30
        self.reset_step_specifics


        # if (info["x_pos"] > self.max_x and info["x_pos"] >=100): # reward agent for progressing through level
        #     self.max_x = info["x_pos"]
        #     #print("new max reached: {}".format(self.max_x))
        #     reward+=2
        # # else:
        # #     if (info["x_pos"] < self.max_x):
        # #         reward-=5
        # if (self.max_x > self.ep_max and self.surpassed == False): # reward agent for improving upon previous episode
        #     print("agent surpassed previous episode")
        #     self.ep_max = self.max_x ## update prev ep max with current max
        #     reward+=2.5
        #     self.surpassed = True
        # if done:
        #     if info["flag_get"]:
        #         reward += 100
        #     else: ## dead / time ran out
        #         reward -= 100 ## 30

        # if (info["x_pos"] == self.prev_x): ## to dissuade agent from retracting / standing still - i.e. when facing a pipe / other object 
        #     reward-=1
        #     self.stationary_count+=1
        # # else: ## progressed
        # #     if info["x_pos"] > self.prev_x:
        # #         if (self.stationary_count >40): # 50 actions per episode, if struggling for long (40) episodes, reward when correct action chosen
        # #             print("reward allocated: stationary")
        # #             reward+=10
        
        
        # ## set prev_x to current
        # self.prev_x = info["x_pos"]
        # if (self.step_count ==250):
        #     print("reset step specifics")
        #     self.reset_step_specifics()
        #print("times Counted: {}".format(self.timesCounted))
        return reward
    def reset_step_specifics(self):
        self.ep_max = 0
        self.stationary_count = 0
        #self.surpassed = False
        self.step_count = 0
    
    
    def reset(self): ## called when agent dies / runs out of steps 
        self.curr_score = 0
        self.lifetime_max = 0

        print("full reset")
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
