import sys  
path = "C:\\Users\\UKGC-PC\\Documents\\metal-mario-master\\IndividualProject"
sys.path.append(path)
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import os
import torch
from src.environment import instantiate_environment
from rightAB.actorcritic import Actor_Critic
from src.convolutional_ae import CAE
from rightAB.train import train
import torch.multiprocessing as mp
import numpy as np


right_only = [['right'],['right','A'],['right','A','B']]

def run_right_a_b (button_to_train):
    torch.manual_seed(123)  
    np.random.seed(123)
    
    env, num_states, num_actions = instantiate_environment()

    print("env: {}".format(env))

    cae_shared_model = cae().to(torch.device('cuda'))
    a3c_shared = Actor_Critic(num_states,num_actions).to(torch.device('cuda'))

    cae_shared_model.share_memory().cuda()
    a3c_shared.share_memory().cuda()

    print('Attempting to load A3C parameters : right,a ...')
    try:
        pretrained_dict = torch.load("{}\\500A3C_super_mario_bros_{}_{}_enc".format(desktop_path+"\\{}".format("right_a"),1,1))#b
        model_dict = a3c_shared.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        a3c_shared.load_state_dict(model_dict)
        print("loaded parameters")
    except:
        print("Failed to load parameters")

    optimiser_a3c = optim.Adam(a3c_shared,lr = 0.001)

    
    threads = []

    for _ in range(0,4):
        process = mp.Process(target=train, args = (optimiser_a3c,a3c_shared,cae_shared_model, button_to_train))
        process.start()
        threads.append(process)
    
    for thread in threads:
        thread.join()


if __name__ == '__main__':  
#def right_main():
    run_right_a_b(['right'])