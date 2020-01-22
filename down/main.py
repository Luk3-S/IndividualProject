import sys  
sys.path.append("C:\\Users\\UKGC-PC\\Documents\\Level 4 Project")

import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import os
import torch
from src.environment import create_env
from down.actorcritic import Actor_Critic
from src.convolutional_ae import CAE
from src.SharedAdam import SharedAdam
from down.train import train

def run_down():
        
    torch.manual_seed(123)


    button = 'down'
    env, num_states, num_actions = create_env(1,1,button,10)

    print("num states: {}".format(num_states))
    print("num actions: {}".format(num_actions))
    print("env: {}".format(env))

    CAE_shared_model = CAE()
    A3C_shared_model = Actor_Critic(num_states,num_actions)

    CAE_shared_model.share_memory()
    A3C_shared_model.share_memory()
    #print("A3C - shared")
    #print(A3C_shared_model)
    #print("num states: {} , num actions: {}".format(num_states,num_actions))


    optimiser_cae = CAE_shared_model.createLossAndOptimiser(CAE_shared_model,0.001)
    optimiser_a3c = SharedAdam(A3C_shared_model.parameters(),lr =0.001)

    train(1, optimiser_a3c,A3C_shared_model,CAE_shared_model,optimiser_cae,False)
run_down()