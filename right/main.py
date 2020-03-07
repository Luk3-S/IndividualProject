import sys  
laptop_path = "C:\\Users\\Luke\\Documents\\diss proj\\IndividualProject"
desktop_path = "C:\\Users\\UKGC-PC\\Documents\\metal-mario-master\\IndividualProject"
sys.path.append(desktop_path)
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import os
import torch
from src.environment import create_env
from right.actorcritic import Actor_Critic
from src.convolutional_ae import CAE
from src.SharedAdam import SharedAdam
from right.train import train

MOVEMENT_OPTIONS = [['right'], ['A'], ['left'], ['down'], ['up'],['B'],['right','A'],['right','A','B']]


def run_right (button_to_train):
    torch.manual_seed(123)
    #button = 'down'
    env, num_states, num_actions = create_env(1,1)

    # print("num states: {}".format(num_states))
    # print("num actions: {}".format(num_actions))
    print("env: {}".format(env))

    CAE_shared_model = CAE()
    A3C_shared_model = Actor_Critic(num_states,num_actions)

    CAE_shared_model.share_memory()
    A3C_shared_model.share_memory()

    print("Beginning train process - no prior parameters to load.")
    #print('Attempting to load A3C parametets ...')
    # try:
    #     pretrained_dict = torch.load("{}\\A3C_super_mario_bros_{}_{}_enc".format(desktop_path+"\\{}".format("right"),1,1))
    #     model_dict = A3C_shared_model.state_dict()
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #     model_dict.update(pretrained_dict) 
    #     A3C_shared_model.load_state_dict(model_dict)
    #     print("loaded parameters")
    # except:
    #     print("Failed to load parameters")


    #print("A3C - shared")
    #print(A3C_shared_model)
    #print("num states: {} , num actions: {}".format(num_states,num_actions))


    optimiser_cae = CAE_shared_model.createLossAndOptimiser(CAE_shared_model,0.001)
    optimiser_a3c = SharedAdam(A3C_shared_model.parameters(),lr =0.001)

    
    train(1, optimiser_a3c,A3C_shared_model,CAE_shared_model,optimiser_cae,True, button_to_train)
# pos =-1
# for button in MOVEMENT_OPTIONS:
#     print(button)
def right_main():
    run_right(['right'])
#     pos+=1
right_main()