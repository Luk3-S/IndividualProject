import sys  
laptop_path = "C:\\Users\\Luke\\Documents\\diss proj\\IndividualProject"
desktop_path = "C:\\Users\\UKGC-PC\\Documents\\metal-mario-master\\IndividualProject"
sys.path.append(desktop_path)
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import os
import torch
from src.environment import instantiate_environment
from right_a.actorcritic import Actor_Critic
from src.convolutional_ae import CAE
from src.SharedAdam import SharedAdam
from right_a_b.train import train
import torch.multiprocessing as mp

MOVEMENT_OPTIONS = [['right'], ['A'], ['left'], ['down'], ['up'],['B'],['right','A'],['right','A','B']]
right_only = [['right'],['right','A'],['right','A','B']]

def run_right_a_b (button_to_train):
    torch.manual_seed(123)
    #button = 'down'
    env, num_states, num_actions = instantiate_environment(1,1)

    # print("num states: {}".format(num_states))
    # print("num actions: {}".format(num_actions))
    print("env: {}".format(env))

    CAE_shared_model = CAE().to(torch.device('cuda'))
    A3C_shared_model = Actor_Critic(num_states,num_actions).to(torch.device('cuda'))

    CAE_shared_model.share_memory().cuda()
    A3C_shared_model.share_memory().cuda()

    print('Attempting to load A3C parameters : right,a ...')
    try:
        pretrained_dict = torch.load("{}\\500A3C_super_mario_bros_{}_{}_enc".format(desktop_path+"\\{}".format("right_a"),1,1))#b
        model_dict = A3C_shared_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        A3C_shared_model.load_state_dict(model_dict)
        print("loaded parameters")
    except:
        print("Failed to load parameters")


    #print("A3C - shared")
    #print(A3C_shared_model)
    #print("num states: {} , num actions: {}".format(num_states,num_actions))


    optimiser_cae = CAE_shared_model.createLossAndOptimiser(CAE_shared_model,0.001)
    optimiser_a3c = SharedAdam(A3C_shared_model.parameters(),lr =0.001)

    
    threads = []

    for _ in range(0,4):
        process = mp.Process(target=train, args = (optimiser_a3c,A3C_shared_model,CAE_shared_model,optimiser_cae, button_to_train))
        process.start()
        threads.append(process)
    
    for thread in threads:
        thread.join()

    #train(1, optimiser_a3c,A3C_shared_model,CAE_shared_model,optimiser_cae,True, button_to_train)
# pos =-1
# for button in MOVEMENT_OPTIONS:
#     print(button)
if __name__ == '__main__':  
#def right_main():
    run_right_a_b(['right'])