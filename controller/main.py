import sys  
path = "C:\\Users\\UKGC-PC\\Documents\\metal-mario-master\\IndividualProject"
sys.path.append(path)
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import os
import torch
from src.environment import instantiate_environment
from controller.controller_a3c import Actor_Critic
from src.convolutional_ae import CAE
from controller.train import train
import torch.multiprocessing as mp
import numpy as np

right_only = [['right'],['right','A'],['right','A','B']]

def run_controller_train (prev_button_pos,test_name,experiment_num):
    torch.manual_seed(123)  
    np.random.seed(123)
  
    env, num_states, num_actions = instantiate_environment()
    button_path=""
    test_name = test_name
    
    if prev_button_pos >=0:
        button_path = ''.join( right_only[prev_button_pos])
  
    print("env: {}".format(env))

    cae_shared = CAE().to(torch.device('cuda'))
    a3c_shared = Actor_Critic(num_states,num_actions).to(torch.device('cuda'))

    cae_shared.share_memory().cuda()
    a3c_shared.share_memory().cuda()

    print('Attempting to load A3C parameters : {} ...'.format(button_path))
   
    print("{}\\{}\\{}_{}_A3C_enc2".format(path,button_path,test_name,button_path))
    try:
        
        
        pretrained_dict = torch.load("{}\\{}\\{}_{}_A3C_enc2".format(path,button_path,test_name,button_path))
        
        model_dict = a3c_shared.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        a3c_shared.load_state_dict(model_dict)
        print("loaded parameters")
    except:
        print("Failed to load parameters")



    
    optimiser_a3c = optim.Adam(a3c_shared.parameters(),lr = 0.001)

    train(optimiser_a3c,a3c_shared,cae_shared,test_name,experiment_num)
    threads = []

    for _ in range(0,4):
        process = mp.Process(target=train, args = (optimiser_a3c,a3c_shared,cae_shared))
        process.start()
        threads.append(process)
    
    for thread in threads:
        thread.join()

    train(optimiser_a3c,a3c_shared,cae_shared,test_name,experiment_num)

if __name__ == '__main__':  

    run_controller_train(2,"",0)