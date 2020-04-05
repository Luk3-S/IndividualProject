import sys  
path = "C:\\Users\\UKGC-PC\\Documents\\metal-mario-master\\IndividualProject" ## add folder path here
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import os
import torch
from src.environment import instantiate_environment
from right.actorcritic import Actor_Critic
from src.convolutional_ae import CAE
from src.all_train import train
import torch.multiprocessing as mp
import numpy as np


right_only = [['right'],['right','A'],['right','A','B']]



def setup_training (button_to_train,prev_button_pos,eps,test_name,experiment_num):
  
    torch.manual_seed(123) 
    np.random.seed(123)
    _, num_states, num_actions = instantiate_environment()
    button_path=""
    if prev_button_pos >=0:
        button_path = ''.join( right_only[prev_button_pos])

    test_name = test_name
    
    cae_shared = CAE().to(torch.device('cuda'))
    a3c_shared = Actor_Critic(num_states,num_actions).to(torch.device('cuda'))

    cae_shared.cuda()
    a3c_shared.cuda()

    cae_shared.share_memory() 
    a3c_shared.share_memory() 


    optimiser_a3c = optim.Adam(a3c_shared.parameters(),lr =0.001)

    if prev_button_pos >=0 :  # no prior parameters to load if loading first behaviour  
        print('{} being trained - attempting to load previous A3C parameters ...'.format(button_to_train))
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
    else:
        print("Beginning train to train first behaviour (right) - no prior parameters to load.")

    train(optimiser_a3c,a3c_shared,cae_shared, button_to_train,eps,test_name,experiment_num)

    threads = []
    for _ in range(0,4):
        process = mp.Process(target=train, args = (optimiser_a3c,a3c_shared,cae_shared, button_to_train,eps,test_name,experiment_num))
        process.start()
        threads.append(process)
    
    for thread in threads:
        thread.join()

if __name__ == '__main__':  

    setup_training(['right'],-1,500,"",0)
