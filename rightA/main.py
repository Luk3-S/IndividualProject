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
from rightA.actorcritic import Actor_Critic
from src.convolutional_ae import CAE
from rightA.train import train
import torch.multiprocessing as mp
import numpy as np
MOVEMENT_OPTIONS = [['right'], ['A'], ['left'], ['down'], ['up'],['B'],['right','A'],['right','A','B']]
right_only = [['right'],['right','A'],['right','A','B']]

def right_a_main(button_to_train):
    torch.manual_seed(123) 
    np.random.seed(123)
    _, num_states, num_actions = instantiate_environment()
    

    test_name = "running rightA on its own"
    
    cae_shared = CAE().to(torch.device('cuda'))
    a3c_shared = Actor_Critic(num_states,num_actions).to(torch.device('cuda'))

    cae_shared.cuda()
    a3c_shared.cuda()

    cae_shared.share_memory() 
    a3c_shared.share_memory() 

    
    

    
    optimiser_a3c = optim.Adam(a3c_shared.parameters(),lr =0.001)

    
    # no previous parameters to load
   

    threads = []

    for _ in range(0,4):
        process = mp.Process(target=train, args = (optimiser_a3c,a3c_shared,cae_shared, button_to_train,500,test_name,0))
        process.start()
        threads.append(process)
    
    for thread in threads:
        thread.join()


right_a_main(['right','A'])