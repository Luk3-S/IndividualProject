import sys  
path = "C:\\Users\\UKGC-PC\\Documents\\metal-mario-master\\IndividualProject"
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import os
import torch
from src.environment import instantiate_environment
from right.actorcritic import Actor_Critic
from src.convolutional_ae import CAE
from src.SharedAdam import SharedAdam
from src.all_train import train
import torch.multiprocessing as mp


MOVEMENT_OPTIONS = [['right'], ['A'], ['down'],['B'],['right','A'],['right','A','B']]
right_only = [['right'],['right','A'],['right','A','B']]



def setup_training (button_to_train,prev_button_pos,eps):
    torch.manual_seed(123)
    env, num_states, num_actions = instantiate_environment()
    test_name = "no_coin"
    button_path=""
    if prev_button_pos >=0:
        button_path = ''.join( right_only[prev_button_pos])

    CAE_shared_model = CAE().to(torch.device('cuda'))
    A3C_shared_model = Actor_Critic(num_states,num_actions).to(torch.device('cuda'))

    CAE_shared_model.cuda()
    A3C_shared_model.cuda()

    CAE_shared_model.share_memory() 
    A3C_shared_model.share_memory() 

    

    optimiser_cae = CAE_shared_model.createLossAndOptimiser(CAE_shared_model,0.001)
    optimiser_a3c = SharedAdam(A3C_shared_model.parameters(),lr =0.001)

    if prev_button_pos >=0 :  # no prior parameters to load if loading first behaviour  
        print('{} being trained - attempting to load previous A3C parameters ...'.format(button_to_train))
        print("{}\\{}\\{}_{}_A3C_SMB_enc".format(path,button_path,test_name,button_path))
        try:
            pretrained_dict = torch.load("{}\\{}\\{}_{}_A3C_SMB_enc2".format(path,button_path,test_name,button_path))

            model_dict = A3C_shared_model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict) 
            A3C_shared_model.load_state_dict(model_dict)
            print("loaded parameters")
        except:
            print("Failed to load parameters")
    else:
        print("Beginning train to train first behaviour: {} - no prior parameters to load.".format(button_to_train))

    train(optimiser_a3c,A3C_shared_model,CAE_shared_model,optimiser_cae, button_to_train,eps)

    # for _ in range(0,4):
    #     process = mp.Process(target=train, args = (optimiser_a3c,A3C_shared_model,CAE_shared_model,optimiser_cae, button_to_train))
    #     process.start()
    #     threads.append(process)
    
    # for thread in threads:
    #     thread.join()

    #train(1, optimiser_a3c,A3C_shared_model,CAE_shared_model,optimiser_cae,True, button_to_train)
# pos =-1
# for button in MOVEMENT_OPTIONS:
#     print(button)
if __name__ == '__main__':  
#def right_main()
   #setup_training(]'right',-1)
   print("")


#     pos+=1
#right_main()