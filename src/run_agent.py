import os
import sys
path = "C:\\Users\\UKGC-PC\\Documents\\metal-mario-master\\IndividualProject"
sys.path.append(path)
os.environ['OMP_NUM_THREADS'] = '1'
import torch
from controller.controller_a3c import Actor_Critic
import gym_super_mario_bros
from environment import instantiate_environment
from convolutional_ae import CAE
import torch.nn.functional as F
from random import randrange
import numpy as np
right_only = [['right'],['right','A'],['right','A','B']]
import heapq

def logger(info,action,done,c):
    status=""
    if (done):
        if info['flag_get']:
            status = "reached end"
        else:
            status = "dead"
    else:
        status = "alive"

    f = open(path+"\\{}\\trained_agent_progress.txt".format("src"),"a")
    f.write("Action Number: {} | Action taken: {} | Position: {} | Status: {} \n".format(c,''.join( action),info["x_pos"],status))
    f.close()



def run_test():
    torch.manual_seed(123)  
    np.random.seed(123)
   
    env, num_states, num_actions = instantiate_environment()
    CAE_model = CAE()
    
    a3c_model = Actor_Critic(num_states, num_actions)

    
    if torch.cuda.is_available():
        a3c_model.load_state_dict(torch.load("{}\\exp2_controller_A3C_enc2".format(path+"\\controller")))
        print("loaded")
        a3c_model.cuda()
        CAE_model.cuda()
    else:
        a3c_model.load_state_dict(torch.load("{}\\exp2_controller_A3C_enc2".format(path+"\\controller"),
                                         map_location=lambda storage, loc: storage))
    
    f = open(path+"\\{}\\trained_agent_progress.txt".format("src"),"a")
    f.write("==================================================== \n {} \n ==========================================\n".format("exp2"))
    f.close()

    
    CAE_model.load_state_dict(torch.load("{}\\cae_model_enc2".format(path+"\\trained_cae"),map_location='cpu'))


    a3c_model.eval()
    CAE_model.eval()
    state = torch.from_numpy(env.reset())
    state = state.cuda()
    done = True
    death_count=0
    
    while True:
        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)
            env.reset()

            # stop executing if agent dies
            if death_count==1:
                print("dead")
                break
            death_count+=1

        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
        if torch.cuda.is_available():
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
            state = state.cuda()
        output_cae = CAE_model(state)
       
        logits, _, h_0, c_0 = a3c_model(output_cae, h_0, c_0)
        policy = F.softmax(logits, dim=1)
        
        action = torch.argmax(policy).item()
        
        print(policy)
        print(right_only[action])
        state, _, done, info = env.step(action)

        logger(info,right_only[action],done,c)
        
        state = torch.from_numpy(state)
        state = state.cuda()
        env.render()
        if info["flag_get"]:
            print("Completed level")
            break
        
        c+=1
run_test()


