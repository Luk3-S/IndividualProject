import sys

path = "C:\\Users\\UKGC-PC\\Documents\\metal-mario-master\\IndividualProject"

import torch
import torch.nn as nn
import torch.nn.functional as F
import gym_super_mario_bros
import gym
import torch.multiprocessing as _mp
from src.environment import instantiate_environment
from src.convolutional_ae import CAE
from right.actorcritic import Actor_Critic
from torch.distributions import Categorical
from tensorboardX import SummaryWriter
import timeit
import numpy as np



def train (A3C_optimiser, a3c_shared,CAE_shared_model,button,eps,test_name,experiment_num):
    
    button_as_string = ''.join(button)
    BUTTON_PRESSED = False

    right_only = [['right'],['right','A'],['right','A','B']]
    no_steps = 100
    max_steps = 1600 ## max steps possible in 400 seconds
    no_episodes =eps# 1000
    test_name = test_name
    
    start_time = timeit.default_timer()
    env, num_states,num_actions = instantiate_environment()
 


    a3c_local_model = Actor_Critic(num_states,num_actions).cuda() 
    cae_local_model = CAE().cuda()  
    a3c_local_model.train()

    torch.manual_seed(123)  
    np.random.seed(123)

    state = torch.from_numpy(env.reset()) 
    state = state.cuda() 
   
    ## make autoencoder 

    step = 0
    episode = 0
    done = True
    save = True
    f = open(path+"\\{}\\ep_log.txt".format(button_as_string),"a")
    f.write("==================================================== \n {} \n ==========================================\n".format(test_name))
    f.write("Episode, loss,rewards,xpos,status\n")
    f.close()
    while True:
        print("episode: {}".format(episode))
        if save == True:
            if episode %100 ==0 : # 500 episode > 0 and episode % 100 ==0 
                print("saved")
                torch.save(a3c_shared.state_dict(),"{}\\{}\\{}_{}_A3C_enc2".format(path,button_as_string,test_name,button_as_string)) # 2nd is button as string
               

        a3c_local_model.load_state_dict(a3c_shared.state_dict())

        try:
            cae_local_model.load_state_dict(torch.load("{}\\CAE_super_mario_bros_1_1_enc1".format(path+"\\trained_cae"),map_location='cpu'))
            #print("Cae loaded")
        except:
            print("no file found")
        cae_local_model.eval()


        if done:
            hx = torch.zeros((1,512),dtype=torch.float)
            cx = torch.zeros((1,512),dtype=torch.float)
        else:
            hx = hx.detach()
            cx = cx.detach()
        
        hx = hx.cuda()  
        cx = cx.cuda()  

        log_policies =[]
        values=[]

        rewards=[]
        entropies=[]

        for _ in range(no_steps): #500
            #print("step: {}".format(step))
            step +=1

            for param in cae_local_model.parameters():
                param.requires_grad = False
            
            ## generate cae's representation of the environment
            #state = torch.Tensor(state)
            output_cae = cae_local_model(state)

            logits,value,hx,cx = a3c_local_model(output_cae,hx,cx)

            policy = F.softmax(logits,dim=1)
            log_policy = F.log_softmax(logits,dim=1)

            entropy = -(policy * log_policy).sum(1,keepdim=True)

            m = Categorical(policy)
            action = m.sample().item()
            
            
            if (right_only[action]==button):
                
              
                BUTTON_PRESSED = True
            
            state,reward,done,info = env.step(action)
            
            
            state = torch.from_numpy(state) 
            state = state.cuda()  

            if step > max_steps or episode > no_episodes:
                done = True

            if done:
                step = 0
                state = torch.from_numpy(env.reset()) 
                state = state.cuda()

            

            
            if (BUTTON_PRESSED):
               
                
            ## only do the following if we have our button pressed? i.e only reward when button 
                values.append(value)
                log_policies.append(log_policy[0,action])
                rewards.append(reward)
                entropies.append(entropy)
                BUTTON_PRESSED = False

                if done: 
                    print("died here")
                    break   
                
            
                R = torch.zeros((1,1),dtype=torch.float) 
                R= R.cuda() 
                if not done:
                    output = cae_local_model(state)
                    _,R,_,_ = a3c_local_model(output,hx,cx)

                gae = torch.zeros((1,1),dtype=torch.float) 
                gae = gae.cuda() 
                advantage_estimator = torch.zeros((1,1),dtype=torch.float) 
                advantage_estimator = advantage_estimator.cuda() 
               
               
       
        
        if (len(values)>0):## check if we've actually pressed said button, and if so, then apply rewards etc.
            policy_loss, value_loss = 0,0
        
            
            previous_value = R
     
            for i in range(len(rewards))[::-1]:
                
                # compute value loss: VL: L = sum(R - V(s))^2
                R = 0.9 * R + rewards[i]
                advantage = R - values[i]
                value_loss = value_loss + (advantage**2)/2

                ## advantage function estimate: reward[i] + gamma (value(st+1) - value(st))
                theta = rewards[i] + (0.9 * previous_value - values[i])
                advantage_estimator = advantage_estimator * 0.9  + theta
            
                # comput policy loss: -log(pi(s)) * A(s) - beta*H(pi)
                policy_loss = policy_loss - log_policies[i] *Variable(advantage_estimator).type(torch.cuda.FloatTensor) - 0.01 * entropies[i]
                previous_value = values[i]

            total_loss = policy_loss +  value_loss

            status=""
            if (done):
                if info['flag_get']:
                    status = "reached end"
                else:
                    status = "dead"
            else:
                status = "alive"

            f = open(path+"\\{}\\ep_log.txt".format(button_as_string),"a")
            f.write("{},{},{},{},{},{} \n".format(experiment_num,episode,round(total_loss.item(),3),round(sum(rewards),3),info["x_pos"],status))
            f.close()

            A3C_optimiser.zero_grad()
           
            total_loss.backward(retain_graph=True)
            #update model
            BUTTON_PRESSED = False
            for local_param, global_param in zip(a3c_local_model.parameters(),a3c_shared.parameters()):
                if global_param.grad is not None:
                    break
                global_param._grad = local_param.grad

            A3C_optimiser.step()
        episode +=1
    
        if episode == no_episodes+1:
            print("Training finished")
            if save:
                end_time = timeit.default_timer()
                print("code runs for %.2f s " % (end_time-start_time))
            return 
