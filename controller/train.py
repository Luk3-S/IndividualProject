
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
from controller.controller_a3c import Actor_Critic
from torch.distributions import Categorical

import timeit
import numpy as np
from torch.autograd import Variable
right_only = [['right'],['A'],['right','A'],['right','A','B']]


def train (A3C_optimiser, a3c_shared,CAE_shared_model,test_name,experiment_num):
  

    no_steps = 100
    max_steps = 1600 
    no_episodes = 2000
    test_name = test_name

    start_time = timeit.default_timer()
    env, num_states,num_actions = instantiate_environment()
    


    a3c_local_model = Actor_Critic(num_states,num_actions).cuda()
    cae_local_model = CAE().cuda()
    a3c_local_model.train()

    torch.manual_seed(123)  
    np.random.seed(123)

    state = torch.from_numpy(env.reset()).cuda()

    ## make autoencoder 

    step = 0
    episode = 0
    done = True
    save = True

    f = open(path+"\\{}\\ep_log.txt".format("controller"),"a")
    f.write("==================================================== \n {} \n ==========================================\n".format(test_name))
    f.write("Episode, loss,rewards,xpos,status\n")
    f.close()
    while True:
        print("episode: {}".format(episode))
        if save == True:
            if episode %100 ==0 : 
                print("saved")
                torch.save(a3c_shared.state_dict(),"{}\\controller\\{}_controller_A3C_enc2".format(path,test_name))


        a3c_local_model.load_state_dict(a3c_shared.state_dict())

        try:
            cae_local_model.load_state_dict(torch.load("{}\\cae_model_enc2".format(path+"\\trained_cae"),map_location='cpu'))
        except:
            print("no file found")
        cae_local_model.eval()


        if done:
            hx = torch.zeros((1,512),dtype=torch.float)
            cx = torch.zeros((1,512),dtype=torch.float)
        else:
            hx = hx.detach()
            cx = cx.detach()
        hx=hx.cuda()
        cx=hx.cuda()

        log_policies =[]
        values=[]

        rewards=[]
        entropies=[]

        for _ in range(no_steps): 
            step +=1

            for param in cae_local_model.parameters():
                param.requires_grad = False
            
            ## generate cae's representation of the environment
            output_cae = cae_local_model(state)

            logits,value,hx,cx = a3c_local_model(output_cae,hx,cx)
           

            policy = F.softmax(logits,dim=1)
           
            log_policy = F.log_softmax(logits,dim=1)

            entropy = -(policy * log_policy).sum(1,keepdim=True)

            m = Categorical(policy)
            action = m.sample().item()

            
            state,reward,done,info = env.step(action)
            state = torch.from_numpy(state).cuda()      

            if step > max_steps or episode > no_episodes:
                done = True

            if done:
                step = 0
                state = torch.from_numpy(env.reset()).cuda()

            ## only do the following if we have our button pressed? i.e only reward when button 
            values.append(value)
            log_policies.append(log_policy[0,action])
            rewards.append(reward)
            entropies.append(entropy)
           ## BUTTON_PRESSED = False

            if done: 
            #    print("episode finished 1")
                break
            
    
            R = torch.zeros((1,1),dtype=torch.float).cuda()

            if not done:
                output = cae_local_model(state)
                _,R,_,_ = a3c_local_model(output,hx,cx)

            gae = torch.zeros((1,1),dtype=torch.float).cuda()

        
       
        policy_loss, value_loss = 0,0
   
        advantage_estimator2 = gae
        previous_value = R
       
        


        for i in range(len(rewards))[::-1]:
            
            # compute value loss: VL: L = sum(R - V(s))^2
            R = 0.9 * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + (advantage**2)/2

            ## advantage function estimate: reward[i] + gamma (value(st+1) - value(st))
            theta = rewards[i] + (0.9 * previous_value - values[i])
            advantage_estimator2 = advantage_estimator2 * 0.9  + theta
        
            # comput policy loss: -log(pi(s)) * A(s) - beta*H(pi)
            policy_loss = policy_loss - log_policies[i] *Variable(advantage_estimator2).type(torch.cuda.FloatTensor) - 0.01 * entropies[i]
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

        f = open(path+"\\{}\\ep_log.txt".format("controller"),"a")
        f.write("{},{},{},{},{},{} \n".format(experiment_num,episode,round(total_loss.item(),3),round(sum(rewards),3),info["x_pos"],status))

        f.close()
        A3C_optimiser.zero_grad()
        total_loss.backward(retain_graph=True)


        for local, shared in zip(a3c_local_model.parameters(),a3c_shared.parameters()):
            if shared.grad is not None:
                break
            shared._grad = local.grad

        A3C_optimiser.step()
        episode +=1
    
        if episode == no_episodes+1:
            print("Training process terminated")
            if save:
                end_time = timeit.default_timer()
                print("code runs for %.2f s " % (end_time-start_time))
            return

        

        
    


