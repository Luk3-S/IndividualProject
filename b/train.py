
import sys  
sys.path.append("C:\\Users\\UKGC-PC\\Documents\\Level 4 Project")

import torch
import torch.nn as nn
import torch.nn.functional as F
import gym_super_mario_bros
import gym
import actorcritic
import torch.multiprocessing as _mp
from src.environment import create_env
from src.convolutional_ae import CAE
from actorcritic import Actor_Critic
from torch.distributions import Categorical
from tensorboardX import SummaryWriter
import timeit



def train (index, A3C_optimiser, A3C_shared_model,CAE_shared_model,CAE_optimiser,save=True,new_stage=False):

    button = 'B'
    if save:
        start_time = timeit.default_timer()
    env, num_states,num_actions = create_env(1,1,button,100)
    #env2 = gym.wrappers.FlattenDictWrapper(env,dict_keys=['observation','desired_goal'])


    a3c_local_model = Actor_Critic(num_states,num_actions)

    

    #return
    cae_local_model = CAE()
    a3c_local_model.train()

    torch.manual_seed(123)

    state = torch.from_numpy(env.reset())
    ## make autoencoder 

    step = 0
    episode = 0
    done = True

    while True:
        print("episode: {}".format(episode))
        if save == True:
            if episode % 500 == 0 and episode > 0 :
                torch.save(CAE_shared_model.state_dict(),"{}/CAE_super_mario_bros_{}_{}enc1".format("/home/luke/Documents/Metal-Mario/saved_models",1,1))
                torch.save(a3c_local_model.state_dict(),"{}/A3C_super_mario_bros_{}_{}enc".format("/home/luke/Documents/Metal-Mario/saved_models",1,1))
            print("process {}. Episode{}".format(index, episode))
        episode +=1


        a3c_local_model.load_state_dict(A3C_shared_model.state_dict())

        #cae_local_model.load_state_dict(torch.load("{}/CAE_super_mario_bros_1_1_enc2".format("/home/luke/Documents/Metal-Mario/saved_models"),map_location='cpu'))

        cae_local_model.eval()


        if done:
            hx = torch.zeros((1,512),dtype=torch.float)
            cx = torch.zeros((1,512),dtype=torch.float)
        else:
            hx = hx.detach()
            cx = cx.detach()

        log_policies =[]
        values=[]

        rewards=[]
        entropies=[]

        for _ in range(20):
            print("step: {}".format(step))
            step +=1

            for param in cae_local_model.parameters():
                param.requires_grad = False
            
           
            output_cae = cae_local_model(state)

            logits,value,hx,cx = a3c_local_model(output_cae,hx,cx)

            policy = F.softmax(logits,dim=1)
            log_policy = F.log_softmax(logits,dim=1)

            entropy = -(policy * log_policy).sum(1,keepdim=True)

            m = Categorical(policy)
            action = m.sample().item()

            state,reward,done,_ = env.step(action)

            state = torch.from_numpy(state)

            if step > 500:
                done = True

            if done:
                step = 0
                state = torch.from_numpy(env.reset())
            
            values.append(value)
            log_policies.append(log_policy[0,action])
            rewards.append(reward)
            entropies.append(entropy)

            if done: 
                break
        
        R = torch.zeros((1,1),dtype=torch.float)

        if not done:
            output = cae_local_model(state)
            _,R,_,_ = a3c_local_model(output,hx,cx)

        gae = torch.zeros((1,1),dtype=torch.float)

        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        next_value = R

        for value, log_policy, reward, entropy in list(zip(values,log_policies,rewards,entropies))[::-1]:
            gae = gae * 0.1 * 0.2 # gamma = 0.1 , tau = 0.2
            gae = gae + reward + 0.1 * next_value.detach() - value.detach()
            next_value = value
            actor_loss = actor_loss + log_policy * gae
            R = R * 0.1 + reward
            critic_loss = critic_loss + (R-value) **2 /2
            entropy_loss = entropy_loss + entropy
        
        total_loss = -actor_loss + critic_loss - 0.3 * entropy_loss # beta = 0.3

        print("index: {}".format(index))
        print("total_loss: {}".format(total_loss))
        print("episode: {}".format(episode))

        A3C_optimiser.zero_grad()
        total_loss.backward()

        #update model

        for local_param, global_param in zip(a3c_local_model.parameters(),A3C_shared_model.parameters()):
            if global_param.grad is not None:
                break
            global_param._grad = local_param.grad

        A3C_optimiser.step()

        if episode == int(20):
            print("Training process {} terminated".format(index))
            if save:
                end_time = timeit.default_timer()
                print("code runs for %.2f s " % (end_time-start_time))
            return

        

        
    


