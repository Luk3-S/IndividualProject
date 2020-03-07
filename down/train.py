
import sys
laptop_path = "C:\\Users\\Luke\\Documents\\diss proj\\IndividualProject"
desktop_path = "C:\\Users\\UKGC-PC\\Documents\\metal-mario-master\\IndividualProject"
sys.path.append(desktop_path)
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym_super_mario_bros
import gym
import torch.multiprocessing as _mp
from src.environment import create_env
from src.convolutional_ae import CAE
from a.actorcritic import Actor_Critic
from torch.distributions import Categorical
from tensorboardX import SummaryWriter
import timeit


def train (index, A3C_optimiser, A3C_shared_model,CAE_shared_model,CAE_optimiser,save,button):
    
    BUTTON_PRESSED = False
    MOVEMENT_OPTIONS = [['right'], ['A'], ['left'], ['down'], ['up'],['B'],['right','A'],['right','A','B']]

    no_steps = 100
    max_steps = 1600 ## max steps possible in 400 seconds
    no_episodes = 1000
    if save:
        start_time = timeit.default_timer()
    env, num_states,num_actions = create_env(1,1)
    #env2 = gym.wrappers.FlattenDictWrapper(env,dict_keys=['observation','desired_goal'])


    a3c_local_model = Actor_Critic(num_states,num_actions)
    cae_local_model = CAE()
    a3c_local_model.train()

    torch.manual_seed(123)

    state = torch.from_numpy(env.reset())
    ## make autoencoder 

    step = 0
    episode = 0
    done = True
    save = True
    while True:
        print("episode: {}".format(episode))
        if save == True:
            if episode %100 ==0 : # 500 episode > 0 and episode % 100 ==0 
                print("saved")
                torch.save(CAE_shared_model.state_dict(),"{}\\CAE_super_mario_bros_{}_{}_enc1".format(desktop_path+"\\trained_models",1,1))
                torch.save(A3C_shared_model.state_dict(),"{}\\A3C_super_mario_bros_{}_{}_enc".format(desktop_path+"\\{}".format("down"),1,1))
                #C:\Users\UKGC-PC\Documents\Level 4 Project\trained_models
                
            #print("process {}. Episode{}".format(index, episode))


        a3c_local_model.load_state_dict(A3C_shared_model.state_dict())

        try:
            cae_local_model.load_state_dict(torch.load("{}\\CAE_super_mario_bros_1_1_enc1".format(desktop_path+"\\trained_models"),map_location='cpu'))
        except:
            print("no file found")
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

        for _ in range(no_steps): #500
            #print("step: {}".format(step))
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
            
            #print(MOVEMENT_OPTIONS[action],MOVEMENT_OPTIONS[action][0]==button[0])
            if (MOVEMENT_OPTIONS[action]==button or button[0] in MOVEMENT_OPTIONS[action]):
            #     #print(MOVEMENT_OPTIONS[action]) ## check if the behaviour that we're training is chosen
                BUTTON_PRESSED = True
            
            state,reward,done,_ = env.step(action)
            state = torch.from_numpy(state)      

            if step > max_steps or episode > no_episodes:
                done = True

            if done:
                step = 0
                state = torch.from_numpy(env.reset())

            #print("step: {}".format(step))
            if (BUTTON_PRESSED):
                #print("")
            ## only do the following if we have our button pressed? i.e only reward when button 
                values.append(value)
                log_policies.append(log_policy[0,action])
                rewards.append(reward)
                entropies.append(entropy)
                BUTTON_PRESSED = False

                if done: 
                    print("episode finished 1")
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
                #print("step count: {}".format(step))
                if done: 
                    #print("episode finished 2")
                    break
        
        if (len(values)>0):## check if we've actually pressed said button, and if so, then apply rewards etc.
            for value, log_policy, reward, entropy in list(zip(values,log_policies,rewards,entropies))[::-1]:
                #print("v:{}, lp:{}, r:{}, e:{} ".format(value,log_policy,reward,entropy))
                gae = gae * 0.9 * 1 # gamma = 0.1 , tau = 0.2
                gae = gae + reward + 0.9 * next_value.detach() - value.detach()
                next_value = value
                actor_loss = actor_loss + log_policy * gae
                R = R * 0.9 + reward
                critic_loss = critic_loss + (R-value) **2 /2
                entropy_loss = entropy_loss + entropy
            
            print(rewards)
            print("episode {} cumulative rewards: {}".format(episode,sum(rewards)))
            print("a: {}, c: {}, e: {}".format(actor_loss,critic_loss,entropy_loss))
            total_loss = -actor_loss + critic_loss - 0.01 * entropy_loss # beta = 0.3
            
            #print("index: {}".format(index))
            print("total_loss: {}".format(total_loss))
            #print("episode: {}".format(episode))
            print("\n")

            A3C_optimiser.zero_grad()
            # if total_loss == 0:
            #      total_loss = torch.tensor([[0.0]],requires_grad=True)
            total_loss.backward(retain_graph=True)
            #update model

            for local_param, global_param in zip(a3c_local_model.parameters(),A3C_shared_model.parameters()):
                if global_param.grad is not None:
                    break
                global_param._grad = local_param.grad

            A3C_optimiser.step()
        episode +=1
    
        if episode == no_episodes+1:
            print("Training process {} terminated".format(index))
            if save:
                end_time = timeit.default_timer()
                print("code runs for %.2f s " % (end_time-start_time))
            return

        

        
    


