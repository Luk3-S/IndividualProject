import os
import sys
laptop_path = "C:\\Users\\Luke\\Documents\\diss proj\\IndividualProject"
desktop_path = "C:\\Users\\UKGC-PC\\Documents\\metal-mario-master\\IndividualProject"
sys.path.append(desktop_path)
os.environ['OMP_NUM_THREADS'] = '1'
import torch
from a.actorcritic import Actor_Critic
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT,COMPLEX_MOVEMENT,RIGHT_ONLY
import gym_super_mario_bros
from environment import create_env
from convolutional_ae import CAE
import torch.nn.functional as F

MOVEMENT_OPTIONS = [['right'], ['A'], ['left'], ['down'], ['up'],['B'],['right','A'],['right','A','B']]


def run_test():
    torch.manual_seed(123)
   
    env, num_states, num_actions = create_env(1,1)#, args.action_type,"{}/video_{}_{}.mp4".format(args.output_path, args.world, args.stage))
    CAE_model = CAE()
    
    a3c_model = Actor_Critic(num_states, num_actions)

    if torch.cuda.is_available():
        a3c_model.load_state_dict(torch.load("{}\\A3C_super_mario_bros_{}_{}_enc".format(desktop_path+"\\lstm", 1, 1)))
        print("loaded")
        a3c_model.cuda()
        CAE_model.cuda()
    else:
        a3c_model.load_state_dict(torch.load("{}\\A3C_super_mario_bros_{}_{}_enc".format(desktop_path+"\\lstm", 1, 1),
                                         map_location=lambda storage, loc: storage))

    CAE_model.load_state_dict(torch.load("{}\\CAE_super_mario_bros_1_1_enc1".format(desktop_path+"\\trained_models"),map_location='cpu'))


    a3c_model.eval()
    CAE_model.eval()
    state = torch.from_numpy(env.reset())
    done = True
    while True:
        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)
            env.reset()
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
        action = int(action)
        #print(MOVEMENT_OPTIONS[action])
        state, _, done, info = env.step(action)
        state = torch.from_numpy(state)
        env.render()
        if info["flag_get"]:
            print("World {} stage {} completed".format(args.world, args.stage))
            break

run_test()