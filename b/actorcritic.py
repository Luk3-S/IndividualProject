import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import math

class Actor_Critic(torch.nn.Module):
    def __init__(self,num_inputs,num_actions):
        super(Actor_Critic,self).__init__()

        self.conv1 = nn.Conv2d(9, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm = nn.LSTMCell(14112,512)
        self.critic_linear = nn.Linear(512,1)
        self.actor_linear = nn.Linear(512,num_actions)

        self._initialise_weights()

    def _initialise_weights(self):
        for module in self.modules():
            if isinstance(module,nn.Conv2d)or isinstance(module,nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih,0)
                nn.init.constant_(module.bias_hh,0)

    def forward(self,x,hx,cx):

        #add hidden layers using relu
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        

        hx, cx = self.lstm(x.view(x.size(0),-1),(hx,cx))
        
        print("reached forward - cae")
        return self.actor_linear(hx), self.critic_linear(hx), hx, cx
