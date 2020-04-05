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
        
        self.lstm = nn.LSTMCell(1152, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)
 
    
    def forward(self, x, hx, cx):
     
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

      

        hx, cx = self.lstm(x.view(x.size(0), -1), (hx, cx))
    
        x = hx
        
        return self.actor_linear(x), self.critic_linear(x), hx, cx
