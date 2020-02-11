import torch
import torch.nn as nn


class lstm(nn.Module):
    def __init__(self,num_inputs, num_actions):
        super(lstm,self).__init__()

        