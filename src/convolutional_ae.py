import sys

path ="C:\\Users\\UKGC-PC\\Documents\\metal-mario-master\\IndividualProject"
sys.path.append(path)
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import math
import time
from torch.autograd import Variable
from torchvision.transforms import transforms
from torchvision import utils
import src.MarioData as md
from torch.utils.data import DataLoader
import os
import torchvision
from random import shuffle, sample,choice
from PIL import Image

class CAE(torch.nn.Module):

    def __init__(self):
        super(CAE,self).__init__()
        ## encoder
        self.conv1 = nn.Conv2d(4,16,3,padding = 1)
        self.conv2 = nn.Conv2d(16,9,3, padding = 1)

        ## decoder 
        self.t_conv1 = nn.ConvTranspose2d(9,16,1)
        self.t_conv2 = nn.ConvTranspose2d(16,4,1)


  
    def forward(self,x):

        #encoder - add hidden layers using relu
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x)) 


        # decoder - add transpose layers
        #x = F.relu(self.transpose_layer_1(x))
        #x = F.relu(self.transose_layer_2(x))
        return x


def train_cae(cae,data_loader,learning_rate,device):

   

    training_batches = len(data_loader)

    loss = torch.nn.MSELoss()
    optimiser = optim.Adam(cae.parameters(),lr = learning_rate)
  
    

    start_time = time.time()


    losses=[]

    cae.train()
    for epoch in range(1):
        running_loss = 0.0
        save_frequency = training_batches //10
        start_time = time.time()
        total_train_loss = 0

        print('Number of images in dataset: {}'.format(len(data_loader)))

        for i,data in enumerate(data_loader,0):
            frames = data['image'].to(device)

            frames_to_stack = [f for f in frames]
            stack_inputs = torch.stack(frames_to_stack)
            ## environment returns 4 frames from the game environment at each step, therefore we stack these frames before completing the forward pass
            stacked_inputs = stack_inputs.view(1,4,84,84)

            # inputs = Variable(images)

            
            # do forward pass 
            outputs = cae(stacked_inputs)
            
            # compute the loss after output reconstructed by cae
            loss_size = loss(outputs,stacked_inputs)

            ## update model
            optimiser.zero_grad()
            loss_size.backward()
            optimiser.step()

            running_loss+=loss_size.data.item()
            total_train_loss +=loss_size.data.item()

            if (i %10 ) ==0:
                
                f = open(path+"\\{}\\cae_log.txt".format("trained_cae"),"a")
                f.write("Epoch: {} | Step: {} | Loss: {}\n".format(epoch,i,running_loss/save_frequency))
                
                
                losses.append(running_loss / save_frequency)

                running_loss = 0.0
                start_time = time.time()
    

  



    end_time = time.time()
    f = open(path+"\\{}\\cae_log.txt".format("trained_cae"),"a")
    f.write("Total Loss: {}\n Loss values: {} ".format(total_train_loss,losses))

    print("Training finished: took {} seconds.".format(end_time-start_time))

def start_train():

    transform2apply = transforms.Compose([
        transforms.Resize((84,84)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    dataset = md.DatasetMario(os.getcwd(),csv_name="allitems.csv",transform_in = transform2apply)

    
    
    ## shuffle indices from dataset and segment into training and test set
    dataset_length = len(dataset)
    indices_list = [i for i in range(dataset_length)]
    
    shuffle(indices_list)

    num_train = round(dataset_length *0.75)
    
    tests = indices_list[num_train:]
    
    trains = indices_list[:num_train]

    train_set = torch.utils.data.Subset(dataset,trains)
    test_set = torch.utils.data.Subset(dataset,tests)
    
    # pytorch datalaoder collects our dataset and provides interable, so we can iterate over (image,state) pairs
    train_loader = DataLoader(train_set,batch_size =4, drop_last = True)
    test_loader = DataLoader(test_set,batch_size =4, drop_last = True)

    device = torch.device("cuda")
    model = CAE().to(device)
    
    ## train first, and then evaluate with a test set

    f = open(path+"\\{}\\cae_log.txt".format("trained_cae"),"a")
    f.write("========================== Cae Loss Values on training set ==========================\n")
    f.close()
    
    train_cae(model, train_loader,learning_rate=0.001, device = device)


    f = open(path+"\\{}\\cae_log.txt".format("trained_cae"),"a")
    f.write("========================== Cae Loss Values on test set ==========================\n")
    f.close()
    train_cae(model,test_loader,learning_rate=0.001,device = device)
    # save model weights
    torch.save(model.state_dict,"{}\\cae_weights\\cae_model_enc2".format(path))
    print("Weights saved")
    


if __name__ == "__main__":
    start_train()
