import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import math
from tensorboardX import SummaryWriter
import time
from torch.autograd import Variable
from torchvision.transforms import transforms
from torchvision import utils
import src.MarioData as md
from torch.utils.data import DataLoader
import os
from torch.utils.data.sampler import SubsetRandomSampler
from skimage import img_as_float
import matplotlib.pyplot as plt
import torchvision
class CAE(torch.nn.Module):

    def __init__(self):
        super(CAE,self).__init__()

        self.conv1 = nn.Conv2d(4,16,3,padding = 1)

        self.conv2 = nn.Conv2d(16,9,3, padding = 1)

        self.t_conv1 = nn.ConvTranspose2d(9,16,1)
        
        self.t_conv2 = nn.ConvTranspose2d(16,4,1)



    def forward(self,x):
        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        x = F.relu(self.t_conv1(x))
        x = F.sigmoid(self.t_conv2(x))


        return x

    
    @staticmethod
    def createLossAndOptimiser(net, learning_rate =0.001):
        loss=torch.nn.MSELoss()

        optimiser = optim.Adam(net.parameters(),lr = learning_rate)
        return loss,optimiser

def trainNet(net,train_loader,val_loader,n_epochs,learning_rate,device):

    sum_path = 'summary/CAE'
    writer = SummaryWriter(sum_path)

    n_batches = len(train_loader)

    loss,optimiser = net.createLossAndOptimiser(net,learning_rate)


    print("Optimiser's state dict")
    for var_name in optimiser.state_dict():
        print(var_name,"\t",optimiser.state_dict()[var_name])

    training_start_time = time.time()


    t_losses=[]
    v_losses=[]

    for epoch in range(n_epochs):
        running_loss = 0.0
        print_every = n_batches //10
        start_time = time.time()
        total_train_loss = 0

        print('length of train loader: {}'.format(len(train_loader)))

        for i,data in enumerate(train_loader,0):
            inputs = data['image'].to(device)


            input_list = []

            for image in inputs:
                input_list.append(image)
            stack_inputs = torch.stack(input_list)
            stacked_inputs = stack_inputs.view(1,4,84,84)

            inputs = Variable(inputs)

            optimiser.zero_grad()
            outputs = net(stacked_inputs)
            

            loss_size = loss(outputs,stacked_inputs)

            loss_size.backward()
            optimiser.step()

            running_loss+=loss_size.data.item()
            total_train_loss +=loss_size.data.item()

            if (i +1) % (print_every + 1) ==0:
                print(i)
                
                current_step_per = int(100 * (i+1) /n_batches)
                writer.add_scalar("Cae train loss",running_loss/print_every,current_step_per)
                print("Epoch [{}/{}] {:d}%, Step[{}/{}],Loss: {:.4f},Accuracy: {:.2f}% took: {:.2f}s"
                .format(epoch + 1,n_epochs,current_step_per,i+1,len(train_loader),running_loss/print_every,0,time.time() -start_time))

                t_losses.append(running_loss / print_every)

                running_loss = 0.0
                start_time = time.time()
        
        total_val_loss = 0
        correct = 0
        total = 0

        for i, data in enumerate(val_loader,0):
            inputs = data['image'].to(device)
            input_list = []

            for im in inputs:
                input_list.append(im)

            stack_inputs = torch.stack(input_list)
            stacked_inputs = stack_inputs.view(1,4,84,84)

            inputs = Variable(inputs)

            val_outputs = net(stacked_inputs)
            val_loss_size = loss(val_outputs,stacked_inputs)
            total_val_loss += val_loss_size.data.item()
            current_step_per = int(100 * (i+1)/n_batches)

            if (i+1) % (print_every +1) ==0:
                writer.add_scalar('CAE val loss', val_loss_size.data.item(),current_step_per)
                v_losses.append(val_loss_size.data.item())
        
        print("Validation loss = {:.2f}; Accuracy: {:.2f}%".format(total_val_loss / len(val_loader),0))
    
    print("t loss")
    print(t_losses)

    print('v loss')
    print(v_losses)
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))


def show_images(images):
    
    grid = utils.make_grid(images)
    plt.imshow(img_as_float(grid.cpu().numpy().transpose((1,2,0))))
    plt.axis('off')
    plt.ioff()
    plt.show()


def testNet(net, test_loader, device):

    with torch.no_grad():
        correct = 0
        total = 0
        for i, data in enumerate(test_loader,0):
            inputs= data['image'].to(device)

            input_list=[]


            for im in inputs:
                input_list.append(im)

            cat_inputs1 = torch.stack(input_list)
            cat_inputs2 = cat_inputs1.view(1,4,84,84)

            inputs = Variable(inputs)

            outputs = net(cat_inputs2)

            outputs = outputs.view(4,1,84,84)
            final_inputs = cat_inputs2
            final_outputs = outputs
        
            show_images(cat_inputs1)

            show_images(final_outputs)

def trainCAE(save_weights, save_model):

    transform2apply = transforms.Compose([
        transforms.Resize((84,84)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    dataset = md.DatasetMario(os.getcwd()+"/",csv_name="allitems.csv",transform_in = transform2apply)

    val_split =0.2
    shuffle = True
    random_seed = 42
    use_cuda=True

    dataset_size = len(dataset)
    n_test = int(dataset_size*0.05)
    n_train = int(dataset_size - 2 * n_test)

    indices = list(range(dataset_size))

    np.random.shuffle(indices)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:(n_train + n_test)]
    test_indices = indices[(n_train + n_test):]

    train_samp = SubsetRandomSampler(train_indices)
    val_samp = SubsetRandomSampler(val_indices)
    test_samp = SubsetRandomSampler(test_indices)


    print("train size: {}".format(len(train_samp)))
    print("test size: {}".format(len(test_samp)))
    print("val size: {}".format(len(val_samp)))

    train_loader = DataLoader(dataset,batch_size =4, sampler = train_samp, num_workers =3, drop_last = True)
    val_loader = DataLoader(dataset,batch_size =4, sampler = val_samp, num_workers =3, drop_last = True)
    test_loader = DataLoader(dataset,batch_size =4, sampler = test_samp, num_workers =3, drop_last = True)

    device = torch.device("cuda"if (torch.cuda.is_available())else "cpu")
    print("cuda available: {}".format(torch.cuda.is_available()))
    CAE_model = CAE().to(device)
    print("Training model")
    print(CAE_model)


    if not os.path.exists(save_weights) and not os.path.exists(save_model):
        trainNet(CAE_model, train_loader,val_loader,n_epochs=1,learning_rate=0.001, device = device)
        torch.save(CAE_model.state_dict,save_weights)
        torch.save(CAE_model,save_model)

    #trainNet(CAE_model, train_loader,val_loader,n_epochs=1,learning_rate=0.001, device = device)
    testNet(CAE_model, test_loader,device =device)

    print("done!")


if __name__ == "__main__":

    trained_cae_dir = 'pretrained/trained_cae'
    trained_cae_weights = '/trained_cae_weights12-gray.pth'
    trained_cae_full_model = '/trained_cae_model12-gray.pth'

    if not os.path.isdir(trained_cae_dir):
        os.makedirs(trained_cae_dir)

    trainCAE((trained_cae_dir+trained_cae_weights),(trained_cae_dir+trained_cae_full_model))

