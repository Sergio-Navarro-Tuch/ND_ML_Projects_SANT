#------------------------------------------------------------------------#
# Imports here

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'  #Define figures quality as retina
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt    #Model for plots generation

import time   #module for timing the progress and actions (used to optimize timing in redesign)
import torch  #importing pytorch module
from torch import nn  #from Pytorch import Neural Networks
from torch import optim
import torchvision
from torchvision import datasets, transforms, models
import torchvision.transforms as transforms
from torch.autograd import Variable
from collections import OrderedDict

import json

import PIL
from PIL import Image

import argparse

import numpy as np
import os, random

import FunctionsAidAI as FAID  #import functions from aiding file
#--------------------------------------------------------------------#

#--------------------------------------------------------------------#
#argument parser
'''
    Args:{--data_dir: path to the folder with the "train", "valid" and "test" folders,
          --save_dir: path to save the checkpoint.pth default=checkpoint.pth
          --arch: type of selected architecture. Available models are 'densenet121', 'vgg16' and 'resnet50'
          --learning_rate: selected learning rate. default=0.003
          --dropout: selected dropout rate. default=0.2
          --hidden_units: number of units for the first hidden layer (int values only). default=512
          --epochs: Number of training epochs selected (int values only). default=5
          --gpu: If True, the system will attempt to run on gpu if one is available, otherwise it will only attempt to run on cpu (just boolean values 'True' or 'False'). default=True
      
      Outputs:{Training results plots
              Saved checkpoint
'''
parser=argparse.ArgumentParser(description='train.py trains for image recognition from pretrained architectures.')

parser.add_argument('--data_dir',help='path to folder with the "train", "valid" and "test" folders',action="store",default="./flower_data/",dest='data_dir')

parser.add_argument('--save_dir',dest="pathsave",action="store",default="./checkpoint.pth",help='Path to save the checkpoint.pth file for the model.')

parser.add_argument('--arch',dest="architecture",action="store",default="densenet121",help='type of selected architecture. Available models are densenet121, vgg16 and resnet50')

parser.add_argument('--learning_rate',action="store",dest="learn_rate",default=0.003,type=float,help='selected learning rate. default=0.003')

parser.add_argument('--dropout',action="store",dest="dropout",default=0.2,type=float,help='selected dropout rate. default=0.2')

parser.add_argument('--hidden_units',action="store",dest="hidden_layer1",default=512,type=int,help='number of units for the first hidden layer (int values only)')

parser.add_argument('--epochs',action="store",dest="epochs",default=5,type=int,help='Number of training epochs selected (int values only)')

parser.add_argument('--gpu',dest="gpu",action="store",type=bool,default=True,help='If True, the system will attempt to run on gpu if one is available, otherwise it will only attempt to run on cpu (just boolean values True or False)')

#----------------------------------------------------------#
#arguments extraction
parsargs=parser.parse_args()

data_path=parsargs.data_dir
architecture=parsargs.architecture
dropout=parsargs.dropout
hidden_layer1=parsargs.hidden_layer1
learn_rate=parsargs.learn_rate
epochs=parsargs.epochs
pathsave=parsargs.pathsave
gpu=parsargs.gpu

#------------------------------------------------------------#
#load data

trainloader,testloader,validloader,numcats=FAID.data_loading_fun(data_path)
print("Data succesfully loaded..")
print(f"A total of {numcats} categories found..")

#------------------------------------------------------------#
#model, optimizer and criterion acquisition

outputs=numcats
model,criterion,optimizer,infeats=FAID.Netset(architecture,dropout,hidden_layer1,learn_rate,outputs,gpu)

#-----------------------------------------------------------#
#Define training device
device=FAID.gpucheck(gpu)

#-----------------------------------------------------------#
#Network training

print("Training begin")
train_losses,test_losses,accuracy_registry,time_batch,time_epoch=[],[],[],[],[]
for epoch in range(epochs):
    steps = 0
    Valsteps=0
    running_loss = 0
    start_epoch=time.time()
    for inputs, labels in trainloader:
        steps += 1
        Valsteps+=1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        start = time.time()  #Time counter start
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        
        timestep=time.time()-start
        time_batch.append(time.time()-start) #Register time taken for the step

        running_loss += loss.item()
        
        
        
#---------------------------------------------------------------------------------------------#
#--------------------------[ Validation on test data ]------------------------------------#        
        if Valsteps==10 or steps==len(trainloader):
        #testing and measurement register for each batch
            test_loss = 0
            accuracy = 0
            model.eval()
            
            Valsteps=0  #Restart Valstep counter

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    train_losses.append(running_loss)
                    test_losses.append(test_loss/len(testloader))
                    accuracy_registry.append(accuracy/len(testloader))

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}.."
                  f"Batch time: {timestep:.3f}")
            
        print(f"Epoch {epoch+1}/{epochs}.. "
            f"Step: {steps}/{len(trainloader)}..")
        running_loss = 0
        model.train()
        time_epoch.append(time.time()-start_epoch)
#---------------------[End training]------------------------#

#-----------------------------------------------------------#
#Saving training results to file "train_results_plots.png"
# Results, loss and training time plotting 
plt.rcParams["figure.figsize"] = (15,10)

plt.subplot(2,2,1)
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Test loss')
plt.title("Training and Testing Loss Plot",loc='center')
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.legend(frameon=False)

plt.subplot(2,2,2)
plt.plot(accuracy_registry,label='Test Accuracy')
plt.title("Test Accuracy",loc='center')
plt.xlabel("Batch")
plt.ylabel("Accuracy [%]")
plt.legend(frameon=False)

plt.subplot(2,2,3)
plt.plot(time_batch,label='Batch time')
plt.title("Training batch time")
plt.xlabel("Batch number")
plt.ylabel("Time [s]")
plt.legend(frameon=False)

plt.subplot(2,2,4)
plt.plot(time_epoch,label='Epoch time')
plt.title("Training Epoch time")
plt.xlabel("Epoch*100")
plt.ylabel("Time[s]")
plt.legend(frameon=False)

plt.show()
plt.savefig('train_results_plots.png')
#-------------------[End train results plotting]------------#

#-----------------------------------------------------------#
#save checkpoint
outputs=numcats

FAID.checksave(pathsave,infeats,hidden_layer1,architecture,outputs,dropout,learn_rate,data_path,model,epochs,optimizer)

#-----------------------------------------------------------#
#End of program printing
print("Program completed")
print("Model trained and checkpointr saved to",pathsave)
