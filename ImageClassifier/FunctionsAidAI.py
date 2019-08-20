'''
    This file contains the functions to aid in the training,
    predicting and printing of the project "Developing and AI application
'''

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
#--------------------------------------------------------------------#

#---------------------------------------------------------------------#
#network structure setup
def Netset(architecture,dropout,hidden_layer1,learn_rate,outputs,gpu):
    '''
    Function to assemble and define the network structure
    Args:{architecture: accepts either 'densenet121', 'vgg16' or 'resnet50'
          dropout: chosen dropout rate
          hidden_layer1: number of elements in hidden layer 1
          learn_rate: learning rate chosen
          }
    Returns:{model: model set up and loaded (if gpu available, exported to gpu)
            criterion: defined criterion used is NLLLoss
            optimizer: Adam optimizer with input learnrate
          
    '''
    if architecture=='densenet121':
        model=models.densenet121(pretrained=True)
        infeats=1024
    elif architecture=='vgg16':
        model=models.vgg16(pretrained=True)
        infeats=25088
    elif architecture=='resnet50':
        model=models.resnet50(pretrained=True)
        infeats=2048
    else:
        print("Your model selection {} is not currently available".format(architecture))
        print("the available architectures are: densenet121,vgg16 and resnet50")
        
    for param in model.parameters():
        classifier = nn.Sequential(OrderedDict([
                                              ('dropout',nn.Dropout(dropout)),
                                              ('fc1', nn.Linear(infeats, hidden_layer1)),
                                              ('relu', nn.ReLU()),
                                              ('fc2', nn.Linear(hidden_layer1, outputs)),
                                              ('output', nn.LogSoftmax(dim=1))
                                              ]))
    
    model.classifier=classifier
    criterion=nn.NLLLoss()
    optimizer=optim.Adam(model.classifier.parameters(),lr=learn_rate)
    model.to(gpucheck(gpu))
    
    return model,criterion,optimizer,infeats
    
#---------------------------------------------------------------------#

#---------------------------------------------------------------------#
#GPU availability checking and setting:
def gpucheck(gpu):
    '''
    Function to check GPU availability and set device
    Args: gpu
    Returns: device
    '''
    if gpu==True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device="cpu" 
    return device
#---------------------------------------------------------------------#

#---------------------------------------------------------------------#
#Data loader
def data_loading_fun(path):
    '''
        Function to load and preprocess the data from the subfolders in 
        the father folder at path
        Args: data father folder path
        Returns: train loader, validation loader, test loader and number of categories
    '''
    data_dir = path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #Define your transforms for the training, validation, and testing sets

    # Transformations for train datasets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    # Transformations for test and validation datasets
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder

    train_data=datasets.ImageFolder(train_dir, transform=train_transforms)  #Train Dataset
    test_data=datasets.ImageFolder(test_dir,transform=test_transforms) #Test dataset
    valid_data=datasets.ImageFolder(valid_dir,transform=test_transforms) #validation dataset


    # Using the image datasets and the tranforms, define the dataloaders
    trainloader=torch.utils.data.DataLoader(train_data, batch_size=64,shuffle=True)
    testloader=torch.utils.data.DataLoader(test_data,batch_size=64)
    validloader=torch.utils.data.DataLoader(valid_data,batch_size=64)
    
    # Get the number of categories
    numcats=OutNums(train_dir)
    return trainloader,testloader,validloader,numcats
#---------------------------------------------------------------------#

#---------------------------------------------------------------------#
#Number of categories for outputs
def OutNums(train_dir):
    '''
    Function to determine number of outputs
    Args: train_dir: path to the training directories
    Returns: number of categories in training set
    '''
    numcats=len(os.listdir(train_dir))
    return numcats
#---------------------------------------------------------------------#

#---------------------------------------------------------------------#
#Checkpoint save
def checksave(pathsave,infeats,hidden_layer1,architecture,outputs,dropout,learn_rate,path,model,epoch,optimizer):
    '''
    Checkpoint saving in selected path
    Args: {path: address to save the file
           infeats: inputs to the system
           hidden_layer1: number of elements in the first hidden layer
           architecture: accepts either 'densenet121', 'vgg16' or 'resnet50'
           outputs: number of categories
           dropout: chosen dropout
           learn_rate: selected learning rate
    Returns: None
    '''
    data_dir=path
    train_dir = data_dir + '/train'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    model.class_to_idx= datasets.ImageFolder(train_dir,transform=train_transforms).class_to_idx
    classifier = nn.Sequential(OrderedDict([
                                              ('dropout',nn.Dropout(dropout)),
                                              ('fc1', nn.Linear(infeats, hidden_layer1)),
                                              ('relu', nn.ReLU()),
                                              ('fc2', nn.Linear(hidden_layer1, outputs)),
                                              ('output', nn.LogSoftmax(dim=1))
                                              ]))
    
    checkpoint={'architecture':architecture,
                'input_size': infeats,
                'hidden_layer1':hidden_layer1,
                'output_size': OutNums(train_dir),
                'dropout':dropout,
                'lr':learn_rate,
                'epoch':epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'state_dict':model.state_dict(),
                'class_to_idx': model.class_to_idx,
                'classifier':classifier}
    # Checkpoint saving on checkpoint.pth file at defined location
    torch.save(checkpoint, pathsave)
                
#---------------------------------------------------------------------#

#---------------------------------------------------------------------#
def checkload(path):
    '''
    Checkpoint loading from selected path file
    Args: path to the checkpoint file
    Returns:{model: neural network with the hyperparameters, 
            optimizer: loaded optimizer
    '''
    if torch.cuda.is_available():
        checkpoint = torch.load(path)
        gpu=True
    else:
        checkpoint=torch.load(path, map_location=lambda storage, loc: storage)
        gpu=False
         
    
        
    architecture=checkpoint['architecture']
    dropout=checkpoint['dropout']
    hidden_layer1=checkpoint['hidden_layer1']
    learn_rate=checkpoint['lr']
    outputs=checkpoint['output_size']
    
    modelnew,criterion,optimizernew,infeats =Netset(architecture,dropout,hidden_layer1,learn_rate,outputs,gpu)
        
    modelnew.classifier = checkpoint['classifier']
    modelnew.load_state_dict(checkpoint['state_dict'])
    modelnew.epochs=checkpoint['epoch']
    modelnew.class_to_idx=checkpoint['class_to_idx']
    optimizernew.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return modelnew,optimizernew
#---------------------------------------------------------------------#

#---------------------------------------------------------------------#
#PIL image processor
def process_image3(image):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model
    Args: image path
    Returns: img: an Numpy array
    '''
    # Open and load image
    image = Image.open(image)
    image.load()
    
    PILImgTransforms=transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224)])
    PILimg=PILImgTransforms(image)
    
    #Normalizing
    img = np.array(PILimg)/255
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225]) 
    img = (img - mean)/std
    
    img = img.transpose((2, 0, 1))
    return img
#---------------------------------------------------------------------#

#---------------------------------------------------------------------#
# Display an image along with the top 5 classes
def img_viewing(image_path,probs,classes,names_cats):
    ''' 
    Function to view the image, predicted classes and names
    Args:{image_path: path to the testing image
          probs: predicted probabilities
          classes: predicted classes
          
    '''
    figure=Image.open(image_path)
    plt.rcParams["figure.figsize"] = (15,10)
    
    plt.subplot(2,1,1)
    plt.imshow(figure)
    plt.title("Original Image",loc='center')
    
    plt.subplot(2,1,2)
    plt.title("Top Predictions",loc='center')
    plt.xlabel("Probability")
    plt.barh(names_cats,probs)
    

    
    plt.show()

#---------------------------------------------------------------------#

#---------------------------------------------------------------------#
#prediction
def predict(image_path, model,topk,catnamepath,gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    Args:{image_path: path to the image to predict
          model: applied model
          topk: number of top categories classification
          catnamepath:path to the json file with the categories names
    '''
    
    # TODO: Implement the code to predict the class from an image file
    #put the model in evaluation mode to drop dropout
    model.eval()
    
    #export to GPU  If no GPU available use model.cpu()
    device=gpucheck(gpu)
    
    #if torch.cuda.is_available():
     #   device="cuda"
    #else:
     #   device="cpu"
    
    model.to(device)
    #print(model)
    #print("is model in cuda?",next(model.parameters()).is_cuda)
    #image processing
    img=process_image3(image_path)
    
    #transform the 2D image into a 1D vector
    img=np.expand_dims(img, 0)  # alternative img.unsqueeze_(0)
    
    #Transfer image for input into GPU or CPU
    img=torch.from_numpy(img).float()
    inputImg=Variable(img).to(device)
    #print(img)
    
    #run the image through the model
    logpsPred=model.forward(inputImg)
    ps = torch.exp(logpsPred)
    probs,labels=ps.topk(topk,dim=1)
    #print(probs)
    probsclone=probs.clone().to("cpu")
    #top_probs=probsclone.numpy().flatten()
    
    print("labels in GPU?: ",labels.is_cuda)
    print("original labels: ",labels)
    clonelabels=labels.clone().to("cpu")
    print("Cloned labels in GPU?: ",clonelabels.is_cuda)
    print("Cloned labels: ",clonelabels)
    top_labels=clonelabels.numpy().flatten()
    print("top Labels index: ",top_labels)    
    
    with open(catnamepath, 'r') as f:
        cat_to_name = json.load(f)
    
    flower_names=[]
    for i in range(len(top_labels)):
        flower_names.append(cat_to_name[str(top_labels[i])])
    #Extract names of the flowers from json dictionary
    #flower_name=[model.class_to_idx.data[lab] for lab in range top_labels]
    return probs,top_labels,flower_names


#---------------------------------------------------------------------#
#image selector
def IMGSelector(patha):
    ''' 
    Selects and process random image for prediction when none selected
    Args:{patha: path to the valid folder
          
          proc_image:processed image
    '''
    pathdir=patha+random.choice(os.listdir(patha))   #Select random subfolder
    files = os.listdir(pathdir)   
    index = random.randrange(0, len(files))  #file index
    
    ImageTest=pathdir+'/'+files[index]
    proc_image=process_image3(ImageTest)
    
    return proc_image,ImageTest
