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
#argument parser
'''
    Args:{--input_image: path to the input image, if none selected would select a random one from the ./flower_data/valid/folder". default=RANDOM
          --data_dir: path to the folder with the "train", "valid" and "test" folders
          --checkpath: path to the checkpoint file. default=./checkpoint.pth
          --topk: number of top classess description to display (just int values). default=5
          --catnamepath: Dictionary for category to name conversion path. default=cat_to_name.json
          --gpu: If True, the system will attempt to run on gpu if one is available, otherwise it will only attempt to run on cpu (just boolean values 'True' or 'False'). default=True
      
      Outputs:{Prediction plots display
'''
parser=argparse.ArgumentParser(description='predict.py.- Loads the model from the save_path and uses the model to predict the classification of the input_image.')

parser.add_argument('--data_dir',help='path to folder with the "train", "valid" and "test" folders. default=./flower_data/',action="store",default="./flower_data/",dest="data_dir")

parser.add_argument('--input_image',default="RANDOM",dest="input_image",action="store",type=str,help='path to the input image, if none selected would select a random one from the ./flower_data/valid/ folder')

parser.add_argument('--checkpath',default="./checkpoint.pth",dest="checkpath",action="store",type=str,help='path to the checkpoint file. default=./checkpoint.pth')

parser.add_argument('--topk',default=5,dest="topk",action="store",type=int,help='number of top classess description to display (just int values). default=5')

parser.add_argument('--catnamepath',default='cat_to_name.json',dest="catnamepath",action="store",type=str,help='Dictionary for category to name conversion path. default=cat_to_name.json')

parser.add_argument('--gpu',dest="gpu",action="store",type=bool,default=True,help='If True, the system will attempt to run on gpu if one is available, otherwise it will only attempt to run on cpu (just boolean values True or False)')

#----------------------------------------------------------#
#arguments extraction
parsargs=parser.parse_args()

imagepath=parsargs.input_image
data_dir=parsargs.data_dir
checkpath=parsargs.checkpath
topk=parsargs.topk
catnamepath=parsargs.catnamepath
gpu=parsargs.gpu

#------------------------------------------------------------#
#Data loaders
#trainloader,testloader,validloader,numcats=FAID.data_loading_fun(data_dir)

#------------------------------------------------------------#
#Image Loader (chooses random image when none selected
if imagepath=='RANDOM':
    data_container=data_dir+'valid/'
    proc_image,ImageOrig=FAID.IMGSelector(data_container)
else:
    proc_image=FAID.process_image3(imagepath)
    ImageOrig=imagepath
    

#------------------------------------------------------------#
#checkpoint loading
model,optimizer,checkpoint=FAID.checkload(checkpath)

#------------------------------------------------------------#
#Categories names loader
with open(catnamepath, 'r') as f:
    cat_to_name = json.load(f)
    
#------------------------------------------------------------#
#Probabilities, top labels and categories names recovery 

probs,classes,names = FAID.predict(ImageOrig, model,topk,catnamepath,gpu,checkpoint)
probscpu=probs.detach().to("cpu").numpy().flatten()
print("probabilities: ",probscpu)
print("classes: ",classes)
print("labels: ", names)

for i in range(topk):
    print(f"top {i+1} prediction {names[i]} with a probability of {probscpu[i]}")

#------------------------------------------------------------#
#Categories image and result viewing

FAID.img_viewing(ImageOrig,probscpu,classes,names)

#------------------------------------------------------------#
#End of program
print("Run completed")
