# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

note, the programs were done on python 3 soit must be run as follows:
train:
  python3 train.py 
      arguments:
      	  -h: display help
          --data_dir: path to the folder with the "train", "valid" and "test" folders
          --save_dir: path to save the checkpoint.pth default=checkpoint.pth
          --arch: type of selected architecture. Available models are 'densenet121', 'vgg16' and 'resnet50'
          --learning_rate: selected learning rate. default=0.003
          --dropout: selected dropout rate. default=0.2
          --hidden_units: number of units for the first hidden layer (int values only). default=512
          --epochs: Number of training epochs selected (int values only). default=5
          --gpu: If True, the system will attempt to run on gpu if one is available, otherwise it will only attempt to run on cpu (just boolean values 'True' or 'False'). default=True

      outputs:
          Training results plots
          Saved checkpoint
---
predict:
  python3 predict.py
      Arguments:
      	  -h: display help
          --input_image: path to the input image, if none selected would select a random one from the ./flower_data/valid/folder". default=RANDOM
          --data_dir: path to the folder with the "train", "valid" and "test" folders
          --checkpath: path to the checkpoint file. default=./checkpoint.pth
          --topk: number of top classess description to display (just int values). default=5
          --catnamepath: Dictionary for category to name conversion path. default=cat_to_name.json
          --gpu: If True, the system will attempt to run on gpu if one is available, otherwise it will only attempt to run on cpu (just boolean values 'True' or 'False'). default=True

      Outputs: 
          Prediction plots display

