"""
Some of the links which I found useful.
https://discuss.pytorch.org/t/freeze-the-learnable-parameters-of-resnet-and-attach-it-to-a-new-network/949/9
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import os
import argparse

from tars.tars_data_loaders import *
from tars.tars_training import *
from tars.tars_model import *

parser = argparse.ArgumentParser(description='Training a pytorch model to classify different plants')
parser.add_argument('-idl', '--input_data_loc', help='', default='data/training_data')
parser.add_argument('-mo', '--model_name', default="resnet18")
parser.add_argument('-f', '--freeze_layers', default=True, action='store_false', help='Bool type')
parser.add_argument('-fi', '--freeze_initial_layers', default=True, action='store_false', help='Bool type')
parser.add_argument('-ep', '--epochs', default=100, type=int)
parser.add_argument('-b', '--batch_size', default=64, type=int)
parser.add_argument('-is', '--input_shape', default=224, type=int)
parser.add_argument('-sl', '--save_loc', default="models/" )
parser.add_argument("-g", '--use_gpu', default=False, action='store_false', help='Bool type gpu')
parser.add_argument("-p", '--use_parallel', default=True, action='store_false', help='Bool type to use_parallel')
parser.add_argument("-mx", '--mixup', default=True, action='store_true' ,help='Use mixup data augementation')
parser.add_argument("-mxal", '--mixup_alpha', default=0.1, type = float, help='Alpha to be used in mixup agumentation')

args = parser.parse_args()


dataloaders, dataset_sizes, class_names = generate_data(args.input_data_loc, args.input_shape, args.model_name, batch_size=args.batch_size)
print(class_names)

print("[Load the model...]")
# Parameters of newly constructed modules have requires_grad=True by default
print("Loading model using class: {}, use_gpu: {}, freeze_layers: {}, freeze_initial_layers: {}, name_of_model: {}".format(len(class_names), args.use_gpu, args.freeze_layers, args.freeze_initial_layers, args.model_name))
model_conv = all_pretrained_models(len(class_names), use_gpu=args.use_gpu, freeze_layers=args.freeze_layers, freeze_initial_layers= args.freeze_initial_layers, name=args.model_name)
if args.use_parallel:
    print("[Using all the available GPUs]")
    model_conv = nn.DataParallel(model_conv, device_ids=[0, 1])

print("[Using CrossEntropyLoss...]")
criterion = nn.CrossEntropyLoss()

print("[Using small learning rate with momentum...]")
optimizer_conv = optim.SGD(list(filter(lambda p: p.requires_grad, model_conv.parameters())), lr=0.001, momentum=0.9)

print("[Creating Learning rate scheduler...]")
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

print("[Training the model begun ....]")
print(args.mixup, args.mixup_alpha)
model_ft = train_model(model_conv, dataloaders, dataset_sizes, criterion, optimizer_conv, exp_lr_scheduler, args.use_gpu,
                       num_epochs=args.epochs, mixup = args.mixup, alpha = args.mixup_alpha)

print("[Save the best model]")
model_save_loc = args.save_loc+args.model_name+"_"+str(args.freeze_layers)+"_freeze"+"_"+str(args.freeze_initial_layers)+"_freeze_initial_layer"+".pth"
torch.save(model_ft.state_dict(), model_save_loc)
