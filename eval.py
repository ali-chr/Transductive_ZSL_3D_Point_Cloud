from __future__ import print_function
import torch
import torch.nn.parallel
import torch.utils.data
import numpy as np
import argparse
import scipy.io
import torch.optim as optim
from src.models import *
from src.loss import *
from src.models import *
from src.util import *
from src.datautil import DataUtil
import yaml

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
Arguments
"""
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ModelNet', choices=['ModelNet', 'ScanObjectNN', 'McGill'], help='name of dataset i.e. ModelNet, ScanObjectNN, McGill')
parser.add_argument('--backbone', type=str, default='PointConv', choices=['EdgeConv', 'PointAugment', 'PointConv', 'PointNet'], help='name of backbone i.e. EdgeConv, PointAugment, PointConv, PointNet')
parser.add_argument('--method', type=str, default='ours', choices=['ours', 'baseline'], help='name of method i.e. ours, baseline')
parser.add_argument('--settings', type=str, default='inductive', choices=['inductive', 'transductive'], help='name of settings i.e. inductive, transductive')
parser.add_argument('--config_path', type=str, required=True, help='configuration path')
parser.add_argument('--model_path', type=str, required=True, help='model path')
args = parser.parse_args()

feature_dim = 2048 if args.backbone == 'EdgeConv' else 1024

config_file = open(args.config_path, 'r')
config = yaml.load(config_file, Loader=yaml.FullLoader)
print(config)

model = S2F(feature_dim)
if args.method=='baseline' and args.method=='inductive':
    model = F2S(feature_dim)

model.to(device)
model.load_state_dict(torch.load(args.model_path))

data_util = DataUtil(dataset=args.dataset, backbone=args.backbone, config=config)
data = data_util.get_data()

result = calculate_accuracy_ours(model, data, config)
print(" ZSL: acc=", result['zsl_acc'],", GZSL: acc_S=",result['gzsl_seen'], ", acc_U=", result['gzsl_unseen'],", HM=",result['gzsl_hm'])