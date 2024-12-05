import warnings
import os
from model_function import *
from model_utils import *
from utils import *
from torch.utils.data import DataLoader
import torch.nn.functional as Fin
import timeit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
from torchdiffeq import odeint as odeint
import matplotlib
matplotlib.use('Agg')
import argparse
import sys
import time
import torch
torch.manual_seed(42)
torch.cuda.empty_cache() 
import torch.optim as optim
import random
import logging
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)
import sys

set_seed(42)
cwd = os.getcwd()
#data_path = {'z500':str(cwd) + '/era5_data/geopotential_500/*.nc','t850':str(cwd) + '/era5_data/temperature_850/*.nc'}
SOLVERS = ["dopri8","dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams',"adaptive_heun","euler"]
parser = argparse.ArgumentParser('ClimODE')

parser.add_argument('--solver', type=str, default="euler", choices=SOLVERS)
parser.add_argument('--atol', type=float, default=5e-3)
parser.add_argument('--rtol', type=float, default=5e-3)
parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")
parser.add_argument('--niters', type=int, default=300)
parser.add_argument('--scale', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--spectral', type=int, default=0,choices=[0,1])
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--weight_decay', type=float, default=1e-5)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_time_scale= slice('2006','2016')
val_time_scale = slice('2016','2016')
test_time_scale = slice('2017','2018')

paths_to_data = [str(cwd) + '/era5_data/geopotential_500/*.nc',str(cwd) + '/era5_data/temperature_850/*.nc',str(cwd) + '/era5_data/2m_temperature/*.nc',str(cwd) + '/era5_data/10m_u_component_of_wind/*.nc',str(cwd) + '/era5_data/10m_v_component_of_wind/*.nc']

num_years = len(range(2006,2016))
model = Climate_encoder_free_uncertain(len(paths_to_data),2,out_types=len(paths_to_data),method=args.solver,use_att=True,use_err=True,use_pos=False).to(device)

from torchsummary import summary

input_shape = (3, 64, 64)  # Adjust this according to your actual input dimensions

# Print model summary
dummy_input = torch.randn(1, 3, 64, 64).to(device)
summary(model(T=None,data=None,atol=0.1,rtol=0.1), dummy_input.shape[1:])