from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import pickle, gzip, os, sys, re
import random 
from random import shuffle
import numpy as np 
import pandas as pd

sys.path.append('/u/flashscratch/d/datduong/GOmultitask')
import helper

sys.path.append('/u/flashscratch/d/datduong/GOmultitask/process_data/go_term_object')
from go_term_object import *

PICKLE_PATH="/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/w2vDim300/EntDataJan19w300Base/GO_all_info_go_graph.pickle"

go_graph_pickle = open(PICKLE_PATH, "rb")
go_graph = pickle.load(go_graph_pickle)

print(go_graph)
