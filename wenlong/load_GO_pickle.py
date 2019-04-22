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

