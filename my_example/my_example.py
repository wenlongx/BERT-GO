


from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
import random, sys
from io import open

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

sys.path.append("/u/flashscratch/d/datduong/pytorch-pretrained-BERT")
import pytorch_pretrained_bert.modeling as modeling
import pytorch_pretrained_bert.tokenization 
import pytorch_pretrained_bert.optimization 

from torch.utils.data import Dataset
import random


input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

config = modeling.BertConfig(vocab_size_or_config_json_file=23408, hidden_size=300,num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072,max_position_embeddings=23408,vocab_path="/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/GO_db_complete_vocab.txt",init_emb_path="/u/flashscratch/d/datduong/w2vModel1Gram11Nov2017/w2vModel1Gram11Nov2017.txt")

model = modeling.BertModel(config=config)

all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)

