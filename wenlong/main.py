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

sys.path.append("/u/flashscratch/d/datduong/pytorch-pretrained-BERT")

import pytorch_pretrained_bert.modeling as modeling
import pytorch_pretrained_bert.tokenization 
import pytorch_pretrained_bert.optimization 

from torch.utils.data import Dataset
import random

PATH="/u/home/w/wenlongx/BERT/pytorch-pretrained-BERT"

# This represents a file that lists all of the possible words occurring in the GO dataset
VOCAB_PATH="/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/GO_db_complete_vocab.txt"

# This we should not change, because it represents the initial word embeddings that
# Dat has already pretrained on PubMed data
INIT_EMB_PATH="/u/flashscratch/d/datduong/w2vModel1Gram11Nov2017/w2vModel1Gram11Nov2017.txt"

# Pickle of the GO term objects
PICKLE_PATH="/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/w2vDim300/EntDataJan19w300Base/GO_all_info_go_graph.pickle"


input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

config = modeling.BertConfig(vocab_size_or_config_json_file=23408, 
        hidden_size=300,
        num_hidden_layers=12, 
        num_attention_heads=12, 
        intermediate_size=3072,
        max_position_embeddings=23408,
        vocab_path=VOCAB_PATH,
        init_emb_path=INIT_EMB_PATH)

model = modeling.BertModel(config=config)

all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)

