import torch
import torch.nn as nn
import argparse
import os
import sys
import pdb

import time
import random
import numpy as np
from transformers import AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

from models_new import Classifier
from models import BertForSequenceClassificationCustom
from dataset import DataSet
from train import Trainer
from train_ict import ICT_Trainer

from utils import set_seeds

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

parser = argparse.ArgumentParser(description='SSL for NLP')

parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--data_seed', default=42, type=int)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--do_lower_case', default=True, type=bool)

parser.add_argument('--train_batch_size', default=16, type=int)
parser.add_argument('--val_batch_size', default=16, type=int)
parser.add_argument('--results_dir', default='results')
parser.add_argument('--hide_tqdm', action='store_true')
parser.add_argument('--total_steps', default=1000, type=int)
parser.add_argument('--check_steps', default=1, type=int)
parser.add_argument('--check_after', default=-1, type=int)
parser.add_argument('--early_stopping', default=-1, type=int)
parser.add_argument('--p_drop_attn', default=0.1, type=float)
parser.add_argument('--p_drop_hidden', default=0.1, type=float)
parser.add_argument('--debug', action='store_true')


# mixup
parser.add_argument('--sup_mixup', choices=['cls', 'word', 'word_cls'])
parser.add_argument('--unsup_mixup', choices=['cls', 'word', 'word_cls'])
parser.add_argument('--alpha', default=1, type=float)
parser.add_argument('--manifold_mixup', action='store_true')

# SSL
parser.add_argument('--no_unsup_loss', action='store_true')
parser.add_argument('--no_sup_loss', action='store_true')

# ICT
parser.add_argument('--ict', action='store_true')

#mixmatch
parser.add_argument('--mixmatch', action='store_true')
parser.add_argument('--lambda_u', default=100, type=float)
parser.add_argument('--T', default=0.85, type=float)

# uda
parser.add_argument('--uda', action='store_true')
parser.add_argument('--tsa', default="", type=str)
parser.add_argument('--unsup_ratio', default=3, type=int)
parser.add_argument('--uda_coeff', default=1, type=int)
parser.add_argument('--uda_softmax_temp', default=0.85, type=float)
parser.add_argument('--uda_confidence_thresh', default=0.45, type=float)

parser.add_argument('--data_parallel', default=True, type=bool)

parser.add_argument('--model_file', default="", type=str)
parser.add_argument('--task', default="SST", type=str)
parser.add_argument('--num_labels', default=2, type=int)

parser.add_argument('--train_cap', default=-1, type=int)
parser.add_argument('--dev_cap', default=-1, type=int)
parser.add_argument('--unsup_cap', default=-1, type=int)
parser.add_argument('--random_cap', action='store_true')

parser.add_argument('--epochs', default=3, type=int)

parser.add_argument('--use_prepro', action='store_true')

cfg, unknown = parser.parse_known_args()

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

MAX_LENGTHS = {
    "SST": 128,
    "dbpedia": 256,
    "imdb": 128,
    "CoLA": 128,
    "agnews": 128
}

NUM_LABELS = {
    "SST": 2,
    "dbpedia": 10,
    "imdb": 2,
    "CoLA": 2,
    "agnews": 4
}

model_cfg = {
	"dim": 768,
	"dim_ff": 3072,
	"n_layers": 12,
	"p_drop_attn": cfg.p_drop_attn,
	"n_heads": 12,
	"p_drop_hidden": cfg.p_drop_hidden,
	"max_len": MAX_LENGTHS[cfg.task],
	"n_segments": 2,
	"vocab_size": 30522
}

model_cfg = AttributeDict(model_cfg)

set_seeds(cfg.seed)

# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

if (cfg.mixmatch or cfg.uda or cfg.ict) and not cfg.no_unsup_loss:
    ssl = True
else:
    ssl = False

dataset = DataSet(cfg, ssl)

train_dataset, val_dataset, unsup_dataset = dataset.get_dataset()

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order.
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = cfg.train_batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = cfg.val_batch_size # Evaluate with this batch size.
        )

unsup_dataloader = None
if unsup_dataset:
    unsup_dataloader = DataLoader(
        unsup_dataset,
        sampler = RandomSampler(unsup_dataset),
        batch_size = cfg.train_batch_size
    )

# Load BertForSequenceClassification, the pretrained BERT model with a single
# linear classification layer on top.
model = BertForSequenceClassificationCustom.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = NUM_LABELS[cfg.task],
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

# Tell pytorch to run this model on the GPU.
model.cuda()

# Parallel GPU mode
if cfg.data_parallel:
    model = nn.DataParallel(model)

# Note: AdamW is a class from the huggingface library (as opposed to pytorch)
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(),
                  lr = cfg.lr, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )



# Number of training epochs. The BERT authors recommend between 2 and 4.
# We chose to run for 4, but we'll see later that this may be over-fitting the
# training data.

# Total number of training steps is [number of batches] x [number of epochs].
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * cfg.epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
if cfg.ict:
    trainer = ICT_Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        train_loader=train_dataloader,
        val_loader=validation_dataloader,
        unsup_loader=unsup_dataloader,
        cfg=cfg,
        num_labels=NUM_LABELS[cfg.task],
        ssl=ssl
    )
else:
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        train_loader=train_dataloader,
        val_loader=validation_dataloader,
        unsup_loader=unsup_dataloader,
        cfg=cfg,
        num_labels=NUM_LABELS[cfg.task]
    )

trainer.iterate(cfg.epochs)
