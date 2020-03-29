import torch
import torch.nn as nn
import argparse
import os
import sys

import time
import random
import numpy as np
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

from dataset import DataSet
from train import Trainer

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

parser = argparse.ArgumentParser(description='SSL for NLP')

parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--lr', default=2e-5, type=float)
parser.add_argument('--do_lower_case', default=True, type=bool)

parser.add_argument('--train_batch_size', default=32, type=int)
parser.add_argument('--val_batch_size', default=32, type=int)

parser.add_argument('--data_parallel', default=True, type=bool)

parser.add_argument('--model_file', default="", type=str)
parser.add_argument('--task', default="SST", type=str)

cfg, unknown = parser.parse_known_args()



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

dataset = DataSet(cfg.task)
train_dataset, val_dataset = dataset.get_dataset()


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

# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
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
epochs = 3

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    device=device,
    scheduler=scheduler,
    train_loader=train_dataloader,
    val_loader=validation_dataloader,
    cfg=cfg
)

trainer.train(epochs)