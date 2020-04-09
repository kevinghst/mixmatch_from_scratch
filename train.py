import time
import datetime
import random
import numpy as np
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

import torch

from utils import *

import pdb


class Trainer():
    def __init__(
            self, 
            model=None, optimizer=None, device=None, scheduler=None,
            train_loader=None, val_loader=None, cfg=None
        ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg

    def seed_torch(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def train(self):
        model = self.model
        optimizer = self.optimizer
        device = self.device
        scheduler = self.scheduler
        train_loader = self.train_loader
        cfg = self.cfg

        total_train_loss = 0
        model.train

        # For each batch of training data...
        for step, batch in enumerate(train_loader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
            
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_loader), elapsed))

            b_input_ids, b_input_mask, b_segment_ids, b_labels, b_num_tokens = batch
            
            batch_size = b_input_ids.size(0)

            model.zero_grad()        

            # convert label_ids to hot vector
            sup_size = b_input_ids.size(0)
            label_ids = torch.zeros(sup_size, 2).scatter_(1, b_labels.view(-1,1), 1).cuda()

            sup_l = np.random.beta(cfg.alpha, cfg.alpha)
            sup_idx = torch.randperm(batch_size)

            #if cfg.mixup == 'word':
            #    for i in range(0, batch_size):
            #        j = sup_idx[i]
            #        b_input_mask = b_

            b_input_ids = b_input_ids.to(device)
            b_input_mask = b_input_mask.to(device)
            b_segment_ids = b_segment_ids.to(device)
            b_num_tokens = b_num_tokens.to(device)

            sup_logits = model(
                input_ids=b_input_ids,
                attention_mask=b_input_mask,
                mixup=cfg.mixup,
                shuffle_idx=sup_idx,
                l=sup_l
            )

            if cfg.mixup:
                sup_label_a, sup_label_b = label_ids, label_ids[sup_idx]
                label_ids = sup_l * sup_label_a + (1 - sup_l) * sup_label_b

            loss = -torch.sum(F.log_softmax(sup_logits, dim=1) * label_ids, dim=1)
            loss = torch.mean(loss)

            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_loader)    
        return avg_train_loss


    def iterate(self, epochs):
        cfg = self.cfg

        # Set the seed value all over the place to make this reproducible.        
        self.seed_torch(cfg.seed)

        # We'll store a number of quantities such as training and validation loss, 
        # validation accuracy, and timings.
        training_stats = []

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        best_val_acc = 0
        best_epoch = 0

        for epoch_i in range(0, epochs):
            model = self.model
            device = self.device
            val_loader = self.val_loader
    
            # ========================================
            #               Training
            # ========================================
    
            # Perform one full pass over the training set.

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            avg_train_loss = self.train()
    
            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))
        
            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            model.eval()

            # Tracking variables 
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            # Evaluate data for one epoch
            for batch in val_loader:
        
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_segment_ids = batch[2].to(device)
                b_labels = batch[3].to(device)
        
                with torch.no_grad():        
                    logits = model(
                        input_ids=b_input_ids,
                        attention_mask=b_input_mask
                    )

                    
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, cfg.num_labels), b_labels.view(-1))

                # Accumulate the validation loss.
                total_eval_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
                total_eval_accuracy += flat_accuracy(logits, label_ids)
        

            # Report the final accuracy for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(val_loader)
            print("  Accuracy: {0:.4f}".format(avg_val_accuracy))

            # update best val accuracy
            if avg_val_accuracy > best_val_acc:
                best_val_acc = avg_val_accuracy
                best_epoch = epoch_i

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(val_loader)
    
            # Measure how long the validation run took.
            validation_time = format_time(time.time() - t0)
    
            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
        print("Best Validation Accuracy: {0:.4f}".format(best_val_acc))
        print("Best Epoch: {}".format(best_epoch))
