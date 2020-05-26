import time
import datetime
import random
import numpy as np
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef

import torch
import torch.nn as nn

from utils import *
from tensorboardX import SummaryWriter
import shutil

import os
import pdb
import pandas as pd


class Trainer():
    def __init__(
            self, 
            model=None, optimizer=None, device=None, scheduler=None,
            train_loader=None, val_loader=None, unsup_loader=None,
            cfg=None, num_labels=None
        ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.unsup_loader = unsup_loader
        self.cfg = cfg
        self.num_labels = num_labels

    # TSA
    def get_tsa_thresh(schedule, current, rampup_length, start, end):
        training_progress = torch.tensor(float(current) / float(rampup_length))
        if schedule == 'linear_schedule':
            threshold = training_progress
        elif schedule == 'exp_schedule':
            scale = 5
            threshold = torch.exp((training_progress - 1) * scale)
        elif schedule == 'log_schedule':
            scale = 5
            threshold = 1 - torch.exp((-training_progress) * scale)
        output = threshold * (end - start) + start
        return output.to(_get_device())


    def linear_rampup(self, current):
        rampup_length = self.cfg.epochs

        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(float(current) / float(rampup_length), 0.0, 1.0)
            return float(current)

    def semi_loss(self, outputs_x, targets_x, outputs_u, targets_u, current_epoch):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, self.cfg.lambda_u * self.linear_rampup(current_epoch)

    def get_loss(self, batch):
            model = self.model
            cfg = self.cfg
            device = self.device

            input_ids, segment_ids, input_mask, labels, num_tokens = batch
            
            batch_size = input_ids.size(0)

            # convert label_ids to hot vector
            label_ids = torch.zeros(batch_size, self.num_labels).scatter_(1, labels.cpu().view(-1,1), 1).cuda()

            sup_l = np.random.beta(cfg.alpha, cfg.alpha)
            sup_l = max(sup_l, 1-sup_l)
            sup_idx = torch.randperm(batch_size)

            if cfg.sup_mixup and 'word' in cfg.sup_mixup:
                input_ids, c_input_ids = pad_for_word_mixup(
                    input_ids, input_mask, num_tokens, sup_idx    
                )
            else:
                c_input_ids=None
            
            #for i in range(0, batch_size):
            #    new_mask = input_mask[i]
            #    new_ids = input_ids[i]
            #    old_ids = c_input_ids[i]
            #    pdb.set_trace()

            sup_logits = model(
                input_ids=input_ids,
                c_input_ids=c_input_ids,
                attention_mask=input_mask,
                mixup=cfg.sup_mixup,
                shuffle_idx=sup_idx,
                l=sup_l,
                manifold_mixup = cfg.manifold_mixup,
                no_pretrained_pool=cfg.no_pretrained_pool
            )

            if cfg.sup_mixup:
                label_ids = mixup_op(label_ids, sup_l, sup_idx)

            loss = -torch.sum(F.log_softmax(sup_logits, dim=1) * label_ids, dim=1)
            loss = torch.mean(loss)

            return loss

    def train(self):
        # Measure how long the training epoch takes.
        t0 = time.time()

        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        train_loader = self.train_loader
        device = self.device
        cfg = self.cfg

        total_train_loss = 0
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_loader):
            batch = [t.to(device) for t in batch]


            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
            
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_loader), elapsed))

            model.zero_grad()

            loss = self.get_loss(batch)

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
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        return avg_train_loss, training_time
    

    def validate(self):
        t0 = time.time()

        model = self.model
        device = self.device
        val_loader = self.val_loader
        cfg = self.cfg

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        total_prec1 = 0
        total_prec3 = 0

        y_pred = np.array([])
        y_true = np.array([])
        y_conf = np.array([])

        # Evaluate data for one epoch
        for batch in val_loader:
            batch = [t.to(device) for t in batch]
            b_input_ids, b_segment_ids, b_input_mask, b_labels = batch
            batch_size = b_input_ids.size(0)

            with torch.no_grad():        
                logits = model(
                    input_ids=b_input_ids,
                    attention_mask=b_input_mask,
                    no_pretrained_pool=cfg.no_pretrained_pool
                )
                    
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, b_labels)
            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.

            probs = F.softmax(logits, dim=-1)

            if self.num_labels == 2:
                logits = logits.detach().cpu().numpy()
                probs = probs.detach().cpu().numpy()
                b_labels = b_labels.to('cpu').numpy()
                total_prec1 += bin_accuracy(logits, b_labels)

                preds = np.argmax(logits, axis=1).flatten()
                conf = np.max(probs, axis=1).flatten()

                y_true = np.append(y_true, b_labels)
                y_pred = np.append(y_pred, preds)
                y_conf = np.append(y_conf, conf)

            else:
                prec1, prec3 = multi_accuracy(logits, b_labels, topk=(1,3))
                total_prec1 += prec1
                total_prec3 += prec3


        avg_prec1 = total_prec1 / len(val_loader)
        avg_prec3 = total_prec3 / len(val_loader)

        avg_val_loss = total_eval_loss / len(val_loader)

        if y_true.size > 0:
            matt_corr = matthews_corrcoef(y_true, y_pred)
        else:
            matt_corr = None
        
        # Report the final accuracy for this validation run.
        print("  Top 1 Accuracy: {0:.4f}".format(avg_prec1))
        
        if matt_corr is not None:
            print("  Matthew Corr: {0:.4f}".format(matt_corr))

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))


        return avg_prec1, avg_prec3, matt_corr, avg_val_loss, validation_time, y_true, y_pred, y_conf

    def iterate(self, epochs):
        cfg = self.cfg

        if cfg.results_dir:
            dir = os.path.join('results', cfg.results_dir)
            if os.path.exists(dir) and os.path.isdir(dir):
                shutil.rmtree(dir)

            writer = SummaryWriter(log_dir=dir)


        # We'll store a number of quantities such as training and validation loss, 
        # validation accuracy, and timings.
        training_stats = []

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        best_metric = 0
        best_epoch = 0
        best_train_loss = None
        best_val_loss = None
        no_improvement = 0

        ece = -1
        best_true = None
        best_pred = None
        best_conf = None

        if cfg.save_predictions:
            avg_prec1, avg_prec3, matt_corr, avg_val_loss, validation_time, y_true, y_pred, y_conf = self.validate()
            df = pd.DataFrame(y_pred)
            file_path = os.path.join('results', cfg.results_dir, 'begin.xlsx')
            df.to_excel(file_path, index=False)

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

            if self.cfg.uda:
                avg_train_loss, training_time = self.train_uda(epoch_i)
            if self.cfg.mixmatch:
                avg_train_loss, training_time = self.train_mixmatch(epoch_i)
            else:
                avg_train_loss, training_time = self.train()
    
            writer.add_scalars('data/losses', {'train_loss': avg_train_loss}, epoch_i+1)
            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

        
            avg_prec1, avg_prec3, matt_corr, avg_val_loss, validation_time, y_true, y_pred, y_conf = self.validate()

            writer.add_scalars('data/losses', {'eval_loss': avg_val_loss}, epoch_i+1)
            writer.add_scalars('data/accuracies', {'eval_acc': avg_prec1}, epoch_i+1)

            if matt_corr:
                writer.add_scalars('data/matt_corr', {'matt_corr': matt_corr}, epoch_i+1)
            
            # selecting metric for early stopping
            if cfg.task == "CoLA":
                metric = matt_corr
            else:
                metric = avg_prec1

            # update best val accuracy
            if metric > best_metric:
                best_metric = metric
                best_epoch = epoch_i
                best_train_loss = avg_train_loss
                best_val_loss = avg_val_loss
                no_improvement = 0
                
                best_true = y_true
                best_pred = y_pred
                best_conf = y_conf
            else:
                no_improvement += 1

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss * 100,
                    'Valid. Loss': avg_val_loss * 100,
                    'Valid. Accur_top1.': avg_prec1,
                    'Training Time': training_time,
                    'Validation Time': validation_time,
                    'Matthew Correlation': matt_corr
                }
            )

            if no_improvement == self.cfg.early_stopping:
                print("Early stopped")
                break

        ece = calculate_ece(best_true, best_pred, best_conf, cfg.ece)

        if cfg.save_predictions:
            df = pd.DataFrame(best_pred)
            file_path = os.path.join('results', cfg.results_dir, 'best.xlsx')
            df.to_excel(file_path, index=False)

        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

        if cfg.task == "CoLA":
            print("Best Matthew Correlation: {0:.4f}".format(best_metric))
        else:
            print("Best Validation Accuracy: {0:.4f}".format(best_metric))

        print("Best Val Loss: {}".format(best_val_loss))
        print("Best Training Loss: {}".format(best_train_loss))
        print("Best Epoch: {}".format(best_epoch))
        print("Expected Calibration Error: {}".format(ece))

        writer.close()
