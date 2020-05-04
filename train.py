import time
import datetime
import random
import numpy as np
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

import torch
import torch.nn as nn

from utils import *
from tensorboardX import SummaryWriter
import shutil

import os
import pdb


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

    def seed_torch(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

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

    def get_loss_ict(self, sup_batch, unsup_batch):
        model = self.model
        cfg = self.cfg
        device = self.device

        # batch
        input_ids, segment_ids, input_mask, og_label_ids, num_tokens = sup_batch
        ori_input_ids, ori_segment_ids, ori_input_mask, \
        aug_input_ids, aug_segment_ids, aug_input_mask, \
        ori_num_tokens, aug_num_tokens = unsup_batch

        # convert label ids to hot vectors
        sup_size = input_ids.size(0)
        label_ids = torch.zeros(sup_size, 2).scatter_(1, og_label_ids.cpu().view(-1,1), 1)
        label_ids = label_ids.cuda(non_blocking=True)

        # sup mixup
        sup_l = np.random.beta(cfg.alpha, cfg.alpha)
        sup_l = max(sup_l, 1-sup_l)
        sup_idx = torch.randperm(sup_size)

        if cfg.sup_mixup == 'word':
            input_ids, c_input_ids = pad_for_word_mixup(
                input_ids, input_mask, num_tokens, sup_idx    
            )
        else:
            c_input_ids = None

        # sup loss
        logits = model(
            input_ids=input_ids, 
            c_input_ids=c_input_ids,
            attention_mask=input_mask,
            mixup=cfg.sup_mixup,
            shuffle_idx=sup_idx,
            l=sup_l
        )

        if cfg.sup_mixup:
            label_ids = mixup_op(label_ids, sup_l, sup_idx)

        sup_loss = -torch.sum(F.log_softmax(logits, dim=1) * label_ids, dim=1)

        if cfg.no_unsup_loss:
            return sup_loss, sup_loss, sup_loss, sup_loss
        
        # unsup loss
        unsup_size = ori_input_ids.size(0)

        with torch.no_grad():
            ori_logits = model(
                input_ids=ori_input_ids,
                attention_mask=ori_input_mask
            )
            ori_prob = F.softmax(ori_logits, dim=-1)    # target

        # mixup
        l = np.random.beta(cfg.alpha, cfg.alpha)
        l = max(l, 1-l)
        idx = torch.randperm(unsup_size)

        if cfg.unsup_mixup == 'word':
            ori_input_ids, c_ori_input_ids = pad_for_word_mixup(
                ori_input_ids, ori_input_mask, ori_num_tokens, idx
            )
        else:
            c_ori_input_ids = None

        logits = model(
            input_ids=ori_input_ids, 
            c_input_ids=c_ori_input_ids,
            attention_mask=ori_input_mask,
            mixup=cfg.mixup,
            shuffle_idx=idx,
            l=l
        )
        
        if cfg.unsup_mixup:
            ori_prob = mixup_op(ori_prob, l, idx)

        probs_u = torch.softmax(logits, dim=1)
        unsup_loss = torch.mean((probs_u - ori_prob)**2)

        w = cfg.uda_coeff * sigmoid_rampup(global_step, cfg.consistency_rampup_ends - cfg.consistency_rampup_starts)
        final_loss = sup_loss + w*unsup_loss
        return final_loss, sup_loss, unsup_loss, w*unsup_loss


    def get_loss(self, batch):
            model = self.model
            cfg = self.cfg
            device = self.device

            input_ids, input_mask, segment_ids, labels, num_tokens = batch
            
            batch_size = input_ids.size(0)

            model.zero_grad()        

            # convert label_ids to hot vector
            label_ids = torch.zeros(batch_size, self.num_labels).scatter_(1, labels.view(-1,1), 1).cuda()

            sup_l = np.random.beta(cfg.alpha, cfg.alpha)
            sup_l = max(sup_l, 1-sup_l)
            sup_idx = torch.randperm(batch_size)

            c_input_ids = input_ids.clone()

            if cfg.mixup == 'word':
                for i in range(0, batch_size):
                    j = sup_idx[i]
                    i_count = int(num_tokens[i])
                    j_count = int(num_tokens[j])

                    if i_count < j_count:
                        small = i
                        big = j
                        small_count = i_count
                        big_count = j_count
                        small_ids = input_ids
                        big_ids = c_input_ids
                    elif i_count > j_count:
                        small = j
                        big = i
                        small_count = j_count
                        big_count = i_count
                        small_ids = c_input_ids
                        big_ids = input_ids

                    if i_count != j_count:
                        first = small_ids[small][0:small_count-1]
                        second = torch.tensor([1] * (big_count - small_count))
                        third = big_ids[big][big_count-1:128]
                        combined = torch.cat((first, second, third), 0)
                        small_ids[small] = combined
                        if i_count < j_count:
                            input_mask[i] = input_mask[j]
            
            #for i in range(0, batch_size):
            #    new_mask = input_mask[i]
            #    new_ids = input_ids[i]
            #    old_ids = c_input_ids[i]
            #    pdb.set_trace()

            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            num_tokens = num_tokens.to(device)

            sup_logits = model(
                input_ids=input_ids,
                c_input_ids=c_input_ids,
                attention_mask=input_mask,
                mixup=cfg.mixup,
                shuffle_idx=sup_idx,
                l=sup_l
            )

            if cfg.mixup:
                sup_label_a, sup_label_b = label_ids, label_ids[sup_idx]
                label_ids = sup_l * sup_label_a + (1 - sup_l) * sup_label_b

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

        total_train_loss = 0
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_loader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
            
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_loader), elapsed))

            # loss function
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
        total_prec5 = 0

        # Evaluate data for one epoch
        for batch in val_loader:
            batch = [t.to(device) for t in batch]
            b_input_ids, b_input_mask, b_segment_ids, b_labels, b_num_tokens = batch
            batch_size = b_input_ids.size(0)

            with torch.no_grad():        
                logits = model(
                    input_ids=b_input_ids,
                    attention_mask=b_input_mask
                )
                    
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, b_labels)
            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.

            if self.num_labels == 2:
                logits = logits.detach().cpu().numpy()
                b_labels = b_labels.to('cpu').numpy()
                total_prec1 += bin_accuracy(logits, b_labels)
            else:
                prec1, prec5 = multi_accuracy(logits, b_labels, topk=(1,5))
                total_prec1 += prec1
                total_prec5 += prec5

        avg_prec1 = total_prec1 / len(val_loader)
        avg_prec5 = total_prec5 / len(val_loader)

        avg_val_loss = total_eval_loss / len(val_loader)
        
        # Report the final accuracy for this validation run.
        print("  Top 1 Accuracy: {0:.4f}".format(avg_prec1))
        print("  Top 5 Accuracy: {0:.4f}".format(avg_prec5))

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))


        return avg_prec1, avg_prec5, avg_val_loss, validation_time

    def iterate(self, epochs):
        cfg = self.cfg

        if cfg.results_dir:
            dir = os.path.join('results', cfg.results_dir)
            if os.path.exists(dir) and os.path.isdir(dir):
                shutil.rmtree(dir)

            writer = SummaryWriter(log_dir=dir)


        # Set the seed value all over the place to make this reproducible.        
        self.seed_torch(cfg.seed)

        # We'll store a number of quantities such as training and validation loss, 
        # validation accuracy, and timings.
        training_stats = []

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        best_val_acc = 0
        best_epoch = 0
        best_train_loss = None
        best_val_loss = None

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

        
            avg_prec1, avg_prec5, avg_val_loss, validation_time = self.validate()

            writer.add_scalars('data/losses', {'eval_loss': avg_val_loss}, epoch_i+1)
            writer.add_scalars('data/accuracies', {'eval_acc': avg_prec1}, epoch_i+1)
    
            # update best val accuracy
            if avg_prec1 > best_val_acc:
                best_val_acc = avg_prec1
                best_epoch = epoch_i
                best_train_loss = avg_train_loss
                best_val_loss = avg_val_loss


            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss * 100,
                    'Valid. Loss': avg_val_loss * 100,
                    'Valid. Accur_top1.': avg_prec1,
                    'Valid. Accur_top5.': avg_prec5,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
        print("Best Epoch: {}".format(best_epoch))
        print("Best Training Loss: {}".format(best_train_loss))
        print("Best Val Loss: {}".format(best_val_loss))
        print("Best Validation Accuracy: {0:.4f}".format(best_val_acc))

        writer.close()
