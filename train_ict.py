import time
import datetime
import random
import numpy as np
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from tqdm import tqdm

import torch
import torch.nn as nn

from utils import *
from tensorboardX import SummaryWriter
import shutil

import os
import pdb


class ICT_Trainer():
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

    def get_loss_ict(self, sup_batch, unsup_batch, global_step):
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
        sup_loss = torch.mean(loss)

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
            mixup=cfg.unsup_mixup,
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
            b_input_ids, b_segment_ids, b_input_mask, b_labels = batch
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

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)


        return avg_prec1, avg_val_loss

    def iterate(self, epochs):
        cfg = self.cfg
        model = self.model
        device = self.device
        optimizer = self.optimizer

        sup_iter = self.repeat_dataloader(self.train_loader)
        unsup_iter = self.repeat_dataloader(self.unsup_loader)

        if cfg.results_dir:
            dir = os.path.join('results', cfg.results_dir)
            if os.path.exists(dir) and os.path.isdir(dir):
                shutil.rmtree(dir)

            writer = SummaryWriter(log_dir=dir)

        meters = AverageMeterSet()

        model.train()
        model = model.to(device)

        global_step = 0
        loss_sum = 0.
        max_acc = [0., 0, 0., 0.]   # acc, step, val_loss, train_loss
        no_improvement = 0

        iter_bar = tqdm(unsup_iter, total=cfg.total_steps, disable=cfg.hide_tqdm) if cfg.ict \
              else tqdm(sup_iter, total=cfg.total_steps, disable=cfg.hide_tqdm)

        for i, batch in enumerate(iter_bar):
            if cfg.ict:
                sup_batch = [t.to(device) for t in next(sup_iter)]
                unsup_batch = [t.to(device) for t in batch]

                unsup_batch_size = unsup_batch_size or unsup_batch[0].shape[0]

                if unsup_batch[0].shape[0] != unsup_batch_size:
                    continue
            else:
                sup_batch = [t.to(device) for t in batch]
                unsup_batch = None

            optimizer.zero_grad()
            final_loss, sup_loss, unsup_loss, weighted_unsup_loss = get_loss_ict(model, sup_batch, unsup_batch, global_step)

            if cfg.no_sup_loss:
                final_loss = unsup_loss
            elif cfg.no_unsup_loss:
                final_loss = sup_loss

            meters.update('train_loss', final_loss.item())
            meters.update('sup_loss', sup_loss.item())
            meters.update('unsup_loss', unsup_loss.item())
            meters.update('w_unsup_loss', weighted_unsup_loss.item())
            meters.update('lr', optimizer.get_lr()[0])

            final_loss.backward()
            optimizer.step()

            # print loss
            global_step += 1

            if global_step % cfg.check_steps == 0 and global_step > cfg.check_after:
                total_accuracy, avg_val_loss = self.validate()

                #logging
                writer.add_scalars('data/eval_acc', {'eval_acc' : total_accuracy}, global_step)
                writer.add_scalars('data/eval_loss', {'eval_loss': avg_val_loss}, global_step)

                if cfg.no_unsup_loss:
                    writer.add_scalars('data/train_loss', {'train_loss': meters['train_loss'].avg}, global_step)
                    writer.add_scalars('data/lr', {'lr': meters['lr'].avg}, global_step)
                else:
                    writer.add_scalars('data/train_loss', {'train_loss': meters['train_loss'].avg}, global_step)
                    writer.add_scalars('data/sup_loss', {'sup_loss': meters['sup_loss'].avg}, global_step)
                    writer.add_scalars('data/unsup_loss', {'unsup_loss': meters['unsup_loss'].avg}, global_step)
                    writer.add_scalars('data/w_unsup_loss', {'w_unsup_loss': meters['w_unsup_loss'].avg}, global_step)
                    writer.add_scalars('data/lr', {'lr': meters['lr'].avg}, global_step)

                meters.reset()

                if max_acc[0] < total_accuracy:
                    max_acc = total_accuracy, global_step, avg_val_loss, final_loss.item()
                    no_improvement = 0
                else:
                    no_improvement += 1

                print("  Top 1 Accuracy: {0:.4f}".format(total_accuracy))
                print("  Validation Loss: {0:.4f}".format(avg_val_loss))
                print("  Train Loss: {0:.4f}".format(final_loss.item()))
                if ssl_mode:
                    print("  Sup Loss: {0:.4f}".format(sup_loss.item()))
                    print("  Unsup Loss: {0:.4f}".format(unsup_loss.item()))
                print("  Learning rate: {0:.7f}".format(optimizer.get_lr()[0]))

                print(
                    'Max Accuracy : %5.3f Best Val Loss : %5.3f Best Train Loss : %5.4f Max global_steps : %d Cur global_steps : %d' 
                    %(max_acc[0], max_acc[2], max_acc[3], max_acc[1], global_step), end='\n\n'
                )

                if no_improvement == cfg.early_stopping:
                    print("Early stopped")
                    break

        writer.close()


    def repeat_dataloader(self, iterable):
        """ repeat dataloader """
        while True:
            for x in iterable:
                yield x