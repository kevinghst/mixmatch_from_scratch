def train_mixmatch(self, epoch):
    t0 = time.time()

    model = self.model
    optimizer = self.optimizer
    device = self.device
    scheduler = self.scheduler
    train_loader = self.train_loader
    unsup_loader = self.unsup_loader
    cfg = self.cfg

    labeled_train_iter = iter(train_loader)

    total_train_loss = 0
    total_sup_loss = 0
    total_unsup_loss = 0

    model.train()

    for step, unsup_batch in enumerate(unsup_loader):
        # Progress update every 40 batches.
        if step % 100 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(unsup_loader), elapsed))


        try:
            sup_batch = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(train_loader)
            sup_batch = labeled_train_iter.next()

        sup_ids, sup_mask, sup_seg, sup_labels, sup_num_tokens = sup_batch
        ori_ids, ori_mask, ori_seg, aug_ids, aug_mask, aug_seg = [t.to(device) for t in unsup_batch]

        batch_size = sup_ids.size(0)

        model.zero_grad()
            
        #convert label_ids to hot vector
        sup_labels = torch.zeros(batch_size, self.num_labels).scatter_(1, sup_labels.view(-1,1), 1)

        sup_ids,sup_mask,sup_seg,sup_labels = sup_ids.cuda(),sup_mask.cuda(),sup_seg.cuda(),sup_labels.cuda(non_blocking=True)

        # compute guessed labels of unlabeled samples:
        with torch.no_grad():
            outputs_u = model(input_ids=ori_ids, attention_mask=ori_mask)
            outputs_u2 = model(input_ids=aug_ids, attention_mask=aug_mask)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p**(1/cfg.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        unsup_labels = torch.cat([targets_u, targets_u], dim=0)

        all_ids = torch.cat([sup_ids, ori_ids, aug_ids], dim=0)
        all_mask = torch.cat([sup_mask, ori_mask, aug_mask], dim=0)

        all_logits = model(input_ids=all_ids, attention_mask=all_mask)

        Lx, Lu, w = self.semi_loss(
            all_logits[:batch_size],
            sup_labels,
            all_logits[batch_size:],
            unsup_labels,
            epoch + float(step/len(unsup_loader))
        )

        loss = Lx + w * Lu

        total_train_loss += loss.item()
        total_sup_loss += Lx.item()
        total_unsup_loss += Lu.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(unsup_loader)
    avg_sup_loss = total_sup_loss / len(unsup_loader)
    avg_unsup_loss = total_unsup_loss / len(unsup_loader)
        
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Average sup loss: {0:.2f}".format(avg_sup_loss))
    print("  Average unsup loss: {0:.2f}".format(avg_unsup_loss))
    print("  Training epcoh took: {:}".format(training_time))

    return avg_train_loss, training_time