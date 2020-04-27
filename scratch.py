def train_uda(self, epoch):
    t0 = time.time()

    model = self.model
    optimizer = self.optimizer
    device = self.device
    scheduler = self.schedulera
    train_loader = self.train_loader
    unsup_loader = self.unsup_loader
    cfg = self.cfga

    labeled_train_iter = iter(train_loader)

    total_train_loss = 0
    total_sup_loss = 0
    total_unsup_loss = 0

    model.train()

    for step, batch in enumerate(unsup_loader):

        model.zero_grad() 

        #batch
        try:
            input_ids, input_mask, segment_ids, label_ids, num_tokens = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(train_loader)
            input_ids, input_mask, segment_ids, label_ids, num_tokens = labeled_train_iter.next()

        ori_input_ids, ori_segment_ids, ori_input_mask, \
        aug_input_ids, aug_segment_ids, aug_input_mask  = batch

        label_ids = label_ids.to(device)
        input_ids = torch.cat((input_ids, aug_input_ids), dim=0).to(device)
        segment_ids = torch.cat((segment_ids, aug_segment_ids), dim=0).to(device)
        input_mask = torch.cat((input_mask, aug_input_mask), dim=0).to(device)

        #logits
        logits = model(input_ids=input_ids, attention_mask=input_mask)

        #sup loss
        sup_criterion = nn.CrossEntropyLoss(reduction='none')
        unsup_criterion = nn.KLDivLoss(reduction='none')
        current = float(epoch) + float(step/len(unsup_loader))
        rampup_length = float(cfg.epochs)

        sup_size = label_ids.shape[0]            
        sup_loss = sup_criterion(logits[:sup_size], label_ids)  # shape : train_batch_size
        if cfg.tsa:
            tsa_thresh = get_tsa_thresh(cfg.tsa, current, rampup_length, start=1./logits.shape[-1], end=1)
            larger_than_threshold = torch.exp(-sup_loss) > tsa_thresh   # prob = exp(log_prob), prob > tsa_threshold
            # larger_than_threshold = torch.sum(  F.softmax(pred[:sup_size]) * torch.eye(num_labels)[sup_label_ids]  , dim=-1) > tsa_threshold
            loss_mask = torch.ones_like(label_ids, dtype=torch.float32) * (1 - larger_than_threshold.type(torch.float32))
            sup_loss = torch.sum(sup_loss * loss_mask, dim=-1) / torch.max(torch.sum(loss_mask, dim=-1), torch_device_one())
        else:
            sup_loss = torch.mean(sup_loss)


        # ori
        with torch.no_grad():
            ori_logits = model(ori_input_ids, ori_segment_ids, ori_input_mask)
            ori_prob   = F.softmax(ori_logits, dim=-1)    # KLdiv target
            # ori_log_prob = F.log_softmax(ori_logits, dim=-1)

            # confidence-based masking
            if cfg.uda_confidence_thresh != -1:
                unsup_loss_mask = torch.max(ori_prob, dim=-1)[0] > cfg.uda_confidence_thresh
                unsup_loss_mask = unsup_loss_mask.type(torch.float32)
            else:
                unsup_loss_mask = torch.ones(len(logits) - sup_size, dtype=torch.float32)
            unsup_loss_mask = unsup_loss_mask.to(_get_device())
                    
        # aug
        # softmax temperature controlling
        uda_softmax_temp = cfg.uda_softmax_temp if cfg.uda_softmax_temp > 0 else 1.
        aug_log_prob = F.log_softmax(logits[sup_size:] / uda_softmax_temp, dim=-1)

        # KLdiv loss
        """
            nn.KLDivLoss (kl_div)
            input : log_prob (log_softmax)
            target : prob    (softmax)
            https://pytorch.org/docs/stable/nn.html

            unsup_loss is divied by number of unsup_loss_mask
            it is different from the google UDA official
            The official unsup_loss is divided by total
            https://github.com/google-research/uda/blob/master/text/uda.py#L175
        """
        unsup_loss = torch.sum(unsup_criterion(aug_log_prob, ori_prob), dim=-1)
        unsup_loss = torch.sum(unsup_loss * unsup_loss_mask, dim=-1) / torch.max(torch.sum(unsup_loss_mask, dim=-1), torch_device_one())
        loss = sup_loss + cfg.uda_coeff*unsup_loss


        total_train_loss += loss.item()
        total_sup_loss += sup_loss.item()
        total_unsup_loss += unsup_loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

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