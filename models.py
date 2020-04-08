from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss


class BertForSequenceClassificationCustom(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_h=False,
        input_h=None
    ):
        if input_h is None:
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )

            pooled_output = outputs[1]
            if output_h:
                return pooled_output
        else:
            pooled_output = input_h
        #outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = logits
        
        return outputs  # logits, (hidden_states), (attentions)
