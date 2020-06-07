from transformers import BertPreTrainedModel
from transformers.modeling_roberta import RobertaModel, RobertaClassificationHead
from transformers.configuration_roberta import RobertaConfig

class RobertaForSequenceClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)

    def forward(
        self,
        input_ids=None,
        c_input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_h=False,
        input_h=None,
        mixup=None,
        shuffle_idx=None,
        l=1,
        manifold_mixup=None,
        no_pretrained_pool=False
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        return logits  # (loss), logits, (hidden_states), (attentions)