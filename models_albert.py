from transformers.modeling_albert import AlbertAttention, load_tf_weights_in_albert, AlbertModel, AlbertPreTrainedModel
from transformers.configuration_albert import AlbertConfig
from transformers.modeling_bert import BertEmbeddings, ACT2FN
#from transformers import AlbertPreTrainedModel

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

import pdb



#class AlbertEmbeddings(BertEmbeddings):
#    """
#    Construct the embeddings from word, position and token_type embeddings.
#    """

#    def __init__(self, config):
#        super().__init__(config)

#        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
#        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
#        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)
#        self.LayerNorm = torch.nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)


#class AlbertTransformer(nn.Module):
#    def __init__(self, config):
#        super().__init__()

#        self.config = config
#        self.output_hidden_states = config.output_hidden_states
#        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
#        self.albert_layer_groups = nn.ModuleList([AlbertLayerGroup(config) for _ in range(config.num_hidden_groups)])

#    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
#        hidden_states = self.embedding_hidden_mapping_in(hidden_states)

#        all_attentions = ()

#        if self.output_hidden_states:
#            all_hidden_states = (hidden_states,)

#        for i in range(self.config.num_hidden_layers):
#            # Number of layers in a hidden group
#            layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)

#            # Index of the hidden group
#            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

#            layer_group_output = self.albert_layer_groups[group_idx](
#                hidden_states,
#                attention_mask,
#                head_mask[group_idx * layers_per_group : (group_idx + 1) * layers_per_group],
#                output_attentions
#            )
#            hidden_states = layer_group_output[0]

#            if output_attentions:
#                all_attentions = all_attentions + layer_group_output[-1]

#            if self.output_hidden_states:
#                all_hidden_states = all_hidden_states + (hidden_states,)

#        outputs = (hidden_states,)
#        if self.output_hidden_states:
#            outputs = outputs + (all_hidden_states,)
#        if output_attentions:
#            outputs = outputs + (all_attentions,)
#        return outputs  # last-layer hidden state, (all hidden states), (all attentions)



#class AlbertModel(AlbertPreTrainedModel):

#    config_class = AlbertConfig
#    load_tf_weights = load_tf_weights_in_albert
#    base_model_prefix = "albert"

#    def __init__(self, config):
#        super().__init__(config)

#        self.config = config
#        self.embeddings = AlbertEmbeddings(config)
#        self.encoder = AlbertTransformer(config)
#        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
#        self.pooler_activation = nn.Tanh()

#        self.init_weights()

#    def my_dtype(self):
#        """
#        Get torch.dtype from module, assuming that the whole module has one dtype.
#        """
#        try:
#            first = next(self.parameters())
#            return first.dtype
#        except StopIteration:
#            # For nn.DataParallel compatibility in PyTorch 1.5

#            def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
#                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
#                return tuples

#            gen = self._named_members(get_members_fn=find_tensor_attributes)
#            first_tuple = next(gen)
#            return first_tuple[1].dtype

#    def convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
#        if head_mask.dim() == 1:
#            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
#            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
#        elif head_mask.dim() == 2:
#            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
#        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
#        head_mask = head_mask.to(dtype=self.my_dtype())  # switch to fload if need + fp16 compatibility
#        return head_mask


#    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked = False):
#        if head_mask is not None:
#            head_mask = convert_head_mask_to_5d(head_mask, num_hidden_layers)
#            if is_attention_chunked is True:
#                head_mask = head_mask.unsqueeze(-1)
#        else:
#            head_mask = [None] * num_hidden_layers

#        return head_mask


#    def get_input_embeddings(self):
#        return self.embeddings.word_embeddings

#    def set_input_embeddings(self, value):
#        self.embeddings.word_embeddings = value

#    def _resize_token_embeddings(self, new_num_tokens):
#        old_embeddings = self.embeddings.word_embeddings
#        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
#        self.embeddings.word_embeddings = new_embeddings
#        return self.embeddings.word_embeddings

#    def _prune_heads(self, heads_to_prune):
#        for layer, heads in heads_to_prune.items():
#            group_idx = int(layer / self.config.inner_group_num)
#            inner_group_idx = int(layer - group_idx * self.config.inner_group_num)
#            self.encoder.albert_layer_groups[group_idx].albert_layers[inner_group_idx].attention.prune_heads(heads)

#    def forward(
#        self,
#        input_ids=None,
#        attention_mask=None,
#        token_type_ids=None,
#        position_ids=None,
#        head_mask=None,
#        inputs_embeds=None,
#        output_attentions=None,
#    ):
#        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

#        if input_ids is not None and inputs_embeds is not None:
#            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
#        elif input_ids is not None:
#            input_shape = input_ids.size()
#        elif inputs_embeds is not None:
#            input_shape = inputs_embeds.size()[:-1]
#        else:
#            raise ValueError("You have to specify either input_ids or inputs_embeds")

#        device = input_ids.device if input_ids is not None else inputs_embeds.device

#        if attention_mask is None:
#            attention_mask = torch.ones(input_shape, device=device)
#        if token_type_ids is None:
#            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

#        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
#        extended_attention_mask = extended_attention_mask.to(dtype=self.my_dtype())  # fp16 compatibility
#        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
#        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

#        embedding_output = self.embeddings(
#            input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
#        )
#        encoder_outputs = self.encoder(
#            embedding_output, extended_attention_mask, head_mask=head_mask, output_attentions=output_attentions,
#        )

#        sequence_output = encoder_outputs[0]

#        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0]))

#        outputs = (sequence_output, pooled_output) + encoder_outputs[
#            1:
#        ]  # add hidden_states and attentions if they are here
#        return outputs


class AlbertForSequenceClassification(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

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
        output_attentions=None,
        output_h=False,
        input_h=None,
        mixup=None,
        shuffle_idx=None,
        l=1,
        manifold_mixup=None,
        manifold_upper_cap=999,
        no_pretrained_pool=False
    ):
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits