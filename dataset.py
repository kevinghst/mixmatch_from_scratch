import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
import torch
import ast

import pdb

MAX_LENGTHS = {
    "SST": 128,
    "dbpedia": 256,
    "imdb": 128
}

NUM_LABELS = {
    "SST": 2,
    "dbpedia": 10,
    "imdb": 2
}


class DataSet():
    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    def preprocess(self, df):
        sentences = df.sentence.values
        labels = df.label.values

        # Tokenize all of the sentences and map the tokens to thier word IDs. 
        input_ids = []
        attention_masks = []
        segment_ids = []
        num_tokens = []
        # For every sentence...
        for sent in sentences:
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
           
            max_len = MAX_LENGTHS[self.cfg.task]

            tokens = self.tokenizer.tokenize(sent)
            if len(tokens) > max_len - 2:
                tokens = tokens[-(max_len - 2):]

            # pad all tokens to the same length using UNS token

            #max_sent_length = 66
            #paddings = max_sent_length - 2 - len(tokens)
            
            #for i in range(0, paddings):
            #    unused_token = '[unused0]'
            #    tokens.append(unused_token)

            encoded_dict = self.tokenizer.encode_plus(
                                tokens,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = max_len,           # Pad & truncate all sentences.
                                pad_to_max_length = True,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
                                is_pretokenized = True
                        )


            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
            segment_ids.append(encoded_dict['token_type_ids'])
            num_tokens.append(len(tokens) + 2)

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        segment_ids = torch.cat(segment_ids, dim=0)
        labels = torch.tensor(labels)
        num_tokens = torch.tensor(num_tokens)

        return input_ids, attention_masks, segment_ids, labels, num_tokens

    def sample_dataset(self, df, total):
        if total <= 0:
            return df
        
        num_classes = NUM_LABELS[self.cfg.task]
        per_class = int(total / num_classes)

        class_pop = [per_class] * num_classes
        for i in range(0, total % num_classes):
            class_pop[i] += 1

        min_label = df['label'].min()
        df_sample = df[df['label'] == min_label - 1]

        for i in range(min_label, min_label + num_classes):
            sample_number = class_pop.pop(0)
            df_sub = df[df['label'] == i].sample(sample_number, random_state=self.cfg.data_seed)
            if i == num_classes:
                df_sub['label'] = 0
            df_sample = pd.concat([df_sample, df_sub])

        self.reindex(df_sample)
        return df_sample

    def retrieve_tensors(self, data, d_type):
        if d_type == 'unsup':
            input_columns = ['ori_input_ids', 'ori_input_mask', 'ori_input_type_ids',
                             'aug_input_ids', 'aug_input_mask', 'aug_input_type_ids']
            tensors = [torch.tensor(data[c].apply(lambda x: ast.literal_eval(x)), dtype=torch.long) for c in input_columns]

            ori_num_tokens = []
            aug_num_tokens = []

            for ori_inp in tensors[0]:
                num = (ori_inp!=0).sum()
                ori_num_tokens.append(num.item())

            for aug_inp in tensors[3]:
                num = (aug_inp!=0).sum()
                aug_num_tokens.append(num.item())

            tensors.append(torch.tensor(ori_num_tokens))
            tensors.append(torch.tensor(aug_num_tokens))

        else:
            input_columns = ['input_ids', 'input_mask', 'input_type_ids', 'label']
            tensors = [torch.tensor(data[c].apply(lambda x: ast.literal_eval(x)), dtype=torch.long)    \
                                                                            for c in input_columns[:-1]]
            tensors.append(torch.tensor(data[input_columns[-1]], dtype=torch.long))

            num_tokens = []
            for inp in tensors[0]:
                num = (inp!=0).sum()
                num_tokens.append(num.item())

            tensors.append(torch.tensor(num_tokens))

        return tensors

    def swap_binary_label(self, df):
        df['label'].replace(0, "1", inplace=True)
        df['label'].replace(1, 0, inplace=True)
        df['label'].replace("1", 1, inplace=True)

    def reindex(self, df):
        new_index = range(df.shape[0])
        new_index = pd.Series(new_index)
        df.set_index([new_index], inplace=True)

    def get_dataset(self):
        # Load the dataset into a pandas dataframe.
        df_unsup = None

        if self.cfg.task == "SST":
            df_train = pd.read_csv("./SST-2/train.tsv", delimiter='\t', header=None, names=['sentence', 'label']).iloc[1:]
            df_dev = pd.read_csv("./SST-2/dev.tsv", delimiter='\t', header=None, names=['sentence', 'label']).iloc[1:]
             
            df_train['label'] = df_train['label'].astype(int)
            df_dev['label'] = df_dev['label'].astype(int)
        elif self.cfg.task == "dbpedia":
            df_train = pd.read_csv("./dbpedia/train.csv", header=None, names=['label', 'title', 'sentence']).iloc[1:]
            df_dev = pd.read_csv("./dbpedia/test.csv", header=None, names=['label', 'title', 'sentence']).iloc[1:]
        elif self.cfg.task == "imdb":
            df_train = pd.read_csv("./imdb/sup_train.csv", header=None, names=['sentence', 'label']).iloc[1:]
            #if self.cfg.use_prepro:
            # use prepro for unsup and val
            f_dev = open("./imdb/imdb_sup_test.txt", 'r', encoding='utf-8')
            df_dev = pd.read_csv(f_dev, sep='\t')
            df_dev.rename(columns={"label_ids": "label"}, inplace=True)
            self.swap_binary_label(df_dev)

            if self.cfg.uda_mode:
                f_unsup = open("./imdb/imdb_unsup_train.txt", 'r', encoding='utf-8')
                df_unsup = pd.read_csv(f_unsup, sep='\t')
                sup_data = 25000
                df_unsup = df_unsup.iloc[sup_data:]
                if self.cfg.unsup_cap > 0:
                    df_unsup = df_unsup.sample(self.cfg.unsup_cap, random_state=self.cfg.data_seed)

                self.reindex(df_unsup)
            else:
                df_dev = pd.read_csv("./imdb/sup_dev.csv", header=None, names=['sentence', 'label'])



        df_train = self.sample_dataset(df_train, self.cfg.train_cap)
        print('Number of training sentences: {:,}\n'.format(df_train.shape[0]))
        input_ids_train, attention_masks_train, seg_ids_train, label_ids_train, num_tokens_train = self.preprocess(df_train)

        df_dev = self.sample_dataset(df_dev, self.cfg.dev_cap)
        print('Number of dev sentences: {:,}\n'.format(df_dev.shape[0]))


        if 'input_ids' in df_dev:
            input_ids_dev, attention_masks_dev, seg_ids_dev, label_ids_dev, num_tokens_dev = self.retrieve_tensors(df_dev, 'sup')
            if self.cfg.uda_mode:
                ori_input_ids, ori_input_mask, ori_seg_ids, aug_input_ids, aug_input_mask, aug_seg_ids, ori_num_tokens, aug_num_tokens = self.retrieve_tensors(df_unsup, 'unsup')
                print('Number of unsup sentences: {:,}\n'.format(ori_input_ids.shape[0]))
        else:
            input_ids_dev, attention_masks_dev, seg_ids_dev, label_ids_dev, num_tokens_dev = self.preprocess(df_dev)

        # Combine the training inputs into a TensorDataset.
        train_dataset = TensorDataset(input_ids_train, seg_ids_train, attention_masks_train, label_ids_train, num_tokens_train)
        val_dataset = TensorDataset(input_ids_dev, seg_ids_dev, attention_masks_dev, label_ids_dev)

        unsup_dataset = None
        if self.cfg.uda_mode:
            unsup_dataset = TensorDataset(ori_input_ids, ori_seg_ids, ori_input_mask, aug_input_ids, aug_seg_ids, aug_input_mask, ori_num_tokens, aug_num_tokens)

        return train_dataset, val_dataset, unsup_dataset