import pandas as pd
from transformers import BertTokenizer, RobertaTokenizer, AlbertTokenizer
from torch.utils.data import TensorDataset, random_split
import torch
import ast
import json

import pdb

MAX_LENGTHS = {
    "SST": 128,
    "dbpedia": 256,
    "imdb": 128,
    "CoLA": 128,
    "agnews": 128,
    "RTE": 128,
    "BoolQ": 256
}

NUM_LABELS = {
    "SST": 2,
    "dbpedia": 10,
    "imdb": 2,
    "CoLA": 2,
    "agnews": 4,
    "RTE": 2,
    "BoolQ": 2
}


class DataSet():
    def __init__(self, cfg, ssl):
        self.cfg = cfg
        if self.cfg.model == "bert":
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        elif self.cfg.model == "roberta":
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        elif self.cfg.model == "albert":
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-large-v2', do_lower_case=True)
        self.ssl = ssl

    def assert_equal(self, tensors):
        size = tensors[0].size(0)

        for tensor in tensors:
            if tensor.size(0) != size:
                return False
        return True

    def preprocess(self, df):
        sentences = df.sentence.values
        labels = df.label.values

        if 'sentence2' in df:
            sentences2 = df.sentence2.values
        else:
            sentences2 = None

        # Tokenize all of the sentences and map the tokens to thier word IDs.
        input_ids = []
        attention_masks = []
        segment_ids = []
        num_tokens = []

        combined_lengths = []
        
        # For every sentence...
        for i, sent in enumerate(sentences):
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

            if sentences2 is not None:
                tokens2 = self.tokenizer.tokenize(sentences2[i])
                if len(tokens2) > max_len - 2:
                    tokens2 = tokens2[-(max_len - 2):]

                combined_lengths.append(len(tokens) + len(tokens2) + 3)

                if len(tokens) + len(tokens2) > max_len - 3:
                    if self.cfg.task == "BoolQ":
                        tokens2 = tokens2[-(max_len - 3 - len(tokens)):]
                    else:
                        tokens = tokens[-(max_len - 3 - len(tokens2)):]


            if sentences2 is not None:
                encoded_dict = self.tokenizer.encode_plus(
                                    tokens,                      # Sentence to encode.
                                    tokens2,                     # The other sentence
                                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                    max_length = max_len,           # Pad & truncate all sentences.
                                    pad_to_max_length = True,
                                    return_attention_mask = True,   # Construct attn. masks.
                                    return_tensors = 'pt',     # Return pytorch tensors.
                                    is_pretokenized = True
                            )
            else:
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
            if 'token_type_ids' in encoded_dict:
                segment_ids.append(encoded_dict['token_type_ids'])
            else:
                holder = torch.tensor([0])
                segment_ids.append(holder)

            num_tokens.append(len(tokens) + 2) #TODO: change for 2 sentence tasks

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        segment_ids = torch.cat(segment_ids, dim=0)
        labels = torch.tensor(labels)
        num_tokens = torch.tensor(num_tokens)

        return input_ids, attention_masks, segment_ids, labels, num_tokens

    def change_multi_label(self, df):
        num_classes = NUM_LABELS[self.cfg.task]
        df['label'].replace({num_classes:0}, inplace=True)


    def sample_dataset(self, df, total):
        if total <= 0:
            return df

        if self.cfg.random_cap:
            return df.sample(total, random_state=self.cfg.data_seed)

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

        if self.cfg.debug:
            pdb.set_trace()

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

    def create_df_from_json(self, path, task):
        data = []
        with open(path) as f:
            for line in f:
                data.append(json.loads(line))

        df = pd.DataFrame.from_records(data)

        if task == "BoolQ":
            df.rename(columns={"question": "sentence", "passage": "sentence2"}, inplace=True)
            if 'test' in path:
                df['label'] = False

            df['label'].replace(True, 1, inplace=True)
            df['label'].replace(False, 0, inplace=True)

        return df

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
        df_test = None

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

            if self.ssl:
                f_unsup = open("./imdb/imdb_unsup_train.txt", 'r', encoding='utf-8')
                df_unsup = pd.read_csv(f_unsup, sep='\t')
                sup_data = 25000
                df_unsup = df_unsup.iloc[sup_data:]
                if self.cfg.unsup_cap > 0:
                    df_unsup = df_unsup.sample(self.cfg.unsup_cap, random_state=self.cfg.data_seed)

                self.reindex(df_unsup)
            else:
                df_dev = pd.read_csv("./imdb/sup_dev.csv", header=None, names=['sentence', 'label'])

            if self.cfg.test_also or self.cfg.test_mode:
                df_test = df_dev.copy()

        elif self.cfg.task == 'CoLA':
            df_train = pd.read_csv("./CoLA/train.tsv", delimiter='\t', header=None, names=['title', 'label', 'star', 'sentence']).iloc[1:]
            df_dev = pd.read_csv("./CoLA/dev.tsv", delimiter='\t', header=None, names=['title', 'label', 'star', 'sentence']).iloc[1:]
            df_train['label'] = df_train['label'].astype(int)
            df_dev['label'] = df_dev['label'].astype(int)

            if self.cfg.test_also or self.cfg.test_mode:
                if self.cfg.test_dev:
                    df_test = df_dev.copy()
                else:
                    df_test = pd.read_csv("./CoLA/test.tsv", delimiter='\t', header=None, names=['index', 'sentence', 'label']).iloc[1:]
                    df_test = df_test.assign(label=0)

        elif self.cfg.task == 'RTE':
            df_train = pd.read_csv('./RTE/train.tsv', delimiter='\t', header=None, names=['idx', 'sentence', 'sentence2', 'label']).iloc[1:]
            df_dev = pd.read_csv('./RTE/dev.tsv', delimiter='\t', header=None, names=['idx', 'sentence', 'sentence2', 'label']).iloc[1:]

            df_train = df_train[df_train['label'].notnull()]
            df_dev = df_dev[df_dev['label'].notnull()]

            df_train['label'].replace({'not_entailment': 0, 'entailment': 1}, inplace=True)
            df_dev['label'].replace({'not_entailment': 0, 'entailment': 1}, inplace=True)

            df_train['label'] = df_train['label'].astype(int)
            df_dev['label'] = df_dev['label'].astype(int)

            if self.cfg.test_also or self.cfg.test_mode:
                with open('./RTE/test.tsv') as f:
                    raw_data = f.read()
                    data = [row.split('\t') for row in raw_data.split('\n')[:-1]][1:]
                    df_test = pd.DataFrame(data, columns=['idx', 'sentence', 'sentence2'])
                    df_test['label'] = 0

            #df.loc[df['column_name'] == some_value]

        elif self.cfg.task == 'agnews':
            df_train = pd.read_csv("./agnews/train.csv", header=None, names=['label', 'title', 'sentence'])
            df_dev = pd.read_csv("./agnews/test.csv", header=None, names=['label', 'title', 'sentence'])

            if self.cfg.test_also or self.cfg.test_mode:
                if self.cfg.test_path == "full":
                    df_test = pd.read_csv('./agnews/test_full.csv', header=None, names=['label', 'title', 'sentence']).iloc[1:]
                    df_test['label'] = df_test['label'].astype(int)
                elif self.cfg.test_path == "32":
                    df_test = pd.read_csv('./agnews/test_32.csv', header=None, names=['label', 'title', 'sentence']).iloc[1:]
                    df_test['label'] = df_test['label'].astype(int)
                else:
                    df_test = df_dev.copy()
                self.change_multi_label(df_test)

        elif self.cfg.task == 'BoolQ':
            df_train = self.create_df_from_json('./BoolQ/train.jsonl', self.cfg.task)
            df_dev = self.create_df_from_json('./BoolQ/val.jsonl', self.cfg.task)

            df_train['label'] = df_train['label'].astype(int)
            df_dev['label'] = df_dev['label'].astype(int)

            if self.cfg.test_also or self.cfg.test_mode:
                df_test = self.create_df_from_json('./BoolQ/test.jsonl', self.cfg.task)
                df_test['label'] = df_test['label'].astype(int)

        df_train = self.sample_dataset(df_train, self.cfg.train_cap)
        print('Number of training sentences: {:,}\n'.format(df_train.shape[0]))
        input_ids_train, attention_masks_train, seg_ids_train, label_ids_train, num_tokens_train = self.preprocess(df_train)

        df_dev = self.sample_dataset(df_dev, self.cfg.dev_cap)
        print('Number of dev sentences: {:,}\n'.format(df_dev.shape[0]))

        if df_test is not None:
            print('Number of test sentences: {:,}\n'.format(df_test.shape[0]))


        if 'input_ids' in df_dev:
            input_ids_dev, attention_masks_dev, seg_ids_dev, label_ids_dev, num_tokens_dev = self.retrieve_tensors(df_dev, 'sup')

            if df_test is not None:
                input_ids_test, attention_masks_test, seg_ids_test, label_ids_test, num_tokens_test = self.retrieve_tensors(df_test, 'sup')

            if self.ssl:
                ori_input_ids, ori_input_mask, ori_seg_ids, aug_input_ids, aug_input_mask, aug_seg_ids, ori_num_tokens, aug_num_tokens = self.retrieve_tensors(df_unsup, 'unsup')
                print('Number of unsup sentences: {:,}\n'.format(ori_input_ids.shape[0]))
        else:
            input_ids_dev, attention_masks_dev, seg_ids_dev, label_ids_dev, num_tokens_dev = self.preprocess(df_dev)

            if df_test is not None:
                input_ids_test, attention_masks_test, seg_ids_test, label_ids_test, num_tokens_test = self.preprocess(df_test)

        # Combine the training inputs into a TensorDataset.
        train_dataset = TensorDataset(input_ids_train, seg_ids_train, attention_masks_train, label_ids_train, num_tokens_train)
        val_dataset = TensorDataset(input_ids_dev, seg_ids_dev, attention_masks_dev, label_ids_dev)

        test_dataset = None
        if df_test is not None:
            test_dataset = TensorDataset(input_ids_test, seg_ids_test, attention_masks_test, label_ids_test)

        unsup_dataset = None
        if self.ssl:
            unsup_dataset = TensorDataset(ori_input_ids, ori_seg_ids, ori_input_mask, aug_input_ids, aug_seg_ids, aug_input_mask, ori_num_tokens, aug_num_tokens)

        return train_dataset, val_dataset, unsup_dataset, test_dataset
