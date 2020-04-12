import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
import torch
import pdb

class DataSet():
    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    def preprocess(self, sentences, labels):
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

            tokens = self.tokenizer.tokenize(sent)

            # pad all tokens to the same length using UNS token

            #max_sent_length = 66
            #paddings = max_sent_length - 2 - len(tokens)
            
            #for i in range(0, paddings):
            #    unused_token = '[unused0]'
            #    tokens.append(unused_token)

            encoded_dict = self.tokenizer.encode_plus(
                                tokens,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = 128,           # Pad & truncate all sentences.
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


    def get_dataset(self):
        # Load the dataset into a pandas dataframe.
        if self.cfg.task == "SST":
            df_train = pd.read_csv("./SST-2/train.tsv", delimiter='\t', header=None, names=['sentence', 'label']).iloc[1:]
            df_dev = pd.read_csv("./SST-2/dev.tsv", delimiter='\t', header=None, names=['sentence', 'label']).iloc[1:]
            df_test = pd.read_csv("./SST-2/test.tsv", delimiter='\t', header=None, names=['idx', 'sentence']).iloc[1:]
             
            df_train['label'] = df_train['label'].astype(int)
            df_dev['label'] = df_dev['label'].astype(int)
         
        if self.cfg.train_cap > 0:
            df_train = df_train.sample(self.cfg.train_cap, random_state=self.cfg.data_seed)

        # Report the number of sentences.
        print('Number of training sentences: {:,}\n'.format(df_train.shape[0]))
        print('Number of dev sentences: {:,}\n'.format(df_dev.shape[0]))
        print('Number of test sentences: {:,}\n'.format(df_test.shape[0]))

        # Display 10 random rows from the data.
        df_train.sample(10)

        # Get the lists of sentences and their labels.
        sentences_train = df_train.sentence.values[1:]
        labels_train = df_train.label.values[1:]

        sentences_dev = df_dev.sentence.values[1:]
        labels_dev = df_dev.label.values[1:]

        input_ids_train, attention_masks_train, seg_ids_train, label_ids_train, num_tokens_train = self.preprocess(sentences_train, labels_train)
        input_ids_dev, attention_masks_dev, seg_ids_dev, label_ids_dev, num_tokens_dev = self.preprocess(sentences_dev, labels_dev)

        # Print sentence 0, now as a list of IDs.
        print('Original: ', sentences_train[0])
        print('Token IDs:', input_ids_train[0])

        # Combine the training inputs into a TensorDataset.
        train_dataset = TensorDataset(input_ids_train, attention_masks_train, seg_ids_train, label_ids_train, num_tokens_train)
        val_dataset = TensorDataset(input_ids_dev, attention_masks_dev, seg_ids_dev, label_ids_dev)

        return train_dataset, val_dataset
