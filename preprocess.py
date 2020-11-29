import pandas as pd
import torch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import pickle
import chardet

class Preprocessing:
    def __init__(self, file, taskname):
        self.file = file
        self.taskname = taskname
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    def pickle_dump_classification(self, encoded_abstract_train, encoded_abstract_val, encoded_abstract_test):
        pickle.dump(encoded_abstract_train, open("data/BERTClassificationEncodings_train.pkl", 'wb'))
        pickle.dump(encoded_abstract_val, open("data/BERTClassificationEncodings_val.pkl", 'wb'))
        pickle.dump(encoded_abstract_test, open("data/BERTClassificationEncodings_test.pkl", 'wb'))

    def pickle_dump_contrastive(self, encoded_abstract1_train, encoded_abstract2_train, encoded_abstract1_val,
                                encoded_abstract2_val, encoded_abstract1_test, encoded_abstract2_test):
        pickle.dump(encoded_abstract1_train, open("data/BERTContrastiveEncodings1_train.pkl", 'wb'))
        pickle.dump(encoded_abstract2_train, open("data/BERTContrastiveEncodings2_train.pkl", 'wb'))
        pickle.dump(encoded_abstract1_val, open("data/BERTContrastiveEncodings1_val.pkl", 'wb'))
        pickle.dump(encoded_abstract2_val, open("data/BERTContrastiveEncodings2_val.pkl", 'wb'))
        pickle.dump(encoded_abstract1_test, open("data/BERTContrastiveEncodings1_test.pkl", 'wb'))
        pickle.dump(encoded_abstract2_test, open("data/BERTContrastiveEncodings2_test.pkl", 'wb'))

    def pickle_dump_labels(self, labels_train, labels_val, labels_test, task=""):
        pickle.dump(labels_train, open("data/BERT" + task + "Labels_train.pkl", 'wb'))
        pickle.dump(labels_val, open("data/BERT" + task + "Labels_val.pkl", 'wb'))
        pickle.dump(labels_test, open("data/BERT" + task + "Labels_test.pkl", 'wb'))

    def split_dataset(self):
        df = pd.read_csv(self.file)
        df.dropna(inplace=True)
        train_set, test_set = train_test_split(df, test_size=0.3, shuffle=True)
        val_set, test_set = train_test_split(test_set, test_size=0.6, shuffle=True)
        train_set = train_set.reset_index(drop=True)
        val_set = val_set.reset_index(drop=True)
        test_set = test_set.reset_index(drop=True)

        abstract1_train = list(train_set['paperAbstract1'])
        abstract2_train = list(train_set['paperAbstract2'])
        abstract1_val = list(val_set['paperAbstract1'])
        abstract2_val = list(val_set['paperAbstract2'])
        abstract1_test = list(test_set['paperAbstract1'])
        abstract2_test = list(test_set['paperAbstract2'])
        temp_train = filter(lambda x: x == '1' or x == '0', list(train_set['label']))
        labels_train = torch.tensor(list([int(float(x)) for x in temp_train])).unsqueeze(dim=1).float()
        temp_val = filter(lambda x: x == '1' or x == '0', list(val_set['label']))
        labels_val = torch.tensor(list([int(float(x)) for x in temp_val])).unsqueeze(dim=1).float()
        temp_test = filter(lambda x: x == '1' or x == '0', list(test_set['label']))
        labels_test = torch.tensor(list([int(float(x)) for x in temp_test])).unsqueeze(dim=1).float()

        return abstract1_train, abstract2_train, abstract1_val, abstract2_val, abstract1_test, abstract2_test, \
               labels_train, labels_val, labels_test

    def preprocess(self):
        if self.taskname == "Classification":
            abstract1_train, abstract2_train, abstract1_val, abstract2_val, abstract1_test, abstract2_test, \
            labels_train, labels_val, labels_test = self.split_dataset()

            encoded_abstract_train = self.tokenizer(abstract1_train, abstract2_train, padding=True, truncation=True,
                                                    return_tensors="pt")
            encoded_abstract_val = self.tokenizer(abstract1_val, abstract2_val, padding=True, truncation=True,
                                                    return_tensors="pt")
            encoded_abstract_test = self.tokenizer(abstract1_test, abstract2_test, padding=True, truncation=True,
                                                    return_tensors="pt")

            self.pickle_dump_classification(encoded_abstract_train, encoded_abstract_val, encoded_abstract_test)
            self.pickle_dump_labels(labels_train, labels_val, labels_test, task="Classification")


        else:
            abstract1_train, abstract2_train, abstract1_val, abstract2_val, abstract1_test, abstract2_test, \
            labels_train, labels_val, labels_test = self.split_dataset()

            encoded_abstract1_train = self.tokenizer(abstract1_train, padding=True, truncation=True,
                                                     return_tensors="pt")
            encoded_abstract2_train = self.tokenizer(abstract2_train, padding=True, truncation=True,
                                                     return_tensors="pt")
            encoded_abstract1_val = self.tokenizer(abstract1_val, padding=True, truncation=True,
                                                     return_tensors="pt")
            encoded_abstract2_val = self.tokenizer(abstract2_val, padding=True, truncation=True,
                                                     return_tensors="pt")
            encoded_abstract1_test = self.tokenizer(abstract1_test, padding=True, truncation=True,
                                                     return_tensors="pt")
            encoded_abstract2_test = self.tokenizer(abstract2_test, padding=True, truncation=True,
                                                     return_tensors="pt")

            self.pickle_dump_contrastive(encoded_abstract1_train, encoded_abstract2_train, encoded_abstract1_val,
                                    encoded_abstract2_val, encoded_abstract1_test, encoded_abstract2_test)
            self.pickle_dump_labels(labels_train, labels_val, labels_test, task="Contrastive")

        print("Preprocessing Done!!")

