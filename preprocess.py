
import numpy as np

import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
from transformers import AutoTokenizer

from torch.utils.data import Dataset, TensorDataset
import pickle


class Preprocessing:
    def __init__(self, file, taskname):
        self.file = file
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

        self.taskname = taskname


    def preprocess(self):
        df = pd.read_csv(self.file, error_bad_lines=False, encoding='latin-1')
        df.dropna(inplace=True)

        abstract1 = list(df['paperAbstract1'])
        abstract2 = list(df['paperAbstract2'])
        labels = torch.tensor(list(df['label'])).unsqueeze(dim=1).float()

        if self.taskname == "Classification":
            encoded_abstract = self.tokenizer(abstract1, abstract2, padding=True, truncation=True, return_tensors="pt")
            # print(encoded_abstract)
            # print(labels.size())
            pickle.dump(encoded_abstract, open("BERTClassificationEncodings.pkl", 'wb'))
            pickle.dump(labels, open("BERTClassificationLabels.pkl", 'wb'))

        else:
            encoded_abstract1 = self.tokenizer(abstract1, padding=True, truncation=True, return_tensors="pt")
            encoded_abstract2 = self.tokenizer(abstract2, padding=True, truncation=True, return_tensors="pt")
            # print(encoded_abstract1, encoded_abstract2)

            # print(encoded_abstract2, encoded_abstract1, labels.size())

            pickle.dump(encoded_abstract1, open("BERTContrastiveEncodings.pkl", 'wb'))
            pickle.dump(encoded_abstract2, open("BERTContrastiveEncodings1.pkl", 'wb'))
            pickle.dump(labels, open("BERTContrastiveLabels.pkl", 'wb'))

        # print(labels, type(labels))
        print("Preprocessing done!!")


# def main():
#     # testing preprocess for contrastive version
#     preprocessCls = Preprocessing('data/test.csv', "Classification")
#     preprocessCls.preprocess()

#     preprocessCls = Preprocessing('data/test.csv', "Contrastive")
#     preprocessCls.preprocess()


# if __name__ == "__main__":
#     main()
