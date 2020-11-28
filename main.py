import numpy as np

import pandas as pd
import torch
import random as rn
from transformers import BertModel, BertTokenizer
from transformers import AutoTokenizer

from torch.utils.data import Dataset, TensorDataset
import pickle

from train import *
from preprocess import *

import numpy as np

import pandas as pd
import torch
import random as rn
from transformers import BertModel, BertTokenizer
from transformers import AutoTokenizer

from torch.utils.data import Dataset, TensorDataset
import pickle


# from train import *
# from preprocess import *


def main():
    rn.seed(321)
    np.random.seed(321)
    torch.manual_seed(321)
    torch.cuda.manual_seed(321)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    preprocessed = False

    task = "Classification"
    task = "Contrastive"

    if not preprocessed:
        preprocessCls = Preprocessing('data_dummy.csv', task)
        preprocessCls.preprocess()

        # preprocessCls = Preprocessing('data/data_dummy.csv', task)
        # preprocessCls.preprocess()

    if task == "Classification":
        encodings = pickle.load(open("BERT" + task + "Encodings.pkl", 'rb'))
        labels = pickle.load(open("BERT" + task + "Labels.pkl", 'rb'))
        trainBERTClassification(encodings.to(device), labels.to(device))


    else:
        encodings1 = pickle.load(open("BERT" + task + "Encodings.pkl", 'rb'))
        labels = pickle.load(open("BERT" + task + "Labels.pkl", 'rb'))
        encodings2 = pickle.load(open("BERT" + task + "Encodings.pkl", 'rb'))
        # print(str(torch.cuda.memory_allocated(device) / 1000000) + 'M')

        trainBERTContrastive(encodings1.to(device), encodings2.to(device), labels.to(device))


if __name__ == "__main__":
    main()
