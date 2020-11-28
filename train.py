import numpy as np

import pandas as pd
import torch
from transformers import BertModel, BertTokenizer, AdamW
from transformers import AutoTokenizer
import torch.nn.functional as F
from torch import nn

from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler
import pickle
from model import *

import numpy as np

import pandas as pd
import torch
import torch
from transformers import BertModel, BertTokenizer, AdamW
from transformers import AutoTokenizer

from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler
import pickle


# from model import *


def contrastiveEuclideanLoss(output1, output2, target, size_average=True):
    distances = (output2 - output1).pow(2).sum(1)  # squared distances
    losses = 0.5 * (target.float() * distances +
                    (1 + -1 * target).float() * F.relu(0 - (distances + 0.00000001).sqrt()).pow(2))
    return losses.mean() if size_average else losses.sum()


def trainBERTClassification(encodings, labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTClassification().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    dataset = TensorDataset(encodings['input_ids'], encodings['token_type_ids'], encodings['attention_mask'], labels)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=8)
    epochs = 20
    count = 0

    print("Starting to train!!")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        batch_loss = 0

        for input_ids, _, attention_mask, labels in dataloader:
            optimizer.zero_grad()

            prob = model(input_ids, attention_mask)
            print(prob)
            loss_func = nn.BCELoss()
            loss = loss_func(prob, labels)
            epoch_loss += loss.item()
            batch_loss += loss.item()
            loss.backward()
            optimizer.step()
            count += 1

            if count % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, count + 1, batch_loss / 2000))
                batch_loss = 0.0

        print("EPOCH Loss ====================", str(epoch_loss))

    print("Training complete!!")


def trainBERTContrastive(encoding1, encoding2, labels):
    # print(str(torch.cuda.memory_allocated(device) / 1000000) + 'M')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTContrastive()

    model = model.to(device)
    # print(str(torch.cuda.memory_allocated(device) / 1000000) + 'M')
    optimizer = AdamW(model.parameters(), lr=1e-5)
    print(encoding1['input_ids'].size())
    dataset = TensorDataset(encoding1['input_ids'], encoding1['token_type_ids'], encoding1['attention_mask'],
                            encoding2['input_ids'], encoding2['token_type_ids'], encoding2['attention_mask'], labels)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=8)
    epochs = 20
    count = 0

    print("Starting to train!!")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        batch_loss = 0

        for input_ids1, _, attention_mask1, input_ids2, _, attention_mask2, labels in dataloader:
            optimizer.zero_grad()
            emd1 = model(input_ids1, attention_mask1)
            emd2 = model(input_ids2, attention_mask2)

            # criterion = nn.CosineEmbeddingLoss()
            # criterion = contrastiveEuclideanLoss
            criterion = nn.MarginRankingLoss()
            loss = criterion(emd1, emd2, 2 * labels - 1)
            epoch_loss += loss.item()
            batch_loss += loss.item()
            loss.backward()
            optimizer.step()
            count += 1

            if count % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, count + 1, batch_loss / 2000))
                batch_loss = 0.0

        print("EPOCH Loss ====================", str(epoch_loss))

    print("Training complete!!")