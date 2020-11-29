import torch.nn.functional as F
from torch import nn
from transformers import BertModel


class BERTContrastive(nn.Module):
    def __init__(self, train=True, dropout=0.1):
        super(BERTContrastive, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768, 512)
        self.linear2 = nn.Linear(512, 256)
        print("Done loading model")

    def forward(self, input_ids, masks=None):
        _, pooled_output = self.bert(input_ids, attention_mask=masks)
        dropout_output = self.dropout(pooled_output)
        linear_output = F.relu(self.linear1(dropout_output))
        linear_output = F.relu(self.linear2(linear_output))
        return linear_output

class BERTClassification(nn.Module):
    def __init__(self, dropout=0.1):
        super(BERTClassification, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, tokens, masks=None):
        _, pooled_output = self.bert(tokens, attention_mask=masks)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        proba = self.sigmoid(linear_output)
        return proba
