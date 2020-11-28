


from transformers import BertModel, BertTokenizer
import torch.nn.functional as F
from torch import nn
import torch
from transformers import BertForSequenceClassification


class BERTContrastive(nn.Module):
    def __init__(self, train=True, dropout=0.1):
        super(BERTContrastive, self).__init__()
        # use pretrained BERT
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768, 512)
        self.linear2 = nn.Linear(512, 256)
        print("Done loading model")
        # if train:
        #     self.bert.train()
        # else:
        #     self.bert.eval()
        #     for param in self.bert.parameters():
        #         param.requires_grad = False
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, masks=None):
        # input_ids = torch.tensor(self.tokenizer.encode(inputs)).unsqueeze(0)  # Batch size 1
        print(input_ids.size(), masks.size())
        _, pooled_output = self.bert(input_ids, attention_mask=masks)
        dropout_output = self.dropout(pooled_output)
        linear_output = F.relu(self.linear1(dropout_output))
        linear_output = F.relu(self.linear2(linear_output))
        # print(pooled_output)
        # last_hidden_states = outputs[0]
        # cls = last_hidden_states[0]
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
        print(tokens.size(), masks.size(), pooled_output.size())
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        proba = self.sigmoid(linear_output)
        return proba
