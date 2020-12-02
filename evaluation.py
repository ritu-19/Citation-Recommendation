import torch
import pickle
import sys
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sigmoid(x):
    return 1/(1 + np.exp(-x)) 

def eval_classification(model, mode="val", batch_size=8):
    encodings = pickle.load(open("data/BERTClassificationEncodings_{}.pkl".format(mode), 'rb')).to(device)
    labels = pickle.load(open("data/BERTClassificationLabels_{}.pkl".format(mode), 'rb'))
    test_dataset = TensorDataset(encodings['input_ids'], encodings['token_type_ids'],
                            encodings['attention_mask'])
    sampler = RandomSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=sampler, batch_size=batch_size)
    model = model.eval()
    preds = []
    print('......................{} summary...................'.format(mode))
    with torch.no_grad():
        for input_ids, _, attention_mask in test_dataloader:
            #print("input ids", input_ids)
            #print("attention masks", attention_mask)
            logits = model(input_ids, attention_mask).cpu().detach().numpy()
            #print("logits", logits)
            preds += list(logits[:, 0])
            #print("preds", preds)
    

    preds = sigmoid(np.asarray(preds))
    #print(preds)
    #print("----------------------------------------------")
    #print(labels)
    preds[preds > 0.5] = 1
    preds[preds <= 0.5] = 0
    preds = preds.reshape(-1, 1)
    correct = (preds == labels.numpy())
    print('ACCURACY ================= ', correct.sum() / preds.shape[0])
    precision, recall, fscore, _ = score(labels.numpy(), preds, average='macro')
    print(classification_report(labels.numpy(), preds))
    sys.stdout.flush()
    return fscore

def eval_contrastive(model, mode="val", batch_size=8):
    encoding1 = pickle.load(open("data/BERTContrastiveEncodings1_{}.pkl".format(mode), 'rb')).to(device)
    encoding2 = pickle.load(open("data/BERTContrastiveEncodings2_{}.pkl".format(mode), 'rb')).to(device)
    labels = pickle.load(open("data/BERTContrastiveLabels_{}.pkl".format(mode), 'rb'))
    test_dataset = TensorDataset(encoding1['input_ids'], encoding1['token_type_ids'], encoding1['attention_mask'],
                            encoding2['input_ids'], encoding2['token_type_ids'], encoding2['attention_mask'], labels)
    sampler = RandomSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=sampler, batch_size=batch_size)
    model = model.eval()
    predictions = []

    print('......................{} summary...................'.format(mode))
    with torch.no_grad():
        for input_ids1, _, attention_mask1, input_ids2, _, attention_mask2, labels_train in test_dataloader:
            emd1 = model(input_ids1, attention_mask1)
            emd2 = model(input_ids2, attention_mask2)
            #print(emd1, emd2)
            cosine_sim = torch.nn.functional.cosine_similarity(emd1, emd2, dim=1).cpu().detach().numpy()
            #print("Cosine sim", cosine_sim)
            cosine_sim[cosine_sim > 0.5] = 1
            cosine_sim[cosine_sim <= 0.5] = 0
            predictions += list(cosine_sim)
            #print("predictions", predictions)
    #print("Predictions shape:", len(predictions))
    #print("Labels shape:", labels.size())
    precision, recall, fscore, _ = score(labels.numpy(), np.asarray(predictions).reshape(-1, 1), average='macro')
    print(classification_report(labels.numpy(), predictions))
    sys.stdout.flush()
    return fscore




