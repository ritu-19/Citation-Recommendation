import torch
import pickle
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from sklearn.metrics import precision_recall_fscore_support, classification_report


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_classification(model, mode="val", batch_size=8):
    encodings = pickle.load(open("data/BERTClassificationEncodings_{}.pkl".format(mode), 'rb')).to(device)
    labels = pickle.load(open("data/BERTClassificationLabels_{}.pkl".format(mode), 'rb'))
    test_dataset = TensorDataset(encodings['input_ids'], encodings['token_type_ids'],
                            encodings['attention_mask'])
    sampler = RandomSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=sampler, batch_size=8)
    model = model.eval()
    predictions = []

    print('......................{} summary...................'.format(mode))
    with torch.no_grad():
        for input_ids, _, attention_mask in test_dataloader:
            logits = model(input_ids, attention_mask).cpu().detach().numpy()
            predictions += list(logits[:, 0] > 0.5)
    precision, recall, fscore, _ = precision_recall_fscore_support(labels.numpy(), predictions)
    print(classification_report(labels.numpy(), predictions))
    return fscore

def eval_contrastive(model, mode="val", batch_size=8):
    encoding1 = pickle.load(open("data/BERTContrastiveEncodings1_{}.pkl".format(mode), 'rb')).to(device)
    encoding2 = pickle.load(open("data/BERTContrastiveEncodings2_{}.pkl".format(mode), 'rb')).to(device)
    labels = pickle.load(open("data/BERTContrastiveLabels_{}.pkl".format(mode), 'rb'))
    test_dataset = TensorDataset(encoding1['input_ids'], encoding1['token_type_ids'], encoding1['attention_mask'],
                            encoding2['input_ids'], encoding2['token_type_ids'], encoding2['attention_mask'], labels)
    sampler = RandomSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=sampler, batch_size=8)
    model = model.eval()
    predictions = []

    print('......................{} summary...................'.format(mode))
    with torch.no_grad():
        for input_ids1, _, attention_mask1, input_ids2, _, attention_mask2, labels_train in test_dataloader:
            emd1 = model(input_ids1, attention_mask1)
            emd2 = model(input_ids2, attention_mask2)

            cosine_sim = torch.nn.functional.cosine_similarity(emd1, emd2, dim=1)
            predictions += list(cosine_sim[0, :] > 0.2)
    precision, recall, fscore, _ = precision_recall_fscore_support(labels.numpy(), predictions)
    print(classification_report(labels.numpy(), predictions))
    return fscore




