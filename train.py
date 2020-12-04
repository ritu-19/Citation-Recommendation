import os
from model import *
from evaluation import *
from pathlib import Path
from transformers import AdamW
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler
from torch.nn import functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE_CLIP = 1e-5

def contrastiveEuclideanLoss(output1, output2, target, size_average=True):
    distances = (output2 - output1).pow(2).sum(1)  # squared distances
    losses = 0.5 * (target.float() * distances +
            (1 + -1 * target).float() * F.relu(0 - (distances + 0.00000001).sqrt()).pow(2))
    return losses.mean() if size_average else losses.sum()


def trainBERTClassification(encodings_train, labels_train, epochs=10, batch_size=8, lr=0.001, lr_decay=0.5, step_size=20):
    model_folder = Path('BERT_CLassification_models')
    model_folder.mkdir(exist_ok=True)
    model = BERTClassification().to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    dataset = TensorDataset(encodings_train['input_ids'], encodings_train['token_type_ids'],
                            encodings_train['attention_mask'], labels_train)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    best_fscore = 0

    print("Training...................")


    for epoch in range(epochs):
        count = 0
        #lr = max(lr * (lr_decay ** (epoch // step_size)), LEARNING_RATE_CLIP)
        if (epoch + 1) % 5 == 0:
            lr = max(lr * lr_decay, LEARNING_RATE_CLIP)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Epoch: {}/{}'.format(epoch+1, epochs))
        print('LR updated to: ', param_group['lr'])
        epoch_loss = 0
        batch_loss = 0
        model = model.train()

        for input_ids, _, attention_mask, labels_train in dataloader:

            loss, prob = model(input_ids, attention_mask, labels_train)
            #print("Train loss", loss)
            #print("Logits in train", prob)
            #loss_func = nn.BCEWithLogitsLoss()
            #loss = F.cross_entropy(prob, labels_train)
            epoch_loss += loss.item()
            batch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += 1

            if count % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, count + 1, batch_loss / 2000))
                batch_loss = 0.0

        fscore = eval_classification(model, mode="val", batch_size=batch_size)
        PATH = os.path.join(model_folder, 'BERTClassification_model_lr-{}_epoch-{}.pth'.format(lr, epoch + 1))
        if fscore > best_fscore:
            print("Saving model:"+"Best score: "+str(best_fscore)+", Fscore: "+str(fscore))
            best_fscore = fscore
            torch.save(model.state_dict(), PATH)

        print("EPOCH Loss ==================== ", str(epoch_loss / count))

    print("Training complete!!")


def trainBERTContrastive(encoding1_train, encoding2_train, labels_train, epochs=10, batch_size=8, lr=0.001, loss_type='contrastive', lr_decay=0.5, step_size=20):
    model_folder = Path('BERT_Contrastive_models')
    model_folder.mkdir(exist_ok=True)
    model = BERTContrastive().to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    dataset = TensorDataset(encoding1_train['input_ids'], encoding1_train['token_type_ids'], encoding1_train['attention_mask'],
                            encoding2_train['input_ids'], encoding2_train['token_type_ids'], encoding2_train['attention_mask'],
                            labels_train)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    best_fscore = 0

    print("Training...................")

    for epoch in range(epochs):
        count = 0
        lr = max(lr * (lr_decay ** (epoch // step_size)), LEARNING_RATE_CLIP)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Epoch: {}/{}'.format(epoch+1, epochs))
        print('LR updated to: ', param_group['lr'])
        epoch_loss = 0
        batch_loss = 0
        model.train()

        for input_ids1, _, attention_mask1, input_ids2, _, attention_mask2, labels_train in dataloader:
            optimizer.zero_grad()
            emd1 = model(input_ids1, attention_mask1)
            emd2 = model(input_ids2, attention_mask2)
            if loss_type == 'contrastive':
                loss = contrastiveEuclideanLoss(emd1, emd2, 2 * labels_train - 1)
            else:
                if loss_type == 'cosine_embedding':
                    criterion = nn.CosineEmbeddingLoss()
                else:
                    #print(emd1.size(), emd2.size(), labels_train.size())
                    criterion = nn.MarginRankingLoss()
                loss = criterion(emd1, emd2, 2 * labels_train - 1)

            epoch_loss += loss.item()
            batch_loss += loss.item()
            loss.backward()
            optimizer.step()
            count += 1

            if count % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, count + 1, batch_loss / 2000))
                batch_loss = 0.0

        fscore = eval_contrastive(model, mode="val", batch_size=batch_size)
        PATH = os.path.join(model_folder, 'BERTContrastive_model_lr-{}_epoch-{}.pth'.format(lr, epoch + 1))
        if fscore > best_fscore:
            print("Saving model:"+"Best score: "+str(best_fscore)+", Fscore: "+str(fscore))
            best_fscore = fscore
            torch.save(model.state_dict(), PATH)
        print("EPOCH Loss ==================== ", str(epoch_loss / count))

    print("Training Complete!!")
