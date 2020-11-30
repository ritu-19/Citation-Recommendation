import os
from model import *
from evaluation import *
from pathlib import Path
from transformers import AdamW
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_folder = Path('models')
model_folder.mkdir(exist_ok=True)
LEARNING_RATE_CLIP = 1e-5

def contrastiveEuclideanLoss(output1, output2, target, size_average=True):
    distances = (output2 - output1).pow(2).sum(1)  # squared distances
    losses = 0.5 * (target.float() * distances +
            (1 + -1 * target).float() * F.relu(0 - (distances + 0.00000001).sqrt()).pow(2))
    return losses.mean() if size_average else losses.sum()


def trainBERTClassification(encodings_train, labels_train, epochs=10, batch_size=8, lr=0.001, lr_decay=0.5, step_size=20):
    model = BERTClassification().to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    dataset = TensorDataset(encodings_train['input_ids'], encodings_train['token_type_ids'],
                            encodings_train['attention_mask'], labels_train)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    count = 0
    best_fscore = 0

    print('Starting to Train!!')
    model = model.train()
    for epoch in range(epochs):
        lr = max(lr * (lr_decay ** (epoch // step_size)), LEARNING_RATE_CLIP)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Epoch: {}/{}'.format(epoch+1, epochs))
        print('LR updated to: ', param_group['lr'])
        model.train()
        epoch_loss = 0
        batch_loss = 0

        for input_ids, _, attention_mask, labels_train in dataloader:
            optimizer.zero_grad()

            prob = model(input_ids, attention_mask)
            loss_func = nn.BCELoss()
            loss = loss_func(prob, labels_train)
            epoch_loss += loss.item()
            batch_loss += loss.item()
            loss.backward()
            optimizer.step()
            count += 1

            if count % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, count + 1, batch_loss / 2000))
                batch_loss = 0.0

        fscore = eval_classification(model, mode="val", batch_size=batch_size)
        print('FSCORE: ', fscore)
        PATH = os.path.join(model_folder, 'BERTClassification_model_lr-{}.pth'.format(lr))
        if fscore > best_fscore:
            best_fscore = fscore
            torch.save(model.state_dict(), PATH)

        print("EPOCH Loss ==================== ", str(epoch_loss))

    print("Training complete!!")


def trainBERTContrastive(encoding1_train, encoding2_train, labels_train, epochs=10, batch_size=8, lr=0.001, loss='contrastive', lr_decay=0.5, step_size=20):
    model = BERTContrastive().to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    dataset = TensorDataset(encoding1_train['input_ids'], encoding1_train['token_type_ids'], encoding1_train['attention_mask'],
                            encoding2_train['input_ids'], encoding2_train['token_type_ids'], encoding2_train['attention_mask'],
                            labels_train)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    count = 0

    print("Training...................")

    for epoch in range(epochs):
        lr = max(lr * (lr_decay ** (epoch // step_size)), LEARNING_RATE_CLIP)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Epoch: {}/{}'.format(epoch+1, epochs))
        print('LR updated to: ', param_group['lr'])
        model.train()
        epoch_loss = 0
        batch_loss = 0

        for input_ids1, _, attention_mask1, input_ids2, _, attention_mask2, labels_train in dataloader:
            optimizer.zero_grad()
            emd1 = model(input_ids1, attention_mask1)
            emd2 = model(input_ids2, attention_mask2)
            if loss == 'constrastive':
                loss = contrastiveEuclideanLoss(emd1, emd2, 2 * labels_train - 1)
            else:
                if loss == 'cosine_embedding':
                    criterion = nn.CosineEmbeddingLoss()
                else:
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

        eval_contrastive(model, mode="val", batch_size=batch_size)

        print("EPOCH Loss ==================== ", str(epoch_loss))

    print("Training Complete!!")
