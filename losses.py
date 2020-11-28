import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()



class SimClassificationLoss(nn.Module):
    """
    Classification Loss
    Takes concatenated embedding and classifies if they are similar 1 or dissimilar 0
    """
    def __init__(self, pos_weight=1):
        super(SimClassificationLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.pos_weight = pos_weight

    def forward(self, output, labels):
        # output is Nx1
        # labels is Nx1
        full_loss = self.loss(output, labels)
        # full_loss is Nx1
        loss = self.pos_weight*full_loss[(labels==1).nonzero()[:,0]].sum() + full_loss[(labels==0).nonzero()[:,0]].sum()
        loss = loss/output.shape[0]
        return loss