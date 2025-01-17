"""Test script to classify target data."""

import torch
import torch.nn as nn

from utils import make_variable


def eval_tgt(encoder, classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0.0
    acc = 0.0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    with torch.no_grad():
        for (images, labels, _) in data_loader:
            images = make_variable(images, volatile=True)
            labels = make_variable(labels).squeeze_()

            preds = classifier(encoder(images))
            # loss += criterion(preds, labels).data[0]
            loss += criterion(preds, labels)

            pred_cls = preds.data.max(1)[1]
            acc += pred_cls.eq(labels.data).cpu().sum()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    return acc

    # print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
