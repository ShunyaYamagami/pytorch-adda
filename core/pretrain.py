"""Pre-train encoder and classifier for source dataset."""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

import params
from utils import make_variable, save_model
import logging
logger = logging.getLogger(__name__)

def train_src(encoder, classifier, data_loader):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=params.c_learning_rate,
        betas=(params.beta1, params.beta2))
    criterion = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################

    for epoch in tqdm(range(params.num_epochs_pre)):
        train_src_step(epoch, encoder, classifier, data_loader, optimizer, criterion)

        # eval model on test set
        if ((epoch + 1) % params.eval_step_pre == 0):
            eval_src(encoder, classifier, data_loader)

        # save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(encoder, "ADDA-source-encoder-{}.pt".format(epoch + 1))
            save_model(
                classifier, "ADDA-source-classifier-{}.pt".format(epoch + 1))

    # # save final model
    save_model(encoder, "ADDA-source-encoder-final.pt")
    save_model(classifier, "ADDA-source-classifier-final.pt")

    return encoder, classifier


def train_src_step(epoch, encoder, classifier, data_loader, optimizer, criterion):
    for step, (images, labels, _) in enumerate(data_loader):
        # make images and labels variable
        images = make_variable(images)
        labels = make_variable(labels.squeeze_())

        # zero gradients for optimizer
        optimizer.zero_grad()

        # compute loss for critic
        preds = classifier(encoder(images))
        # preds = F.softmax(preds, dim=-1).max()
        loss = criterion(preds, labels)

        # optimize source classifier
        loss.backward()
        optimizer.step()

    # print step info
    if ((epoch + 1) % params.log_per_epoch == 0):
        logger.info("Epoch [{}/{}]: loss={}"
                .format(epoch + 1,
                        params.num_epochs_pre,
                        loss))


def eval_src(encoder, classifier, data_loader):
    """Evaluate classifier for source domain."""
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
            # labels = make_variable(labels)
            labels = make_variable(labels.squeeze_())

            preds = encoder(images)
            preds = classifier(preds)
            loss += criterion(preds, labels)

            pred_cls = preds.data.max(1)[1]
            acc += pred_cls.eq(labels.data).cpu().sum()

        loss /= len(data_loader)
        acc /= len(data_loader.dataset)

    logger.info("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
