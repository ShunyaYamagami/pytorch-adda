"""Pre-train encoder and classifier for source dataset."""

import os

import torch
import torch.nn as nn
import torch.optim as optim

import params
from utils import make_variable


def train_src(net, data_loader):
    """Train classifier for source domain."""
    print("Classifier for source domain:")
    print(net)

    optimizer = optim.Adam(net.parameters(),
                           lr=params.c_learning_rate,
                           betas=(params.beta1, params.beta2))
    criterion = nn.NLLLoss()

    for epoch in range(params.num_epochs):
        net.train()
        for step, (images, labels) in enumerate(data_loader):
            images = make_variable(images)
            labels = make_variable(labels.squeeze_())

            optimizer.zero_grad()

            _, preds = net(images)
            loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()

            if ((step + 1) % params.log_step == 0):
                print("Epoch [{}/{}] Step [{}/{}]: loss={}"
                      .format(epoch + 1,
                              params.num_epochs,
                              step + 1,
                              len(data_loader),
                              loss.data[0]))

        if ((epoch + 1) % params.save_step == 0):
            if not os.path.exists(params.model_root):
                os.makedirs(params.model_root)
            torch.save(net.state_dict(), os.path.join(
                params.model_root,
                "classifier_src-{}.pt".format(epoch + 1)))
