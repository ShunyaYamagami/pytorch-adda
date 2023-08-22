"""Adversarial adaptation to train target encoder."""

import os
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm

import params
from utils import make_variable
import logging
logger = logging.getLogger(__name__)

def train_tgt(src_encoder:nn.Module, tgt_encoder:nn.Module, critic,
              src_data_loader, tgt_data_loader, checkpoint_path, resume=''):
    """Train encoder for target domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    tgt_encoder.train()
    critic.train()

    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
                               lr=params.c_learning_rate,
                               betas=(params.beta1, params.beta2))
    optimizer_critic = optim.Adam(critic.parameters(),
                                  lr=params.d_learning_rate,
                                  betas=(params.beta1, params.beta2))
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    ####################
    # 2. train network #
    ####################

    best_acc = 0.0
    start_epoch = 0
    if resume:
        checkpoints = torch.load(checkpoint_path)
        src_encoder.load_state_dict(checkpoints['src_encoder'])
        tgt_encoder.load_state_dict(checkpoints['tgt_encoder'])
        optimizer_critic.load_state_dict(checkpoints['optimizer_critic'])
        optimizer_tgt.load_state_dict(checkpoints['optimizer_tgt'])
        start_epoch = checkpoints['epoch']
        best_acc = checkpoints['best_acc']

    for epoch in tqdm(range(start_epoch, params.num_epochs)):
        # zip source and target data pair
        accs = torch.Tensor()
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((images_src, _, domain_src), (images_tgt, _, domain_tgt)) in data_zip:
            ###########################
            # 2.1 train discriminator #
            ###########################

            # make images variable
            images_src = make_variable(images_src)
            images_tgt = make_variable(images_tgt)

            # zero gradients for optimizer
            optimizer_critic.zero_grad()

            # extract and concat features
            feat_src = src_encoder(images_src)
            feat_tgt = tgt_encoder(images_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)

            # predict on discriminator
            pred_concat = critic(feat_concat.detach())

            # prepare real and fake label
            # label_src = make_variable(torch.ones(feat_src.size(0)).long())
            # label_tgt = make_variable(torch.zeros(feat_tgt.size(0)).long())
            # label_concat = torch.cat((label_src, label_tgt), 0)
            label_concat = make_variable(torch.cat([domain_src.squeeze_(), domain_tgt.squeeze_()], dim=0))

            # compute loss for critic
            loss_critic = criterion(pred_concat, label_concat)
            loss_critic.backward()

            # optimize critic
            optimizer_critic.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            # acc = (pred_cls == label_concat).float().mean()
            accs = torch.cat([accs, (pred_cls == label_concat).long().cpu()])

            ############################
            # 2.2 train target encoder #
            ############################

            # zero gradients for optimizer
            optimizer_critic.zero_grad()
            optimizer_tgt.zero_grad()

            # extract and target features
            feat_tgt = tgt_encoder(images_tgt)

            # predict on discriminator
            pred_tgt = critic(feat_tgt)

            # prepare fake labels
            label_tgt = make_variable(torch.ones(feat_tgt.size(0)).long())

            # compute loss for target encoder
            loss_tgt = criterion(pred_tgt, label_tgt)
            loss_tgt.backward()

            # optimize target encoder
            optimizer_tgt.step()

        #######################
        # 2.3 print epoch info #
        #######################
        acc_epoch = torch.mean(accs)
        if best_acc < acc_epoch:
            best_acc = acc_epoch
            with open(os.path.join(params.log_dir, 'best.txt'), 'w') as f:
                f.write(f'Epoch: {epoch}  {best_acc:.3f}')
        if ((epoch + 1) % params.log_per_epoch == 0):
            logger.info("Epoch [{:4d}/{:4d}] \td_loss={:.3}\tg_loss={:.3} \tacc={:.3f}"
                    .format(epoch + 1,
                            params.num_epochs,
                            loss_critic,
                            loss_tgt,
                            acc_epoch))

        #############################
        # 2.4 save model parameters #
        #############################
        if ((epoch + 1) % params.log_ckpt_per_epoch == 0) or (epoch == params.num_epochs - 1):
            torch.save({
                'src_encoder': src_encoder.state_dict(),
                'tgt_encoder': tgt_encoder.state_dict(),
                'optimizer_critic': optimizer_critic.state_dict(),
                'optimizer_tgt': optimizer_tgt.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
                }, checkpoint_path)

    return tgt_encoder
