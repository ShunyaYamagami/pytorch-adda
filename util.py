import numpy as np
import os
import torch
from scipy.special import softmax
from models import models, resnet, alexnet

import logging
logger = logging.getLogger(__name__)



def get_model(config, use_weights=True):
    feature_extractor, class_classifier, domain_classifier = resnet.get_models(config.num_classes, config.num_domains, use_weights, out_dim=64)
    
    if config.resume:
        feature_extractor.load_state_dict(torch.load(os.path.join(config.checkpoint_dir, "feature_extractor_latest.tar")))
        class_classifier.load_state_dict(torch.load(os.path.join(config.checkpoint_dir, "class_classifier_latest.tar")))
        domain_classifier.load_state_dict(torch.load(os.path.join(config.checkpoint_dir, "domain_classifier_latest.tar")))

    return feature_extractor.cuda(), class_classifier.cuda(), domain_classifier.cuda()

