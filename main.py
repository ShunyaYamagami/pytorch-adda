"""Main script for ADDA."""

import params
from core import eval_src, eval_tgt, train_src, train_tgt
from models import Discriminator, LeNetClassifier, LeNetEncoder
from utils import get_data_loader, init_model, init_random_seed
from concurrent.futures import ThreadPoolExecutor


import logging
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)

    # load dataset
    src_data_loader = get_data_loader(params.parent, params.labeled_filename, params.image_size)
    tgt_data_loader = get_data_loader(params.parent, params.unlabeled_filename, params.image_size)
    # src_data_loader_eval = get_data_loader(params.parent, params.test0_filename, params.image_size, train=False)
    # tgt_data_loader_eval = get_data_loader(params.parent, params.test1_filename, params.image_size, train=False)

    # load models
    src_encoder = init_model(net=LeNetEncoder(img_size=params.image_size),
                             restore=params.src_encoder_restore)
    src_classifier = init_model(net=LeNetClassifier(params.num_classes),
                                restore=params.src_classifier_restore)
    tgt_encoder = init_model(net=LeNetEncoder(img_size=params.image_size),
                             restore=params.tgt_encoder_restore)
    critic = init_model(Discriminator(input_dims=params.d_input_dims,
                                      hidden_dims=params.d_hidden_dims,
                                      output_dims=params.d_output_dims),
                        restore=params.d_model_restore)

    if not params.resume:
        # train source model
        logger.info("=== Training classifier for source domain ===")
        logger.info(">>> Source Encoder <<<")
        logger.info(src_encoder)
        logger.info(">>> Source Classifier <<<")
        logger.info(src_classifier)

        # if not (src_encoder.restored and src_classifier.restored and
        #         params.src_model_trained):
        # src_encoder, src_classifier = train_src(
        #     src_encoder, src_classifier, src_data_loader)

        # eval source model
        logger.info("=== Evaluating classifier for source domain ===")
        # eval_src(src_encoder, src_classifier, src_data_loader_eval)

        # train target encoder by GAN
        logger.info("=== Training encoder for target domain ===")
        logger.info(">>> Target Encoder <<<")
        logger.info(tgt_encoder)
        logger.info(">>> Critic <<<")
        logger.info(critic)
    
        # init weights of target encoder with those of source encoder
        # if not tgt_encoder.restored:
        #     tgt_encoder.load_state_dict(src_encoder.state_dict())
    else:
        logger.info(f'''
            === Resuming from "{params.resume}" ===
            parent  = {params.parent}
            dset    = {params.dset}
            task    = {params.task}
            log_dir = {params.log_dir}
            checkpoint_path = {params.checkpoint_path}
        ''')


    # if not (tgt_encoder.restored and critic.restored and
    #         params.tgt_model_trained):
    tgt_encoder = train_tgt(src_encoder, tgt_encoder, src_classifier, critic,
                            src_data_loader, tgt_data_loader, params.checkpoint_path, params.resume)

    # eval target encoder on test set of target dataset
    logger.info("=== Evaluating classifier for encoded target domain ===")
    del src_data_loader, tgt_data_loader
    tgt_data_loader_eval = get_data_loader(params.parent, params.test1_filename, params.image_size, train=False)
    logger.info(">>> source only <<<")
    eval_tgt(src_encoder, src_classifier, tgt_data_loader_eval)
    logger.info(">>> domain adaption <<<")
    eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval)
