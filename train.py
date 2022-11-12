"""
@Author: Yiting CHEN
@Email: chenyiting@whu.edu.cn
"""

import torch

torch.cuda.empty_cache()

import time
from torch.utils.tensorboard import SummaryWriter
from dataloader import nyu_dataset
from trainOption import TrainOptions
from model import FCRN_wrapper
from utils import setup_seed


if __name__ == "__main__":
    opt = TrainOptions().parse()
    setup_seed(opt.seed)
    writer = SummaryWriter()

    dataset = nyu_dataset(opt)
    FCRN_wrapper = FCRN_wrapper(opt, dataset, writer)

    for epoch in range(opt.n_epochs):
        epoch_start_time = time.time()

        FCRN_wrapper.train()

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        print('End of epoch %d / %d \t Time Taken: %d sec' % (FCRN_wrapper.epoch,
                                                              FCRN_wrapper.opt.n_epochs,
                                                              epoch_time))

        FCRN_wrapper.update_learning_rate()
        FCRN_wrapper.evaluate()