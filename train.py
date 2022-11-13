"""
@Author: Yiting CHEN
@Email: chenyiting@whu.edu.cn
"""

import torch

torch.cuda.empty_cache()

import time
from torch.utils.tensorboard import SummaryWriter
from dataset.dataloader import nyu_dataset
from utils.trainOption import TrainOptions
from model.fcrn import FCRN_wrapper
from utils.utils import setup_seed

if __name__ == "__main__":
    opt = TrainOptions().parse()
    setup_seed(opt.seed)
    writer = SummaryWriter()

    dataset = nyu_dataset(opt)
    FCRN_wrapper = FCRN_wrapper(opt, dataset, writer)

    if opt.pretrain:
        FCRN_wrapper.load_pretrained_model(opt.pretrained_model)

    for epoch in range(FCRN_wrapper.epoch, opt.n_epochs):
        epoch_start_time = time.time()

        FCRN_wrapper.train()

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        print('End of epoch %d / %d \t Time Taken: %d sec' % (FCRN_wrapper.epoch,
                                                              FCRN_wrapper.opt.n_epochs,
                                                              epoch_time))

        FCRN_wrapper.update_learning_rate()
        FCRN_wrapper.evaluate()

    torch.cuda.empty_cache()
    del FCRN_wrapper
