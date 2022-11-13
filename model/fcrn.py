"""
@Author: Yiting CHEN
@Email: chenyiting@whu.edu.cn
"""

import torch
import torch.nn as nn
from model.network import projection, skip, fast_up_projection, BerHuLoss
import os
from torch.autograd import Variable
import logging
from tqdm import tqdm
from utils.utils import create_dataloader, create_sampler, create_optimizer
from torch.optim import lr_scheduler
from torchvision.utils import make_grid
import math


class FCRN(nn.Module):
    """
    Implementation of paper "Deeper Depth Prediction with Fully Convolutional Residual Networks"
    https://arxiv.org/pdf/1606.00373.pdf
    """

    def __init__(self, input_c, output_c, size=None):
        super(FCRN, self).__init__()
        self.pred_size = size
        self.up_sample = nn.Upsample(size=size, mode='bilinear')

        self.conv_1 = nn.Conv2d(in_channels=input_c, out_channels=64, kernel_size=7, stride=2, padding=2)
        self.bn_1 = nn.BatchNorm2d(num_features=64)
        self.maxPooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

        self.projection_1 = projection(input_c=64, d1=64, d2=256, stride=1)
        self.skip_1 = skip(input_c=256, d1=64, d2=256)
        self.skip_2 = skip(input_c=256, d1=64, d2=256)
        self.projection_2 = projection(input_c=256, d1=128, d2=512, stride=2)
        self.skip_3 = skip(input_c=512, d1=128, d2=512)
        self.skip_4 = skip(input_c=512, d1=128, d2=512)
        self.skip_5 = skip(input_c=512, d1=128, d2=512)
        self.projection_3 = projection(input_c=512, d1=256, d2=1024, stride=2)

        self.skip_6 = skip(input_c=1024, d1=256, d2=1024)
        self.skip_7 = skip(input_c=1024, d1=256, d2=1024)
        self.skip_8 = skip(input_c=1024, d1=256, d2=1024)
        self.skip_9 = skip(input_c=1024, d1=256, d2=1024)
        self.skip_10 = skip(input_c=1024, d1=256, d2=1024)

        self.projection_4 = projection(input_c=1024, d1=512, d2=2048, stride=2)

        self.skip_11 = skip(input_c=2048, d1=512, d2=2048)
        self.skip_12 = skip(input_c=2048, d1=512, d2=2048)

        self.conv_2 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1)
        self.bn_2 = nn.BatchNorm2d(num_features=1024)

        self.up_proj_1 = fast_up_projection(input_c=1024, output_c=512)
        self.up_proj_2 = fast_up_projection(input_c=512, output_c=256)
        self.up_proj_3 = fast_up_projection(input_c=256, output_c=128)
        self.up_proj_4 = fast_up_projection(input_c=128, output_c=64)

        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=output_c, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv_1(x)  # exp: (batch_size, 3, 304, 228)
        out = self.bn_1(out)
        out = self.maxPooling(out)  # (batch_size, 64, 76, 57)

        out = self.projection_1(out)
        out = self.skip_1(out)
        out = self.skip_2(out)  # (batch_size, 256, 76, 57)

        out = self.projection_2(out)
        out = self.skip_3(out)
        out = self.skip_4(out)
        out = self.skip_5(out)  # (batch_size, 512, 38, 29)

        out = self.projection_3(out)
        out = self.skip_6(out)
        out = self.skip_7(out)
        out = self.skip_8(out)
        out = self.skip_9(out)
        out = self.skip_10(out)  # (batch_size, 1024, 19, 15)

        out = self.projection_4(out)
        out = self.skip_11(out)
        out = self.skip_12(out)  # (batch_size, 2048, 10, 8)

        out = self.conv_2(out)  # (batch_size, 1024, 10, 8)

        out = self.up_proj_1(out)  # (batch_size, 512, 20, 16)
        out = self.up_proj_2(out)  # (batch_size, 256, 40, 32)
        out = self.up_proj_3(out)  # (batch_size, 128, 80, 64)
        out = self.up_proj_4(out)  # (batch_size, 64, 160, 128)

        out = self.conv_3(out)  # (batch_size, 1, 160, 128)
        out = self.relu(out)
        out = self.up_sample(out)

        return out


class FCRN_wrapper:
    def __init__(self, opt, dataset, writer=None):
        # initialize settings and device
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(self.opt.gpu_ids[0])) if self.opt.gpu_ids else torch.device('cpu')

        # initialize tensorboard
        self.writer = writer

        # initialize model
        self.model = FCRN(self.opt.input_c, self.opt.output_c, size=self.opt.load_size)
        self.model.to(self.device)
        self.weights_init(self.model)
        self.set_requires_grad()

        # training parameters
        self.epoch = 0
        self.iter = 0
        self.train_loss = None
        self.val_loss = None

        # prepare dataset
        self.train_dataset = None
        self.val_dataset = None
        self.initialize_dataset(dataset)

        # parameters optimization
        # self.criterion = nn.MSELoss()
        self.criterion = BerHuLoss()

        self.criterion.to(self.device)
        self.optimizer = create_optimizer(type=self.opt.optimizer, model=self.model, lr=self.opt.lr)
        self.scheduler = self.get_scheduler()

        # save
        self.best_loss = None

        # training data
        self._x = None
        self._y = None
        self._pred = None

    def set_requires_grad(self, requires_grad=True):
        """ Set require grad """
        for param in self.model.parameters():
            param.requires_grad = requires_grad

    def initialize_dataset(self, dataset):
        train_sampler, val_sampler = create_sampler(len(dataset), self.opt.split)
        self.train_dataset, self.val_dataset = create_dataloader(dataset,
                                                                 self.opt.batch_size,
                                                                 self.opt.num_threads,
                                                                 train_sampler, val_sampler)

    def load_pretrained_model(self, pth_path):
        """ continues training with previous model """
        print(' loading pretrained model from {}'.format(pth_path))
        state_dict = torch.load(pth_path, map_location=str(self.device))
        self.model.load_state_dict(state_dict['model'])
        self.epoch = state_dict['epoch'] + 1
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if self.opt.optimizer == "adam":
            self.optimizer.param_groups[0]['capturable'] = True  # a little bug for adam in pytorch
        self.iter = int(self.epoch * len(self.train_dataset) / self.opt.batch_size)
        print(' starting training from epoch: {}, iteration: {}'.format(self.epoch, self.iter))


    def save_checkpoint(self):
        checkpoint_filename = os.path.join(self.opt.checkpoints_dir, 'best_model' + '.pth')
        if torch.cuda.is_available():
            torch.save({'model': self.model.cpu().state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'epoch': self.epoch},
                       checkpoint_filename)
        else:
            torch.save({'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'epoch': self.epoch},
                       checkpoint_filename)
        self.model.to(self.device)

    def train(self, vis=True):
        """ Training given dataset for one epoch """
        self.epoch += 1
        print('epoch %d training' % self.epoch)
        self.model.train()
        epoch_loss = 0
        pbar = tqdm(total=len(self.train_dataset))

        for data in self.train_dataset:
            self.iter += 1

            self._x = data['rgb'].to(self.device)
            self._y = data['depth'].to(self.device)
            self._pred = self.model(self._x)
            self.train_loss = self.criterion(self._pred, self._y)

            self.optimizer.zero_grad()
            self.train_loss.backward()
            self.optimizer.step()
            # for x in self.optimizer.param_groups[0]['params']:
            #     print(x.grad)

            loss_value = self.train_loss
            epoch_loss += loss_value

            self.writer.add_scalar("Loss", loss_value, self.iter)

            pbar.update(1)
        if vis:
            rgb_imgs = make_grid(self._x)
            self.writer.add_image('input_rgb', rgb_imgs, 0)
            pred_depth = make_grid(self._pred)
            self.writer.add_image('pred_depth', pred_depth, 0)
            gt_depth = make_grid(self._y)
            self.writer.add_image('gt_depth', gt_depth, 0)

        epoch_loss = epoch_loss / len(self.train_dataset)
        logging.info(epoch_loss)

    def evaluate(self):
        """ Model Evaluation """
        print('evaluating model ...')
        total_loss = 0

        self.model.eval()
        for data in self.val_dataset:
            self._x = data['rgb'].to(self.device)
            self._y = data['depth'].to(self.device)
            with torch.no_grad():
                self._pred = self.model(self._x)
            self.val_loss = self.criterion(self._pred, self._y)
            total_loss += self.val_loss

        epoch_loss = total_loss / len(self.val_dataset)
        if self.best_loss is None:
            self.best_loss = epoch_loss
            return epoch_loss
        else:
            if epoch_loss < self.best_loss:
                print('saving the model at the end of epoch %d, iters %d' % (self.epoch, self.iter))
                self.save_checkpoint()
                self.best_loss = epoch_loss
                return epoch_loss
            else:
                return epoch_loss

    def get_scheduler(self):
        """Get scheduler according to utils"""
        if self.opt.lr_policy == 'linear':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch - self.opt.n_epochs) / float(self.opt.n_epochs_decay + 1)
                return lr_l

            scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)
        elif self.opt.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.opt.n_epochs_decay, gamma=0.1)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', self.opt.lr_policy)
        return scheduler


    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_visual_images(self):
        """Get current rgb, ground truth and prediction image for summary writer"""
        assert self._x.shape[0] == self._pred.shape[0]

        num_figs = self._x.shape[0] if self._x.shape[0] <= 4 else 4
        output_figs = []
        for i in range(num_figs):
            output_figs[i] = [self._x[i], pred[i]]

        return output_figs

    def weights_init(self, m):
        """ Initialize filters with Gaussian random weights """
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sim_data = Variable(torch.rand(4, 3, 360, 240)).to(device)
    net = FCRN(input_c=3, output_c=1, size=(360, 240)).to(device)
    pred = net(sim_data)
    print(pred.shape)
