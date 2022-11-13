import argparse
import os
import torch


class TrainOptions:
    """This class includes training options.
    """

    def __init__(self):
        self.parser = None
        self.opt = None

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""

        # basic parameters
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--seed', default=999, type=int, help='random seed')

        # training parameters
        parser.add_argument('--name', type=str, default='depth_estimation',
                            help='name of the experiment. It decides where to store samples and models')

        # dataset parameters
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')

        parser.add_argument('--split', default=0.9, type=float, help='training data / ALL data, the rest is val data ')

        parser.add_argument('--dataroot', default='nyu_data/data', help='path to images')
        parser.add_argument('--train', default='nyu2_train', help='path to training data')
        parser.add_argument('--test', default='nyu2_test', help='path to testing data')

        parser.add_argument('--input_c', type=int, default=3,
                            help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_c', type=int, default=1,
                            help='# of output image channels: 3 for RGB and 1 for grayscale')

        parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
        parser.add_argument('--load_size', type=tuple, default=(228, 304), help='scale images to this size')

        parser.add_argument('--rand_zoom', type=int, default=1,
                            help='random zoom the image in to scale [0.6, 1]')

        # we do not rotate at this time
        # parser.add_argument('--rand_rotate', type=int, default=1,
        #                     help='random rotate the image between angle [-30, 30]')

        parser.add_argument('--if_flip', type=int, default=1,
                            help='random flip the image')

        # training parameters
        parser.add_argument('--optimizer', type=str, default='adam', help='adam | sgd')
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=15,
                            help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear',
                            help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50,
                            help='multiply by a gamma every lr_decay_iters iterations')

        # network saving and loading parameters
        parser.add_argument('--pretrain', type=int, default=0, help='if resume training')
        parser.add_argument('--pretrained_model', type=str, default='checkpoints/best_model.pth',
                            help='resume training from previous best checkpoint')

        return parser

    def gather_options(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser = self.initialize(parser)
        return parser.parse_args()

    def parse(self, save_opt=True):
        """Parse our options"""
        opt = self.gather_options()
        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if not os.path.exists(expr_dir):
            mkdirs(expr_dir)
        if save_opt:
            message = ''
            message += '----------------- Options ---------------\n'
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
            message += '----------------- End -------------------'
            print(message)

            # save to the disk
            expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
            mkdirs(expr_dir)
            file_name = os.path.join(expr_dir, '{}_opt.txt'.format('train'))
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')
        self.opt = opt
        return self.opt


def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    opt = TrainOptions().parse()  # get training options
