import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
from PIL import Image
import torch
import random
import torch.utils.data


def save_depth_img(input_tensor, file_name):
    depth_npy = input_tensor.cpu().numpy()
    normalizer = mpl.colors.Normalize(vmin=depth_npy.min(), vmax=np.percentile(depth_npy, 95))
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma_r')
    mapped_img = (mapper.to_rgba(depth_npy)[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(mapped_img)
    img.save(file_name)


def save_rgb_image(input_tensor, file_name):
    rgb_npy = input_tensor.cpu().numpy()
    img = Image.fromarray(rgb_npy.astype('uint8'))
    img.save(file_name)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def create_sampler(length, split, shuffle=True):
    indices = list(range(length))
    split = int(np.floor(split * length))
    if shuffle:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    return train_sampler, val_sampler


def create_dataloader(dataset, batch_size, num_workers, train_sampler, val_sampler):
    train_data = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             sampler=train_sampler)
    val_data = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           num_workers=num_workers,
                                           sampler=val_sampler)

    return train_data, val_data


def create_optimizer(**kwargs):
    optimizer = None
    if kwargs['type'] == 'adam':
        optimizer = torch.optim.Adam(kwargs['model'].parameters(), lr=kwargs['lr'])

    if kwargs['type'] == 'sgd':
        optimizer = torch.optim.SGD(kwargs['model'].parameters(), lr=kwargs['lr'], momentum=kwargs['monentum'])

    if optimizer is not None:
        return optimizer
    else:
        raise Exception("pls set the optimizer correctly")


