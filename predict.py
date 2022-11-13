"""
@Author: Yiting CHEN
@Time: 2022/11/9 上午12:18
@Email: chenyiting@whu.edu.cn
version: python 3.9
Created by PyCharm
"""
import random
from model.fcrn import FCRN
import torch
from dataset.dataloader import nyu_dataset
from utils.trainOption import TrainOptions
import matplotlib.pyplot as plt


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    net = FCRN(input_c=3, output_c=1, size=(228, 304))
    test_model_pth = "checkpoints/best_model_25.pth"
    state_dict = torch.load(test_model_pth, map_location=str(device))
    net.load_state_dict(state_dict['model'])
    net.to(device)
    net.eval()

    opt = TrainOptions().parse()  # get utils
    data_loader = nyu_dataset(opt)
    data_loader.set_test_mode()
    idx = random.randint(0, len(data_loader))
    data = data_loader[idx]

    plt.imshow(data['rgb'].permute(1, 2, 0))  # input rgb
    plt.show()
    plt.imshow(data['depth'].permute(1, 2, 0))  # gt depth
    plt.show()
    with torch.no_grad():
        pred = net(data['rgb'].unsqueeze(0).to(device))
    pred_img = pred[0].cpu().numpy().transpose(1, 2, 0)  # depth prediction
    plt.imshow(pred_img)
    plt.show()
