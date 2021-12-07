import os
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import scipy.io as scio
import argparse
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import random
import h5py
import re
# from test import test
from math import exp
from dataloader_CRAFTS import CRAFTSDataset
from mseloss_CRAFTS import Maploss
from collections import OrderedDict
# from eval.script import getresult
from PIL import Image
from torchvision.transforms import transforms
from craft import CRAFT
from torch.autograd import Variable
from torch.multiprocessing import Pool, Process, set_start_method
from tqdm import tqdm
import datetime
from torchutil import *



random.seed(42)
parser = argparse.ArgumentParser(description='CRAFT reimplementation')
parser.add_argument('--cuda_device', default='0,2,3',type=str, help='assign cuda devices')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float, #1.2768e-5
                    help='initial learning rate')
parser.add_argument('--momentum', default=1e-3, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float, #5e-4
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.99, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--batch_size', default=6, type=int)

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def adjust_learning_rate(optimizer, gamma, step):
    lr = args.lr * (0.8 ** step)
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__' :
    print('TwinReader Data Loading...')
    sy_time=time.time()
    dt = datetime.datetime.today()
    weight_save_dir = './saved_model/'+str(dt.month)+str(dt.day)
    os.makedirs(weight_save_dir, exist_ok = True)
    torch.multiprocessing.set_start_method('spawn')

    weight_save_dir += '/'+str(len([l for l in os.listdir(weight_save_dir) if l !='.ipynb_checkpoints']))
    os.makedirs(weight_save_dir, exist_ok = True)
    
    
    real_img_dir = './data/' 
    net = CRAFT()
    
    net.conv_cls[-1] = nn.Conv2d(16,4, kernel_size = 1) # 마지막 레이어만 4판으로 교체
    init_weights(net.conv_cls[-1].modules())

    
    net.load_state_dict(copyStateDict(torch.load('./saved_model/CRAFT_best.pth')))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    cudnn.benchmark = True
    
    real_time=time.time()
    
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net).to(device)
    net.train()
    
    real_dataloader = CRAFTSDataset( data_folder = real_img_dir, use_net = False, watershed_on = False, delimiter = '\t')
    real_data_loader = torch.utils.data.DataLoader(
        real_dataloader,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True)
    print("Real Data loading time::", time.time()-real_time)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = Maploss()
    step_index = 0

    loss_time = 0
    loss_value = 0
    compare_loss = 1
    print("Training ....")
    for epoch in range(3000):
        train_time_st = time.time()
        loss_value = 0
        if epoch % 50 == 0 and epoch != 0:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        st = time.time()

        for index, (real_images, real_gh_label, real_gah_label, real_mask, real_ori_x, real_ori_y) in tqdm(enumerate(real_data_loader)):
            images = real_images #torch.cat((syn_images,real_images), 0)
            gh_label = real_gh_label # torch.cat((syn_gh_label, real_gh_label), 0)
            gah_label = real_gah_label #torch.cat((syn_gah_label, real_gah_label), 0)
            mask = real_mask #torch.cat((syn_mask, real_mask), 0)
            images = Variable(images.type(torch.FloatTensor)).to(device)
            gh_label = gh_label.type(torch.FloatTensor)
            gah_label = gah_label.type(torch.FloatTensor)
            gh_label = Variable(gh_label).to(device)
            gah_label = Variable(gah_label).to(device)
            mask = mask.type(torch.FloatTensor)
            mask = Variable(mask).to(device)
            real_ori_x = real_ori_x.type(torch.FloatTensor)
            real_ori_x = Variable(real_ori_x).to(device)
            real_ori_y = real_ori_y.type(torch.FloatTensor)
            real_ori_y = Variable(real_ori_y).to(device)
            
            out, feature = net(images)
            optimizer.zero_grad()
    
            out1 = out[:, :, :, 0].to(device)
            out2 = out[:, :, :, 1].to(device)
            out3 = out[:, :, :, 2].to(device)
            out4 = out[:, :, :, 3].to(device)

            loss = criterion(gh_label, gah_label, real_ori_x, real_ori_y, out1, out2, out3, out4, mask) 
            loss.backward()
            optimizer.step()
            loss_value += loss.item()
            if index % 2 == 0 and index > 0:
                et = time.time()
                print('epoch {}:({}/{}) batch || training time for 2 batch {} || training loss {}'.format(epoch, index, len(real_data_loader), et-st, loss_value))
                loss_time = 0
                loss_value = 0
                st = time.time()
        print('Saving state, iter:', epoch)
        torch.save(net.state_dict(), os.path.join(str(weight_save_dir), 'CRAFT' + repr(epoch) + '.pth')) # net.module.state_dict()